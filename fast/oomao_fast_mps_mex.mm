// Cropped complex DFT through owned Metal/MPS state. Physical arrays stay CPU double.
#include "mex.h"
#include "oomao_fast_threads.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static const char* shaders=R"METAL(
#include <metal_stdlib>
using namespace metal;
// The first product has real/imaginary row blocks, each crop x (n*directions).
kernel void transposeFirst(device const float* first [[buffer(0)]],device float* packed [[buffer(1)]],
 constant uint4& size [[buffer(2)]],uint k [[thread_position_in_grid]]){
 uint n=size.x,c=size.y,d=size.z;if(k>=d*2*n*c)return;
 uint col=k%c,row=(k/c)%(2*n),direction=k/(c*2*n),part=row/n,x=row%n;
 packed[k]=first[(part*c+col)*(n*d)+direction*n+x];
}
// E*first^T is the transpose of the desired field: row-major storage is MATLAB order.
kernel void packResult(device const float* blocks [[buffer(0)]],device float2* out [[buffer(1)]],
 constant uint4& size [[buffer(2)]],uint k [[thread_position_in_grid]]){
 uint c=size.y,d=size.z,pixels=c*c;if(k>=d*pixels)return;
 uint direction=k/pixels,i=k%pixels;out[k]=float2(blocks[direction*2*pixels+i],blocks[direction*2*pixels+pixels+i]);
}
)METAL";

struct Buffer {
 __strong id<MTLBuffer> value=nil;
 size_t bytes=0;
 void ensure(id<MTLDevice> device,size_t required){
  if(required<=bytes)return;
  if(required>512ull*1024*1024)throw std::runtime_error("MPS workspace buffer exceeds 512 MiB.");
  id<MTLBuffer> next=[device newBufferWithLength:required options:MTLResourceStorageModeShared];
  if(!next)throw std::runtime_error("Metal buffer allocation failed.");value=next;bytes=required;
 }
};
static MPSMatrix* matrix(id<MTLBuffer> buffer,size_t offset,size_t rows,size_t columns){
 MPSMatrixDescriptor* descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:columns*4 dataType:MPSDataTypeFloat32];
 MPSMatrix* result=[[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:descriptor];
 if(!result)throw std::runtime_error("MPS matrix allocation failed.");return result;
}
static MPSMatrixMultiplication* product(id<MTLDevice> device,size_t rows,size_t columns,size_t inner){
 MPSMatrixMultiplication* result=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO
  resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
 if(!result)throw std::runtime_error("MPS multiplication allocation failed.");return result;
}
static MPSMatrix* batchedMatrix(id<MTLBuffer> buffer,size_t rows,size_t columns,size_t count){
 MPSMatrixDescriptor* descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns matrices:count
  rowBytes:columns*4 matrixBytes:rows*columns*4 dataType:MPSDataTypeFloat32];
 MPSMatrix* result=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
 if(!result)throw std::runtime_error("MPS batched matrix allocation failed.");return result;
}
struct Roots {Buffer block;std::map<size_t,Buffer> repeated;};
struct Slot {
 Buffer input,first,intermediate,second,output;
 __strong id<MTLCommandBuffer> command=nil;
 size_t base=0,directions=0;
};
struct Context {
 __strong id<MTLDevice> device=nil;
 __strong id<MTLCommandQueue> queue=nil;
 __strong id<MTLComputePipelineState> transpose=nil,pack=nil;
 std::array<Slot,2> slots;
 static constexpr size_t chunk=32,tile=32;
 std::map<std::array<int,4>,std::unique_ptr<Roots>> roots;
 size_t threads;
 explicit Context(size_t count):threads(count){
  device=MTLCreateSystemDefaultDevice();
  if(!device||!MPSSupportsMTLDevice(device))throw std::runtime_error("A supported Metal/MPS device is required.");
  queue=[device newCommandQueue];if(!queue)throw std::runtime_error("Metal queue allocation failed.");
  MTLCompileOptions* options=[[MTLCompileOptions alloc] init];options.fastMathEnabled=NO;
  NSError* error=nil;
  id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:shaders] options:options error:&error];
  if(!library)throw std::runtime_error(error?error.localizedDescription.UTF8String:"Metal shader compilation failed.");
  transpose=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"transposeFirst"] error:&error];
  pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packResult"] error:&error];
  if(!transpose||!pack)throw std::runtime_error(error?error.localizedDescription.UTF8String:"Metal pipeline creation failed.");
 }
 void encode(id<MTLCommandBuffer> command,id<MTLComputePipelineState> pipeline,Buffer& source,Buffer& destination,
             const std::array<uint32_t,4>& size,size_t count){
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  if(!encoder)throw std::runtime_error("Metal command encoder allocation failed.");
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:source.value offset:0 atIndex:0];
  [encoder setBuffer:destination.value offset:0 atIndex:1];[encoder setBytes:size.data() length:sizeof(size) atIndex:2];
  NSUInteger width=std::min(NSUInteger(256),pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];[encoder endEncoding];
 }
};
static std::map<uint64_t,std::unique_ptr<Context>> contexts;
static uint64_t nextHandle=1;
static void cleanup(){contexts.clear();oomaoPoolCleanup();}
static uint64_t handle(const mxArray* input){
 if(!mxIsUint64(input)||mxIsComplex(input)||mxGetNumberOfElements(input)!=1)throw std::runtime_error("Owned uint64 MPS handle required.");
 return *static_cast<const uint64_t*>(mxGetData(input));
}
static int integer(const mxArray* input){
 if(!mxIsDouble(input)||mxIsComplex(input)||mxIsSparse(input)||mxGetNumberOfElements(input)!=1)throw std::runtime_error("Integer scalar required.");
 double value=mxGetScalar(input);if(!std::isfinite(value)||value<0||value>16384||value!=std::floor(value))throw std::runtime_error("Integer outside 0..16384.");
 return int(value);
}
struct Pupil {
 struct Input {const double *amplitude,*phase;const mxLogical* mask;size_t na,np;};
 int n;std::vector<Input> inputs;
 Pupil(const mxArray* amplitudes,const mxArray* phases,const mxArray* masks){
  size_t count=mxGetNumberOfElements(masks);
  if(!mxIsCell(amplitudes)||!mxIsCell(phases)||!mxIsCell(masks)||!count||count>16384||
   mxGetNumberOfElements(amplitudes)!=count||mxGetNumberOfElements(phases)!=count)throw std::runtime_error("Matching pupil cell arrays required.");
  const mxArray* first=mxGetCell(masks,0);if(!first||mxGetNumberOfDimensions(first)!=2||mxGetM(first)<2||mxGetM(first)>16384||mxGetM(first)!=mxGetN(first))throw std::runtime_error("Square pupil mask required.");
  n=int(mxGetM(first));size_t pixels=size_t(n)*n;inputs.reserve(count);
  for(size_t k=0;k<count;++k){
   const mxArray *a=mxGetCell(amplitudes,k),*p=mxGetCell(phases,k),*m=mxGetCell(masks,k);
   if(!a||!p||!m||!mxIsDouble(a)||!mxIsDouble(p)||mxIsComplex(a)||mxIsComplex(p)||mxIsSparse(a)||mxIsSparse(p)||
    !mxIsLogical(m)||mxIsSparse(m)||mxGetNumberOfDimensions(m)!=2||mxGetM(m)!=size_t(n)||mxGetN(m)!=size_t(n))throw std::runtime_error("Full double amplitude/phase and matching logical mask required.");
   size_t na=mxGetNumberOfElements(a),np=mxGetNumberOfElements(p);
   if((na!=1&&(mxGetM(a)!=size_t(n)||mxGetN(a)!=size_t(n)||mxGetNumberOfDimensions(a)!=2))||
    (np!=1&&(mxGetM(p)!=size_t(n)||mxGetN(p)!=size_t(n)||mxGetNumberOfDimensions(p)!=2)))throw std::runtime_error("Scalar or pupil-sized amplitude/phase required.");
   inputs.push_back({mxGetDoubles(a),mxGetDoubles(p),mxGetLogicals(m),na,np});
  }
 }
};
static mxArray* diffract(Context& context,const mxArray* field,const mxArray* phasor,int nf,int start,int crop,const Pupil* pupil=nullptr){
 if(!pupil&&(!mxIsSingle(field)||mxIsSparse(field)||mxGetNumberOfDimensions(field)!=2))throw std::runtime_error("MPS diffraction requires full real/complex single pupil pages.");
 const int n=pupil?pupil->n:int(mxGetM(field));
 if(n<2||n>16384||(!pupil&&mxGetN(field)%n)||nf<n||crop<1||start+crop>nf)throw std::runtime_error("Invalid MPS pupil/FFT/crop geometry.");
 size_t directions=pupil?pupil->inputs.size():mxGetN(field)/n;if(!directions||directions>16384)throw std::runtime_error("Unsupported pupil-page count.");
 bool shifted=mxIsEmpty(phasor);int shift=shifted?(nf+1)/2:0;
 if(!shifted&&(!mxIsDouble(phasor)||!mxIsComplex(phasor)||mxIsSparse(phasor)||mxGetM(phasor)!=size_t(n)||mxGetN(phasor)!=size_t(n)))throw std::runtime_error("Empty or complex double pupil-sized phasor required.");
 auto key=std::array<int,4>{n,nf,start+shift,crop};auto found=context.roots.find(key);
 if(found==context.roots.end()){
  if(context.roots.size()>=16)context.roots.erase(context.roots.begin());
  auto roots=std::make_unique<Roots>();roots->block.ensure(context.device,size_t(2*crop)*2*n*4);
  float* values=static_cast<float*>(roots->block.value.contents);
  for(int row=0;row<crop;++row)for(int x=0;x<n;++x){
   size_t index=(size_t((start+shift+row)%nf)*x)%nf;
   double angle=-2*3.141592653589793238462643383279502884*double(index)/nf;
   float re=float(std::cos(angle)),im=float(std::sin(angle));
   values[size_t(row)*2*n+x]=re;values[size_t(row)*2*n+n+x]=-im;
   values[size_t(crop+row)*2*n+x]=im;values[size_t(crop+row)*2*n+n+x]=re;
  }
  found=context.roots.emplace(key,std::move(roots)).first;
 }
 MPSMatrix* roots=matrix(found->second->block.value,0,2*crop,2*n);
 const mxComplexSingle* complex=(!pupil&&mxIsComplex(field))?mxGetComplexSingles(field):nullptr;
 const float* real=(pupil||complex)?nullptr:mxGetSingles(field);
 const mxComplexDouble* phase=shifted?nullptr:mxGetComplexDoubles(phasor);
 std::unique_ptr<mxArray,decltype(&mxDestroyArray)> result(mxCreateUninitNumericMatrix(crop,size_t(crop)*directions,mxSINGLE_CLASS,mxCOMPLEX),mxDestroyArray);
 if(!result)throw std::runtime_error("MATLAB output allocation failed.");

 // Any exception must drain submitted work before input/output storage can die.
 struct Pending {
  Context& context;
  ~Pending(){for(auto& slot:context.slots)if(slot.command){if(slot.command.status>=MTLCommandBufferStatusCommitted)[slot.command waitUntilCompleted];slot.command=nil;}}
 } pending{context};
 auto finish=[&](Slot& slot){
  if(!slot.command)return;
  [slot.command waitUntilCompleted];
  if(slot.command.status!=MTLCommandBufferStatusCompleted)
   throw std::runtime_error(slot.command.error?slot.command.error.localizedDescription.UTF8String:"Metal command failed.");
  std::memcpy(mxGetComplexSingles(result.get())+slot.base*crop*crop,
              slot.output.value.contents,slot.directions*crop*crop*sizeof(mxComplexSingle));
  slot.command=nil;
 };
 size_t batch=0;
 for(size_t base=0;base<directions;base+=context.chunk,++batch){
  auto& slot=context.slots[batch%2];finish(slot);
  size_t d=std::min(context.chunk,directions-base),columns=size_t(n)*d;
  slot.input.ensure(context.device,size_t(2*n)*columns*4);slot.first.ensure(context.device,size_t(2*crop)*columns*4);
  slot.intermediate.ensure(context.device,d*2*n*crop*4);slot.second.ensure(context.device,d*2*crop*crop*4);
  slot.output.ensure(context.device,d*crop*crop*sizeof(mxComplexSingle));
  float* packed=static_cast<float*>(slot.input.value.contents);
  auto pixel=[&](size_t direction,size_t row,size_t col) __attribute__((always_inline)) {
   size_t k=row+col*n,index=k+(base+direction)*n*n;mxComplexSingle value;
   if(pupil){
    const auto& q=pupil->inputs[base+direction];double a=q.amplitude[q.na==1?0:k],p=q.phase[q.np==1?0:k];
    if(!q.mask[k]&&std::isfinite(a)&&std::isfinite(p))value={0,0};
    else {a*=double(q.mask[k]);p*=double(q.mask[k]);value={float(a*std::cos(p)),float(a*std::sin(p))};}
   }else value=complex?complex[index]:mxComplexSingle{real[index],0};
   float re=value.real/float(nf),im=value.imag/float(nf);
   if(!shifted){float pr=float(phase[k].real),pi=float(phase[k].imag);value={re*pr-im*pi,re*pi+im*pr};}else value={re,im};
   packed[row*columns+direction*n+col]=value.real;
   packed[(n+row)*columns+direction*n+col]=value.imag;
  };
  size_t tiles=(n+context.tile-1)/context.tile;
  oomaoPool().run(d*tiles*tiles,context.threads,[&](size_t begin,size_t end){
    for(size_t k=begin;k<end;++k){size_t direction=k/(tiles*tiles),row0=((k/tiles)%tiles)*context.tile,col0=(k%tiles)*context.tile;
     for(size_t row=row0;row<std::min(row0+context.tile,size_t(n));++row)
      for(size_t col=col0;col<std::min(col0+context.tile,size_t(n));++col)pixel(direction,row,col);
    }
  });
  id<MTLCommandBuffer> command=[context.queue commandBuffer];if(!command)throw std::runtime_error("Metal command allocation failed.");
  MPSMatrixMultiplication* first=product(context.device,2*crop,columns,2*n);
  [first encodeToCommandBuffer:command leftMatrix:roots rightMatrix:matrix(slot.input.value,0,2*n,columns)
   resultMatrix:matrix(slot.first.value,0,2*crop,columns)];
  auto sizes=std::array<uint32_t,4>{uint32_t(n),uint32_t(crop),uint32_t(d),0};
  context.encode(command,context.transpose,slot.first,slot.intermediate,sizes,d*2*n*crop);
  MPSMatrixMultiplication* second=product(context.device,2*crop,crop,2*n);
  auto& repeated=found->second->repeated[d];
  if(!repeated.bytes){
   size_t bytes=size_t(2*crop)*2*n*4;repeated.ensure(context.device,d*bytes);
   for(size_t k=0;k<d;++k)std::memcpy(static_cast<char*>(repeated.value.contents)+k*bytes,found->second->block.value.contents,bytes);
  }
  second.batchSize=d;
  [second encodeToCommandBuffer:command leftMatrix:batchedMatrix(repeated.value,2*crop,2*n,d)
   rightMatrix:batchedMatrix(slot.intermediate.value,2*n,crop,d)
   resultMatrix:batchedMatrix(slot.second.value,2*crop,crop,d)];
  context.encode(command,context.pack,slot.second,slot.output,sizes,d*crop*crop);
  slot.base=base;slot.directions=d;slot.command=command;[command commit];
 }
 for(auto& slot:context.slots)finish(slot);

 return result.release();
}
static void dispatch(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 if(nrhs<1||!mxIsChar(in[0]))throw std::runtime_error("MPS operation name required.");
 char* raw=mxArrayToString(in[0]);std::string operation(raw?raw:"");mxFree(raw);
 if(operation=="create"){
  if(nlhs!=1||nrhs!=2)throw std::runtime_error("create requires threads and one handle output.");
  size_t threads=oomaoThreads(in[1]);auto context=std::make_unique<Context>(threads);uint64_t key=nextHandle++;
  out[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(out[0]))=key;
  contexts.emplace(key,std::move(context));mexLock();oomaoPool();mexAtExit(cleanup);return;
 }
 if(nrhs<2)throw std::runtime_error("MPS handle required.");
 auto found=contexts.find(handle(in[1]));if(found==contexts.end())throw std::runtime_error("Unknown or released MPS handle.");

 if(operation=="destroy"){
  if(nlhs||nrhs!=2)throw std::runtime_error("destroy requires only a handle and no output.");
  contexts.erase(found);mexUnlock();return;
 }
 if(operation=="diffractPupil"){
  if(nlhs!=1||nrhs!=10||integer(in[9])!=32)throw std::runtime_error("MPS pupil diffraction requires cells, geometry and precision32.");
  Pupil pupil(in[2],in[3],in[4]);out[0]=diffract(*found->second,nullptr,in[5],integer(in[6]),integer(in[7]),integer(in[8]),&pupil);return;
 }
 if(operation=="diffract"){
  if(nlhs!=1||nrhs!=7)throw std::runtime_error("diffract requires handle, field, phasor, FFT size and crop start/width.");
  out[0]=diffract(*found->second,in[2],in[3],integer(in[4]),integer(in[5]),integer(in[6]));return;
 }
 throw std::runtime_error("Unknown MPS operation.");
}
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 std::string failure;
 @autoreleasepool { @try { try {dispatch(nlhs,out,nrhs,in);}
  catch(const std::exception& error){failure=error.what();}
 } @catch(NSException* error){failure=error.reason?error.reason.UTF8String:"Metal/MPS Objective-C exception.";} }
 if(!failure.empty())mexErrMsgIdAndTxt("oomao:mps:execution","%s",failure.c_str());
}
