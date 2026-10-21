// Compiled CUDA backend. No gpuArray, MATLAB GPU toolbox, or Python calls.
#include "mex.h"
#include "cuda.h"
#include "oomao_nvrtc_api.hpp"
#include "oomao_cufft_api.hpp"
#include "oomao_cublas_api.hpp"
#include "oomao_fast_threads.hpp"
#include <algorithm>
#include <atomic>
#include <array>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void driver(CUresult code) {
 if(code!=CUDA_SUCCESS){const char* text=nullptr;cuGetErrorString(code,&text);throw std::runtime_error(text?text:"CUDA driver failure");}
}
static void rtc(int code){if(code)throw std::runtime_error(nvrtcGetErrorString(code));}
static void fft(int code){if(code)throw std::runtime_error("cuFFT error "+std::to_string(code));}
static void blas(int code){if(code)throw std::runtime_error("cuBLAS error "+std::to_string(code));}
struct Current {Current(CUcontext c){driver(cuCtxPushCurrent(c));}~Current(){CUcontext c;cuCtxPopCurrent(&c);}};
struct Buffer {
 CUdeviceptr value=0;size_t bytes=0;
 void reset(){if(value)cuMemFree(value);value=0;bytes=0;}
 void ensure(size_t count){if(count<=bytes)return;size_t free,total;driver(cuMemGetInfo(&free,&total));if(count>free||free-count<512ull*1024*1024)throw std::runtime_error("CUDA workspace would leave less than 512 MiB free.");reset();driver(cuMemAlloc(&value,count));bytes=count;}
 ~Buffer(){reset();}
};
template<class T>struct PinnedVector {
 T* pointer=nullptr;size_t capacity=0,count=0;
 PinnedVector()=default;PinnedVector(const PinnedVector&)=delete;PinnedVector& operator=(const PinnedVector&)=delete;
 void reset(){if(pointer)cuMemFreeHost(pointer);pointer=nullptr;capacity=0;count=0;}
 void resize(size_t n,unsigned flags=CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_WRITECOMBINED){
  if(n>capacity){if(n>512ull*1024*1024/sizeof(T))throw std::runtime_error("Pinned staging buffer exceeds 512 MiB limit.");reset();void* next=nullptr;driver(cuMemHostAlloc(&next,n*sizeof(T),flags));pointer=static_cast<T*>(next);capacity=n;}
  count=n;
 }
 T* data(){return pointer;}size_t size()const{return count;}
 ~PinnedVector(){reset();}
};
struct Plan {
 int handle=0;size_t bytes=0;
 Plan(int n,int batches,bool single,CUstream stream){try{fft(cufftPlan1d(&handle,n,single?OOMAO_CUFFT_C2C:OOMAO_CUFFT_Z2Z,batches));fft(cufftSetStream(handle,stream));fft(cufftGetSize(handle,&bytes));}catch(...){if(handle)cufftDestroy(handle);handle=0;throw;}}
 ~Plan(){if(handle)cufftDestroy(handle);}
};
static const char* kernels=R"CUDA(
template<class T>struct Z {T re,im;};
template<class T>__device__ void prepare(const Z<T>* field,const Z<double>* phase,Z<T>* first,int n,int d,int nf,int usePhase) {
 unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long count=(unsigned long long)nf*n*d;if(k>=count)return;
 int row=k%nf;unsigned long long col=k/nf;Z<T> v={0,0};
 if(row<n){Z<T> a=field[row+(unsigned long long)n*col];T re=a.re/T(nf),im=a.im/T(nf);
  if(usePhase){Z<double> p=phase[row+(unsigned long long)n*(col%n)];T pr=T(p.re),pi=T(p.im);v.re=re*pr-im*pi;v.im=re*pi+im*pr;}
  else{v.re=re;v.im=im;}}
 first[k]=v;
}
template<class T>__device__ void gather(const Z<T>* first,Z<T>* second,int n,int d,int nf,int crop,int start,int shift) {
 unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long count=(unsigned long long)nf*crop*d;if(k>=count)return;
 int x=k%nf;unsigned long long col=k/nf;int direction=col%d,y=col/d;
 int frequency=(start+y+shift)%nf;Z<T> v={0,0};
 if(x<n)v=first[frequency+(unsigned long long)nf*(x+n*direction)];second[k]=v;
}
template<class T>__device__ void extract(const Z<T>* second,Z<T>* output,int d,int nf,int crop,int start,int shift) {
 unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long count=(unsigned long long)crop*crop*d;if(k>=count)return;
 int y=k%crop;unsigned long long col=k/crop;int direction=col/crop,x=col%crop;
 int frequency=(start+x+shift)%nf;output[k]=second[frequency+(unsigned long long)nf*(direction+d*y)];
}
extern "C" __global__ void prepare32(const Z<float>* f,const Z<double>* p,Z<float>* o,int n,int d,int nf,int flag){prepare(f,p,o,n,d,nf,flag);}
extern "C" __global__ void prepare64(const Z<double>* f,const Z<double>* p,Z<double>* o,int n,int d,int nf,int flag){prepare(f,p,o,n,d,nf,flag);}
extern "C" __global__ void gather32(const Z<float>* f,Z<float>* o,int n,int d,int nf,int c,int s,int shift){gather(f,o,n,d,nf,c,s,shift);}
extern "C" __global__ void gather64(const Z<double>* f,Z<double>* o,int n,int d,int nf,int c,int s,int shift){gather(f,o,n,d,nf,c,s,shift);}
extern "C" __global__ void extract32(const Z<float>* f,Z<float>* o,int d,int nf,int c,int s,int shift){extract(f,o,d,nf,c,s,shift);}
extern "C" __global__ void extract64(const Z<double>* f,Z<double>* o,int d,int nf,int c,int s,int shift){extract(f,o,d,nf,c,s,shift);}
template<class T>__device__ void compact(const Z<T>* f,const Z<double>* p,Z<T>* out,int n,int d,int nf,int usePhase){
 unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=(unsigned long long)n*n*d)return;
 Z<T> a=f[k];T re=a.re/T(nf),im=a.im/T(nf);Z<T> v;
 if(usePhase){Z<double> z=p[k%((unsigned long long)n*n)];T pr=T(z.re),pi=T(z.im);v={re*pr-im*pi,re*pi+im*pr};}else{v={re,im};}out[k]=v;
}
extern "C" __global__ void compact32(const Z<float>* f,const Z<double>* p,Z<float>* o,int n,int d,int nf,int flag){compact(f,p,o,n,d,nf,flag);}
extern "C" __global__ void compact64(const Z<double>* f,const Z<double>* p,Z<double>* o,int n,int d,int nf,int flag){compact(f,p,o,n,d,nf,flag);}
extern "C" __global__ void sparseSurface64(const double* values,const int* offsets,const int* modes,const double* coefficients,double* out,int pixels,int actuators,int directions,int lanes){
 unsigned long long thread=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,k=thread/lanes;if(k>=(unsigned long long)pixels*directions)return;
 int pixel=k%pixels,direction=k/pixels,lane=thread%lanes;double v=0;
 for(int j=offsets[pixel]+lane;j<offsets[pixel+1];j+=lanes)v+=values[j]*coefficients[modes[j]+(unsigned long long)direction*actuators];
 unsigned mask=__activemask();for(int offset=lanes/2;offset>0;offset/=2)v+=__shfl_down_sync(mask,v,offset,lanes);if(lane==0)out[k]=v;
}
template<class T>__device__ void pupil(const double* amplitudes,const double* phases,const unsigned char* masks,const unsigned long long* meta,Z<T>* out,int pixels,int directions,int waves,const Z<double>* phasor,int nf,int prepared){
 unsigned long long k=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=(unsigned long long)pixels*directions*waves)return;
 unsigned long long wave=k/((unsigned long long)pixels*directions),d=(k/pixels)%directions,i=k%pixels;const unsigned long long* m=meta+5*d;
 double a=amplitudes[m[0]+(m[3]==1?0:i+(m[3]>(unsigned long long)pixels?wave*pixels:0))],p=phases[m[1]+(m[4]==1?0:i+(m[4]>(unsigned long long)pixels?wave*pixels:0))];
 unsigned char mask=masks[m[2]+i];Z<T> value;if(!mask&&isfinite(a)&&isfinite(p))value={0,0};else{a=a*double(mask);p=p*double(mask);value={T(a*cos(p)),T(a*sin(p))};}
 // Preserve the original cast, normalization and phasor arithmetic order.
 if(prepared){T re=value.re/T(nf),im=value.im/T(nf);if(phasor){Z<double> q=phasor[i];T pr=T(q.re),pi=T(q.im);value={re*pr-im*pi,re*pi+im*pr};}else value={re,im};}out[k]=value;
}
extern "C" __global__ void pupil32(const double* a,const double* p,const unsigned char* m,const unsigned long long* meta,Z<float>* out,int pixels,int directions,int waves,const Z<double>* phasor,int nf,int prepared){pupil(a,p,m,meta,out,pixels,directions,waves,phasor,nf,prepared);}
extern "C" __global__ void pupil64(const double* a,const double* p,const unsigned char* m,const unsigned long long* meta,Z<double>* out,int pixels,int directions,int waves,const Z<double>* phasor,int nf,int prepared){pupil(a,p,m,meta,out,pixels,directions,waves,phasor,nf,prepared);}
extern "C" __global__ void transpose64(const double* in,double* out,int rows,int columns){
 __shared__ double tile[32][33];int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<rows&&y+j<columns)tile[threadIdx.y+j][threadIdx.x]=in[x+(unsigned long long)(y+j)*rows];__syncthreads();
 x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<columns&&y+j<rows)out[x+(unsigned long long)(y+j)*columns]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" __global__ void sparseDirections64(const double* values,const int* offsets,const int* modes,const double* transposedCoefficients,double* transposedOut,int pixels,int directions){
 unsigned long long thread=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,warp=thread/32;int pixel=warp%pixels,direction=int(warp/pixels)*32+threadIdx.x%32;if(direction>=directions||warp>=(unsigned long long)pixels*((directions+31)/32))return;
 double v=0;for(int j=offsets[pixel];j<offsets[pixel+1];++j)v=__fma_rn(values[j],transposedCoefficients[direction+(unsigned long long)modes[j]*directions],v);transposedOut[direction+(unsigned long long)pixel*directions]=v;
}
extern "C" __global__ void add64(double* a,const double* b,int n){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n)a[k]=a[k]+b[k];}
)CUDA";

struct Reconstruction {Buffer matrices,projector,slopes,work,output;int rows=0,cols=0,directions=0;};
struct Roots {Buffer values;};
struct Surface {Buffer values,offsets,modes,coefficients,output,transposedCoefficients,transposedOutput;int pixels=0,actuators=0;};
struct Border {Buffer a,b,z,noise,x,y;int rows=0,cols=0;};
struct PupilSlot {
 Buffer amplitude,phase,mask,meta,field,first,output,pointers;
 PinnedVector<double> hostAmplitude,hostPhase;
 PinnedVector<unsigned char> hostMask;
 PinnedVector<unsigned long long> hostMeta;
 PinnedVector<CUdeviceptr> hostPointers;
 PinnedVector<mxComplexSingle> hostOutput;
 CUevent ready=nullptr,done=nullptr;bool pending=false;size_t base=0,directions=0;
 void reset(){amplitude.reset();phase.reset();mask.reset();meta.reset();field.reset();first.reset();output.reset();pointers.reset();hostAmplitude.reset();hostPhase.reset();hostMask.reset();hostMeta.reset();hostPointers.reset();hostOutput.reset();if(ready)cuEventDestroy(ready);if(done)cuEventDestroy(done);ready=done=nullptr;}
};
struct Context {
 std::array<PupilSlot,2> pupilSlots;CUstream upload=nullptr;
 CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;CUmodule module=nullptr;
 cublasHandle_t blasHandle=nullptr;CUfunction add,compact32,compact64,surface,pupil32,pupil64,transpose,sparseDirections;
 Buffer pupilAmplitude,pupilPhase,pupilMask,pupilMetadata,pupilField;
 PinnedVector<double> hostAmplitudes,hostPhases;
 PinnedVector<unsigned char> hostMasks;std::vector<unsigned long long> hostPupilMetadata;
 std::map<uint64_t,std::unique_ptr<Reconstruction>> reconstructions;uint64_t nextReconstruction=1;
 std::map<uint64_t,std::unique_ptr<Surface>> surfaces;uint64_t nextSurface=1;
 std::map<uint64_t,std::unique_ptr<Border>> borders;uint64_t nextBorder=1;
 CUfunction prep32,prep64,gather32,gather64,extract32,extract64;
 Buffer input,phase,first,second,output;
 std::map<std::array<int,5>,std::unique_ptr<Roots>> roots;
 std::map<std::array<int,3>,std::unique_ptr<Plan>> plans;
 // Device pointer arrays for the second, direction-batched MFT product.
 std::map<std::array<unsigned long long,7>,std::unique_ptr<Buffer>> batchPointers;
 int batchLimit=8;size_t hostThreads=1;
 explicit Context(int batch,int threads){
  batchLimit=batch;hostThreads=size_t(threads);driver(cuInit(0));driver(cuDeviceGet(&device,0));driver(cuDevicePrimaryCtxRetain(&context,device));
  try{Current current(context);driver(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));blas(cublasCreate_v2(&blasHandle));blas(cublasSetStream_v2(blasHandle,stream));blas(cublasSetMathMode(blasHandle,2));compile();}
  catch(...){cleanup();throw;}
 }
 void cleanup() noexcept {
  if(context){CUcontext old;cuCtxPushCurrent(context);if(stream)cuStreamSynchronize(stream);if(upload)cuStreamSynchronize(upload);for(auto& slot:pupilSlots)slot.reset();if(upload)cuStreamDestroy(upload);upload=nullptr;plans.clear();roots.clear();borders.clear();surfaces.clear();reconstructions.clear();batchPointers.clear();pupilAmplitude.reset();pupilPhase.reset();pupilMask.reset();pupilMetadata.reset();pupilField.reset();hostAmplitudes.reset();hostPhases.reset();hostMasks.reset();if(blasHandle)cublasDestroy_v2(blasHandle);blasHandle=nullptr;input.reset();phase.reset();first.reset();second.reset();output.reset();if(module)cuModuleUnload(module);if(stream)cuStreamDestroy(stream);cuCtxPopCurrent(&old);cuDevicePrimaryCtxRelease(device);context=nullptr;module=nullptr;stream=nullptr;}
 }
 ~Context(){cleanup();}
 void compile(){
  nvrtcProgram program=nullptr;rtc(nvrtcCreateProgram(&program,kernels,"oomao_cuda.cu",0,nullptr,nullptr));
  try{
   int major,minor;driver(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));driver(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
   std::string arch="--gpu-architecture=sm_"+std::to_string(major)+std::to_string(minor);
   const char* flags[]={arch.c_str(),"--std=c++17","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true"};
   int status=nvrtcCompileProgram(program,6,flags);
   if(status){size_t bytes=0;nvrtcGetProgramLogSize(program,&bytes);std::vector<char> log(bytes);nvrtcGetProgramLog(program,log.data());throw std::runtime_error(log.data());}
   size_t bytes;rtc(nvrtcGetCUBINSize(program,&bytes));std::vector<char> cubin(bytes);rtc(nvrtcGetCUBIN(program,cubin.data()));driver(cuModuleLoadData(&module,cubin.data()));
   driver(cuModuleGetFunction(&transpose,module,"transpose64"));driver(cuModuleGetFunction(&sparseDirections,module,"sparseDirections64"));driver(cuModuleGetFunction(&pupil32,module,"pupil32"));driver(cuModuleGetFunction(&pupil64,module,"pupil64"));driver(cuModuleGetFunction(&surface,module,"sparseSurface64"));driver(cuModuleGetFunction(&compact32,module,"compact32"));driver(cuModuleGetFunction(&compact64,module,"compact64"));driver(cuModuleGetFunction(&add,module,"add64"));driver(cuModuleGetFunction(&prep32,module,"prepare32"));driver(cuModuleGetFunction(&prep64,module,"prepare64"));driver(cuModuleGetFunction(&gather32,module,"gather32"));driver(cuModuleGetFunction(&gather64,module,"gather64"));driver(cuModuleGetFunction(&extract32,module,"extract32"));driver(cuModuleGetFunction(&extract64,module,"extract64"));
  }catch(...){nvrtcDestroyProgram(&program);throw;}rtc(nvrtcDestroyProgram(&program));
 }
 Plan& plan(int n,int batches,bool single){
  std::array<int,3> key={n,batches,single?32:64};auto found=plans.find(key);if(found!=plans.end())return *found->second;
  if(plans.size()>=16){driver(cuStreamSynchronize(stream));plans.erase(plans.begin());}
  auto value=std::make_unique<Plan>(n,batches,single,stream);auto* ptr=value.get();plans.emplace(key,std::move(value));return *ptr;
 }
 void launch(CUfunction fn,size_t count,void** args){driver(cuLaunchKernel(fn,unsigned((count+255)/256),1,1,256,1,1,0,stream,args,nullptr));}
 void transform(Buffer& data,int nf,int batches,bool single){auto& p=plan(nf,batches,single);if(single)fft(cufftExecC2C(p.handle,reinterpret_cast<void*>(data.value),reinterpret_cast<void*>(data.value),-1));else fft(cufftExecZ2Z(p.handle,reinterpret_cast<void*>(data.value),reinterpret_cast<void*>(data.value),-1));}
};
static std::map<uint64_t,std::unique_ptr<Context>> contexts;
static uint64_t nextId=1;
static void cleanup(){contexts.clear();oomaoPoolCleanup();}
static int integer(const mxArray* value){if(!mxIsDouble(value)||mxIsComplex(value)||mxIsSparse(value)||mxGetNumberOfElements(value)!=1)throw std::runtime_error("Real integer scalar required.");double v=mxGetScalar(value);if(!std::isfinite(v)||v<0||v>16384||v!=std::floor(v))throw std::runtime_error("Integer outside supported range 0..16384.");return int(v);}
static uint64_t id(const mxArray* value){if(!mxIsUint64(value)||mxGetNumberOfElements(value)!=1)throw std::runtime_error("An owned CUDA uint64 handle is required.");return *static_cast<const uint64_t*>(mxGetData(value));}
static mxArray* diffract(Context& c,const mxArray* field,const mxArray* phasor,int nf,int start,int crop,CUdeviceptr deviceInput=0,int inputSide=0,int inputDirections=0,int inputPrecision=64){
 if(field&&(mxIsSparse(field)||!(mxIsSingle(field)||mxIsDouble(field))||mxGetNumberOfDimensions(field)!=2))throw std::runtime_error("Full real/complex single/double pupil pages required.");
 int n=field?int(mxGetM(field)):inputSide;if(n<2||n>16384||(field&&mxGetN(field)%n)||nf<n||crop<1||start+crop>nf)throw std::runtime_error("Invalid pupil/FFT/crop shape.");
 int directions=field?int(mxGetN(field)/n):inputDirections;bool single=field?mxIsSingle(field):inputPrecision==32,withPhase=!mxIsEmpty(phasor);
 if(directions<1||(!field&&!deviceInput))throw std::runtime_error("No pupil input.");
 if(withPhase&&(!mxIsDouble(phasor)||!mxIsComplex(phasor)||mxIsSparse(phasor)||mxGetM(phasor)!=size_t(n)||mxGetN(phasor)!=size_t(n)))throw std::runtime_error("Empty or pupil-sized complex double phasor required.");
 Current current(c.context);size_t element=single?sizeof(mxComplexSingle):sizeof(mxComplexDouble);
 const void* fieldData=field?mxGetData(field):nullptr;
 std::vector<mxComplexSingle> realSingle;std::vector<mxComplexDouble> realDouble;
 if(field&&!mxIsComplex(field)){
  size_t count=mxGetNumberOfElements(field);
  if(single){realSingle.resize(count);const float* p=mxGetSingles(field);for(size_t k=0;k<count;++k)realSingle[k]={p[k],0};fieldData=realSingle.data();}
  else{realDouble.resize(count);const double* p=mxGetDoubles(field);for(size_t k=0;k<count;++k)realDouble[k]={p[k],0};fieldData=realDouble.data();}
 }
 mxArray* out=mxCreateUninitNumericMatrix(crop,size_t(crop)*directions,single?mxSINGLE_CLASS:mxDOUBLE_CLASS,mxCOMPLEX);
 try{
  if(withPhase){c.phase.ensure(size_t(n)*n*sizeof(mxComplexDouble));driver(cuMemcpyHtoDAsync(c.phase.value,mxGetComplexDoubles(phasor),size_t(n)*n*sizeof(mxComplexDouble),c.stream));}
  for(int base=0;base<directions;base+=c.batchLimit){int d=std::min(c.batchLimit,directions-base),shift=withPhase?0:(nf+1)/2,phaseFlag=withPhase?1:0;
   if(!deviceInput)c.input.ensure(size_t(n)*n*d*element);c.first.ensure(size_t(nf)*n*d*element);c.second.ensure(size_t(nf)*crop*d*element);c.output.ensure(size_t(crop)*crop*d*element);
   const char* source=fieldData?static_cast<const char*>(fieldData)+size_t(base)*n*n*element:nullptr;
   CUdeviceptr input=c.input.value;
   if(deviceInput)input=deviceInput+size_t(base)*n*n*element;else driver(cuMemcpyHtoDAsync(input,source,size_t(n)*n*d*element,c.stream));
   void* prepArgs[]={&input,&c.phase.value,&c.first.value,&n,&d,&nf,&phaseFlag};c.launch(single?c.prep32:c.prep64,size_t(nf)*n*d,prepArgs);c.transform(c.first,nf,n*d,single);
   void* gatherArgs[]={&c.first.value,&c.second.value,&n,&d,&nf,&crop,&start,&shift};c.launch(single?c.gather32:c.gather64,size_t(nf)*crop*d,gatherArgs);c.transform(c.second,nf,crop*d,single);
   void* extractArgs[]={&c.second.value,&c.output.value,&d,&nf,&crop,&start,&shift};c.launch(single?c.extract32:c.extract64,size_t(crop)*crop*d,extractArgs);
   char* destination=static_cast<char*>(mxGetData(out))+size_t(base)*crop*crop*element;driver(cuMemcpyDtoHAsync(destination,c.output.value,size_t(crop)*crop*d*element,c.stream));
  }
  driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);mxDestroyArray(out);throw;}
 return out;
}
template<class T,class Z>static mxArray* diffractMft(Context& c,const mxArray* field,const mxArray* phase,int nf,int start,int crop,CUdeviceptr deviceInput=0,int inputSide=0,int inputDirections=0,bool preparedInput=false){
 bool single=sizeof(T)==4,withPhase=!mxIsEmpty(phase);int n=field?int(mxGetM(field)):inputSide,directions=field?(n?int(mxGetN(field)/n):0):inputDirections,shift=withPhase?0:(nf+1)/2;
 if(n<2||n>16384||directions<1||(!field&&!deviceInput)||(field&&(mxGetN(field)%n||mxGetNumberOfDimensions(field)!=2||mxIsSparse(field)||!(mxIsSingle(field)||mxIsDouble(field))))||nf<n||crop<1||start+crop>nf)throw std::runtime_error("Invalid MFT input.");
 if(withPhase&&(!mxIsDouble(phase)||!mxIsComplex(phase)||mxIsSparse(phase)||mxGetM(phase)!=size_t(n)||mxGetN(phase)!=size_t(n)))throw std::runtime_error("Invalid MFT phasor.");
 Current current(c.context);std::array<int,5> key={n,nf,start+shift,crop,single?32:64};auto found=c.roots.find(key);
 if(found==c.roots.end()){
  if(c.roots.size()>=16){driver(cuStreamSynchronize(c.stream));c.roots.erase(c.roots.begin());}
  auto r=std::make_unique<Roots>();r->values.ensure(size_t(crop)*n*sizeof(Z));std::vector<Z> roots(size_t(crop)*n);
  for(int x=0;x<n;++x)for(int j=0;j<crop;++j){size_t index=(size_t((start+shift+j)%nf)*x)%nf;double angle=-2*3.141592653589793238462643383279502884*double(index)/nf;roots[j+crop*x]={T(std::cos(angle)),T(std::sin(angle))};}
  driver(cuMemcpyHtoDAsync(r->values.value,roots.data(),roots.size()*sizeof(Z),c.stream));driver(cuStreamSynchronize(c.stream));found=c.roots.emplace(key,std::move(r)).first;
 }
 std::vector<Z> converted;const void* source=field?mxGetData(field):nullptr;
 if(field&&!mxIsComplex(field)){converted.resize(mxGetNumberOfElements(field));const T* p=static_cast<const T*>(source);for(size_t k=0;k<converted.size();++k)converted[k]={p[k],0};source=converted.data();}
 mxArray* out=mxCreateUninitNumericMatrix(crop,size_t(crop)*directions,single?mxSINGLE_CLASS:mxDOUBLE_CLASS,mxCOMPLEX);
 try{if(withPhase&&!preparedInput){c.phase.ensure(size_t(n)*n*sizeof(mxComplexDouble));driver(cuMemcpyHtoDAsync(c.phase.value,mxGetData(phase),size_t(n)*n*sizeof(mxComplexDouble),c.stream));}
 const Z one={1,0},zero={0,0};auto multiply=[&](int ta,int tb,int m,int nn,int inner,CUdeviceptr a,int lda,CUdeviceptr b,int ldb,CUdeviceptr dest,int ldc){
  if(single)blas(cublasCgemm_v2(c.blasHandle,ta,tb,m,nn,inner,&one,reinterpret_cast<void*>(a),lda,reinterpret_cast<void*>(b),ldb,&zero,reinterpret_cast<void*>(dest),ldc));
  else blas(cublasZgemm_v2(c.blasHandle,ta,tb,m,nn,inner,&one,reinterpret_cast<void*>(a),lda,reinterpret_cast<void*>(b),ldb,&zero,reinterpret_cast<void*>(dest),ldc));};
 for(int base=0;base<directions;base+=c.batchLimit){int d=std::min(c.batchLimit,directions-base),flag=withPhase?1:0;
  size_t elements=size_t(n)*n*d;if(!deviceInput)c.input.ensure(elements*sizeof(Z));if(!preparedInput)c.first.ensure(elements*sizeof(Z));c.second.ensure(size_t(crop)*n*d*sizeof(Z));c.output.ensure(size_t(crop)*crop*d*sizeof(Z));
  CUdeviceptr input=c.input.value;
  if(deviceInput)input=deviceInput+size_t(base)*n*n*sizeof(Z);else driver(cuMemcpyHtoDAsync(input,static_cast<const char*>(source)+size_t(base)*n*n*sizeof(Z),elements*sizeof(Z),c.stream));
  CUdeviceptr first=input;
  if(!preparedInput){first=c.first.value;void* args[]={&input,&c.phase.value,&first,&n,&d,&nf,&flag};c.launch(single?c.compact32:c.compact64,elements,args);}
  multiply(0,0,crop,n*d,n,found->second->values.value,crop,first,n,c.second.value,crop);
  std::array<unsigned long long,7> key={c.second.value,found->second->values.value,c.output.value,(unsigned long long)d,(unsigned long long)n,(unsigned long long)crop,sizeof(Z)};
   auto it=c.batchPointers.find(key);
   if(it==c.batchPointers.end()){if(c.batchPointers.size()>=16){driver(cuStreamSynchronize(c.stream));c.batchPointers.clear();}auto b=std::make_unique<Buffer>();b->ensure(size_t(3)*d*sizeof(CUdeviceptr));std::vector<CUdeviceptr> p(3*d);
    for(int j=0;j<d;++j){p[j]=c.second.value+size_t(j)*crop*n*sizeof(Z);p[d+j]=found->second->values.value;p[2*d+j]=c.output.value+size_t(j)*crop*crop*sizeof(Z);}
    driver(cuMemcpyHtoDAsync(b->value,p.data(),p.size()*sizeof(CUdeviceptr),c.stream));driver(cuStreamSynchronize(c.stream));it=c.batchPointers.emplace(key,std::move(b)).first;}
   CUdeviceptr pointers=it->second->value;
   if(single)blas(cublasCgemmBatched(c.blasHandle,0,1,crop,crop,n,&one,reinterpret_cast<void*>(pointers),crop,reinterpret_cast<void*>(pointers+size_t(d)*8),crop,&zero,reinterpret_cast<void*>(pointers+size_t(2)*d*8),crop,d));
   else blas(cublasZgemmBatched(c.blasHandle,0,1,crop,crop,n,&one,reinterpret_cast<void*>(pointers),crop,reinterpret_cast<void*>(pointers+size_t(d)*8),crop,&zero,reinterpret_cast<void*>(pointers+size_t(2)*d*8),crop,d));
  driver(cuMemcpyDtoHAsync(static_cast<char*>(mxGetData(out))+size_t(base)*crop*crop*sizeof(Z),c.output.value,size_t(crop)*crop*d*sizeof(Z),c.stream));
 }driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);mxDestroyArray(out);throw;}return out;
}
static void realDouble(const mxArray* a){if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||mxGetNumberOfDimensions(a)!=2)throw std::runtime_error("Full real double array required.");}
static mxArray* token(uint64_t id){mxArray* a=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(a))=id;return a;}
static mxArray* prepareBorder(Context& c,const mxArray* a,const mxArray* b){
 realDouble(a);realDouble(b);int m=int(mxGetM(a));int n=int(mxGetN(a));
 if(m<1||n<1||m>16384||n>32768||mxGetM(b)!=size_t(m)||mxGetN(b)!=size_t(m))throw std::runtime_error("Invalid border coefficient shapes.");
 Current current(c.context);auto p=std::make_unique<Border>();p->rows=m;p->cols=n;
 p->a.ensure(size_t(m)*n*8);p->b.ensure(size_t(m)*m*8);p->z.ensure(size_t(n)*8);p->noise.ensure(size_t(m)*8);p->x.ensure(size_t(m)*8);p->y.ensure(size_t(m)*8);
 driver(cuMemcpyHtoDAsync(p->a.value,mxGetDoubles(a),size_t(m)*n*8,c.stream));driver(cuMemcpyHtoDAsync(p->b.value,mxGetDoubles(b),size_t(m)*m*8,c.stream));driver(cuStreamSynchronize(c.stream));
 uint64_t key=c.nextBorder++;c.borders.emplace(key,std::move(p));return token(key);
}
static mxArray* applyBorder(Context& c,uint64_t key,const mxArray* z,const mxArray* noise){
 auto it=c.borders.find(key);if(it==c.borders.end())throw std::runtime_error("Unknown or released CUDA border plan.");Border& p=*it->second;
 realDouble(z);realDouble(noise);if(mxGetM(z)!=size_t(p.cols)||mxGetN(z)!=1||mxGetM(noise)!=size_t(p.rows)||mxGetN(noise)!=1)throw std::runtime_error("Border vector shape mismatch.");
 Current current(c.context);mxArray* out=mxCreateUninitNumericMatrix(p.rows,1,mxDOUBLE_CLASS,mxREAL);
 try{driver(cuMemcpyHtoDAsync(p.z.value,mxGetDoubles(z),size_t(p.cols)*8,c.stream));driver(cuMemcpyHtoDAsync(p.noise.value,mxGetDoubles(noise),size_t(p.rows)*8,c.stream));
 double one=1,zero=0;
 blas(cublasDgemv_v2(c.blasHandle,0,p.rows,p.cols,&one,reinterpret_cast<double*>(p.a.value),p.rows,reinterpret_cast<double*>(p.z.value),1,&zero,reinterpret_cast<double*>(p.x.value),1));
 blas(cublasDgemv_v2(c.blasHandle,0,p.rows,p.rows,&one,reinterpret_cast<double*>(p.b.value),p.rows,reinterpret_cast<double*>(p.noise.value),1,&zero,reinterpret_cast<double*>(p.y.value),1));
 void* args[]={&p.x.value,&p.y.value,&p.rows};c.launch(c.add,p.rows,args);driver(cuMemcpyDtoHAsync(mxGetDoubles(out),p.x.value,size_t(p.rows)*8,c.stream));driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);mxDestroyArray(out);throw;}return out;
}
static mxArray* prepareSurface(Context& c,const mxArray* transpose){
 if(!mxIsDouble(transpose)||!mxIsSparse(transpose)||mxIsComplex(transpose)||mxGetM(transpose)<1||mxGetM(transpose)>65536||mxGetN(transpose)<1||mxGetN(transpose)>10000000)throw std::runtime_error("A bounded real double sparse transpose is required.");
 size_t pixels=mxGetN(transpose),actuators=mxGetM(transpose);const mwIndex* jc=mxGetJc(transpose);const mwIndex* ir=mxGetIr(transpose);size_t nz=jc[pixels];if(nz>2147483647)throw std::runtime_error("Sparse layout exceeds 32-bit index limit.");
 std::vector<int> offsets(pixels+1),modes(nz);for(size_t k=0;k<=pixels;++k)offsets[k]=int(jc[k]);for(size_t k=0;k<nz;++k)modes[k]=int(ir[k]);
 Current current(c.context);auto p=std::make_unique<Surface>();p->pixels=int(pixels);p->actuators=int(actuators);p->values.ensure(std::max(size_t(8),nz*8));p->offsets.ensure(offsets.size()*4);p->modes.ensure(std::max(size_t(4),nz*4));
 driver(cuMemcpyHtoDAsync(p->offsets.value,offsets.data(),offsets.size()*4,c.stream));
 if(nz){driver(cuMemcpyHtoDAsync(p->values.value,mxGetDoubles(transpose),nz*8,c.stream));driver(cuMemcpyHtoDAsync(p->modes.value,modes.data(),nz*4,c.stream));}driver(cuStreamSynchronize(c.stream));
 uint64_t key=c.nextSurface++;c.surfaces.emplace(key,std::move(p));return token(key);
}
static mxArray* applySurface(Context& c,uint64_t key,const mxArray* coefficients,int lanes){
 auto it=c.surfaces.find(key);if(it==c.surfaces.end())throw std::runtime_error("Unknown or released CUDA surface plan.");Surface& p=*it->second;realDouble(coefficients);
 if((lanes!=0&&lanes!=1&&lanes!=4&&lanes!=8&&lanes!=16&&lanes!=32)||mxGetM(coefficients)!=size_t(p.actuators)||mxGetN(coefficients)>16384)throw std::runtime_error("DM coefficient dimensions differ.");int d=int(mxGetN(coefficients));mxArray* out=mxCreateUninitNumericMatrix(p.pixels,d,mxDOUBLE_CLASS,mxREAL);if(!d)return out;
 Current current(c.context);
 try{p.coefficients.ensure(size_t(p.actuators)*d*8);p.output.ensure(size_t(p.pixels)*d*8);driver(cuMemcpyHtoDAsync(p.coefficients.value,mxGetDoubles(coefficients),size_t(p.actuators)*d*8,c.stream));
 if(lanes==0){
  p.transposedCoefficients.ensure(size_t(p.actuators)*d*8);p.transposedOutput.ensure(size_t(p.pixels)*d*8);
  auto transpose=[&](CUdeviceptr source,CUdeviceptr destination,int rows,int columns){void* args[]={&source,&destination,&rows,&columns};driver(cuLaunchKernel(c.transpose,(rows+31)/32,(columns+31)/32,1,32,8,1,0,c.stream,args,nullptr));};
  transpose(p.coefficients.value,p.transposedCoefficients.value,p.actuators,d);
  void* args[]={&p.values.value,&p.offsets.value,&p.modes.value,&p.transposedCoefficients.value,&p.transposedOutput.value,&p.pixels,&d};c.launch(c.sparseDirections,size_t(p.pixels)*((d+31)/32)*32,args);
  transpose(p.transposedOutput.value,p.output.value,d,p.pixels);
 }else{void* args[]={&p.values.value,&p.offsets.value,&p.modes.value,&p.coefficients.value,&p.output.value,&p.pixels,&p.actuators,&d,&lanes};c.launch(c.surface,size_t(p.pixels)*d*lanes,args);}
 driver(cuMemcpyDtoHAsync(mxGetDoubles(out),p.output.value,size_t(p.pixels)*d*8,c.stream));driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);mxDestroyArray(out);throw;}return out;
}
static mxArray* formPupil(Context& c,const mxArray* amplitudes,const mxArray* phases,const mxArray* masks,int precision,bool download,int* sideOut=nullptr,int* directionsOut=nullptr,const mxArray* phasor=nullptr,int nf=1,bool prepared=false){
 if(!mxIsCell(amplitudes)||!mxIsCell(phases)||!mxIsCell(masks)||(precision!=32&&precision!=64))throw std::runtime_error("Pupil cells and precision 32/64 required.");
 size_t directions=mxGetNumberOfElements(amplitudes);if(!directions||directions>1024||mxGetNumberOfElements(phases)!=directions||mxGetNumberOfElements(masks)!=directions)throw std::runtime_error("Pupil cell counts differ.");
 auto* first=mxGetCell(masks,0);if(!first||!mxIsLogical(first)||mxIsSparse(first)||mxGetNumberOfDimensions(first)!=2)throw std::runtime_error("Logical pupil masks required.");size_t rows=mxGetM(first),columns=mxGetN(first),pixels=rows*columns,waves=1;if(!pixels||pixels>10000000)throw std::runtime_error("Pupil size outside supported range.");
 for(size_t d=0;d<directions;++d){auto* p=mxGetCell(phases,d);auto* a=mxGetCell(amplitudes,d);auto* m=mxGetCell(masks,d);if(!p||!a||!m)throw std::runtime_error("Uninitialized pupil cell.");if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a))throw std::runtime_error("Real double amplitude required.");if(!mxIsDouble(p)||mxIsComplex(p)||mxIsSparse(p))throw std::runtime_error("Real double phase required.");if(!mxIsLogical(m)||mxIsSparse(m)||mxGetNumberOfDimensions(m)!=2||mxGetM(m)!=rows||mxGetN(m)!=columns)throw std::runtime_error("Pupil mask shapes differ.");size_t np=mxGetNumberOfElements(p);if(np!=1&&(mxGetM(p)!=rows||np%pixels))throw std::runtime_error("Pupil phase shape mismatch.");waves=std::max(waves,np==1?size_t(1):np/pixels);}
 if(!download&&(rows!=columns||waves!=1))throw std::runtime_error("Fused diffraction requires square, single-wave pupils.");
 if(sideOut)*sideOut=int(rows);if(directionsOut)*directionsOut=int(directions);
 Current current(c.context);
 size_t amplitudeCount=0,phaseCount=0,maskCount=0;c.hostPupilMetadata.resize(5*directions);
 struct PupilInput {const double* amplitude;const double* phase;const mxLogical* mask;size_t na,np;bool sameAmplitude=false,sameMask=false;};std::vector<PupilInput> inputs(directions);
 auto* canonicalAmplitude=mxGetCell(amplitudes,0);const double* firstAmplitude=mxGetDoubles(canonicalAmplitude);size_t firstCount=mxGetNumberOfElements(canonicalAmplitude);const mxLogical* firstMask=mxGetLogicals(mxGetCell(masks,0));
 for(size_t d=0;d<directions;++d){auto* a=mxGetCell(amplitudes,d);auto* p=mxGetCell(phases,d);size_t na=mxGetNumberOfElements(a),np=mxGetNumberOfElements(p);if((na!=1&&na!=pixels&&na!=pixels*waves)||(np!=1&&np!=pixels&&np!=pixels*waves))throw std::runtime_error("Unsupported pupil broadcasting.");inputs[d]={mxGetDoubles(a),mxGetDoubles(p),mxGetLogicals(mxGetCell(masks,d)),na,np,false};}
 const size_t threads=directions>=4&&pixels>=65536?c.hostThreads:1;
 CUdeviceptr phaseDevice=0;bool withPhase=prepared&&phasor&&!mxIsEmpty(phasor);
 if(withPhase&&(!mxIsDouble(phasor)||!mxIsComplex(phasor)||mxIsSparse(phasor)||mxGetM(phasor)!=rows||mxGetN(phasor)!=columns))throw std::runtime_error("Invalid fused phasor.");

 oomaoPool().run(directions,threads,[&](size_t begin,size_t end){for(size_t d=begin;d<end;++d){auto& a=inputs[d];a.sameAmplitude=d>0&&a.na==firstCount&&(a.amplitude==firstAmplitude||std::memcmp(a.amplitude,firstAmplitude,a.na*8)==0);a.sameMask=d>0&&(a.mask==firstMask||std::memcmp(a.mask,firstMask,pixels)==0);}});
 for(size_t d=0;d<directions;++d){const auto& a=inputs[d];bool sameMask=a.sameMask;auto* m=c.hostPupilMetadata.data()+5*d;m[0]=a.sameAmplitude?0:amplitudeCount;m[1]=phaseCount;m[2]=sameMask?0:maskCount;m[3]=a.na;m[4]=a.np;if(!a.sameAmplitude)amplitudeCount+=a.na;if(!sameMask)maskCount+=pixels;phaseCount+=a.np;}
 c.hostAmplitudes.resize(amplitudeCount);c.hostPhases.resize(phaseCount);c.hostMasks.resize(maskCount);
 oomaoPool().run(directions,threads,[&](size_t begin,size_t end){for(size_t d=begin;d<end;++d){const auto* m=c.hostPupilMetadata.data()+5*d;const auto& input=inputs[d];if(d==0||m[0]!=0)std::memcpy(c.hostAmplitudes.data()+m[0],input.amplitude,m[3]*8);std::memcpy(c.hostPhases.data()+m[1],input.phase,m[4]*8);if(d==0||m[2]!=0)std::memcpy(c.hostMasks.data()+m[2],input.mask,pixels);}
  std::atomic_thread_fence(std::memory_order_seq_cst); // publish each worker's WC stores before DMA
 });
 auto& amplitudeValues=c.hostAmplitudes;auto& phaseValues=c.hostPhases;auto& maskValues=c.hostMasks;auto& meta=c.hostPupilMetadata;
 mwSize shape[]={rows,columns*directions,waves};mxArray* out=download?mxCreateUninitNumericArray(3,shape,precision==32?mxSINGLE_CLASS:mxDOUBLE_CLASS,mxCOMPLEX):nullptr;size_t count=pixels*directions*waves,bytes=count*(precision==32?sizeof(mxComplexSingle):sizeof(mxComplexDouble));
 try{c.pupilAmplitude.ensure(amplitudeValues.size()*8);c.pupilPhase.ensure(phaseValues.size()*8);c.pupilMask.ensure(maskValues.size());c.pupilMetadata.ensure(meta.size()*8);c.pupilField.ensure(bytes);
 if(withPhase){c.phase.ensure(pixels*sizeof(mxComplexDouble));driver(cuMemcpyHtoDAsync(c.phase.value,mxGetComplexDoubles(phasor),pixels*sizeof(mxComplexDouble),c.stream));phaseDevice=c.phase.value;}
 driver(cuMemcpyHtoDAsync(c.pupilAmplitude.value,amplitudeValues.data(),amplitudeValues.size()*8,c.stream));driver(cuMemcpyHtoDAsync(c.pupilPhase.value,phaseValues.data(),phaseValues.size()*8,c.stream));driver(cuMemcpyHtoDAsync(c.pupilMask.value,maskValues.data(),maskValues.size(),c.stream));driver(cuMemcpyHtoDAsync(c.pupilMetadata.value,meta.data(),meta.size()*8,c.stream));int pix=int(pixels),dir=int(directions),wave=int(waves);
 int preparedFlag=prepared?1:0;void* args[]={&c.pupilAmplitude.value,&c.pupilPhase.value,&c.pupilMask.value,&c.pupilMetadata.value,&c.pupilField.value,&pix,&dir,&wave,&phaseDevice,&nf,&preparedFlag};c.launch(precision==32?c.pupil32:c.pupil64,count,args);if(download)driver(cuMemcpyDtoHAsync(mxGetData(out),c.pupilField.value,bytes,c.stream));driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);if(out)mxDestroyArray(out);throw;}return out;
}
static mxArray* prepareReconstruction(Context& c,const mxArray* cells,const mxArray* projector){
 if(!mxIsCell(cells)||!mxGetNumberOfElements(cells)||mxGetNumberOfElements(cells)>1024)throw std::runtime_error("Nonempty reconstructor cell array required.");
 realDouble(projector);size_t m=mxGetM(projector),d=mxGetNumberOfElements(cells);if(!m||m>16384||mxGetN(projector)!=m)throw std::runtime_error("Square bounded projector required.");
 const mxArray* first=mxGetCell(cells,0);if(!first)throw std::runtime_error("Missing reconstructor.");realDouble(first);size_t n=mxGetN(first);
 if(!n||n>32768||m*d>2147483647||m*n*d>512ull*1024*1024/8)throw std::runtime_error("Reconstruction storage exceeds limit.");
 std::vector<double> stacked(m*n*d);
 for(size_t k=0;k<d;++k){const mxArray* a=mxGetCell(cells,k);if(!a)throw std::runtime_error("Missing reconstructor.");realDouble(a);if(mxGetM(a)!=m||mxGetN(a)!=n)throw std::runtime_error("Reconstructor shapes differ.");
  for(size_t j=0;j<n;++j)std::memcpy(stacked.data()+j*m*d+k*m,mxGetDoubles(a)+j*m,m*8);
 }
 Current current(c.context);auto p=std::make_unique<Reconstruction>();p->rows=int(m);p->cols=int(n);p->directions=int(d);
 p->matrices.ensure(stacked.size()*8);p->projector.ensure(m*m*8);p->slopes.ensure(n*8);p->work.ensure(m*d*8);p->output.ensure(m*d*8);
 try{driver(cuMemcpyHtoDAsync(p->matrices.value,stacked.data(),stacked.size()*8,c.stream));driver(cuMemcpyHtoDAsync(p->projector.value,mxGetDoubles(projector),m*m*8,c.stream));driver(cuStreamSynchronize(c.stream));}
 catch(...){cuStreamSynchronize(c.stream);throw;}
 uint64_t key=c.nextReconstruction++;c.reconstructions.emplace(key,std::move(p));return token(key);
}
static mxArray* applyReconstruction(Context& c,uint64_t key,const mxArray* slopes){
 auto it=c.reconstructions.find(key);if(it==c.reconstructions.end())throw std::runtime_error("Unknown reconstruction plan.");auto& p=*it->second;realDouble(slopes);
 if(mxGetNumberOfElements(slopes)!=size_t(p.cols))throw std::runtime_error("Slope count differs.");
 Current current(c.context);mxArray* out=mxCreateUninitNumericMatrix(p.rows,p.directions,mxDOUBLE_CLASS,mxREAL);
 try{driver(cuMemcpyHtoDAsync(p.slopes.value,mxGetDoubles(slopes),size_t(p.cols)*8,c.stream));double one=1,zero=0;
 blas(cublasDgemv_v2(c.blasHandle,0,p.rows*p.directions,p.cols,&one,reinterpret_cast<double*>(p.matrices.value),p.rows*p.directions,reinterpret_cast<double*>(p.slopes.value),1,&zero,reinterpret_cast<double*>(p.work.value),1));
 blas(cublasDgemm_v2(c.blasHandle,0,0,p.rows,p.directions,p.rows,&one,reinterpret_cast<double*>(p.projector.value),p.rows,reinterpret_cast<double*>(p.work.value),p.rows,&zero,reinterpret_cast<double*>(p.output.value),p.rows));
 driver(cuMemcpyDtoHAsync(mxGetDoubles(out),p.output.value,size_t(p.rows)*p.directions*8,c.stream));driver(cuStreamSynchronize(c.stream));
 }catch(...){cuStreamSynchronize(c.stream);mxDestroyArray(out);throw;}return out;
}
static mxArray* diffractPupilPipeline(Context& c,const mxArray* amplitudes,const mxArray* phases,const mxArray* masks,const mxArray* phasor,int nf,int start,int crop){
 if(!mxIsCell(amplitudes)||!mxIsCell(phases)||!mxIsCell(masks))throw std::runtime_error("Pupil cells required.");
 size_t directions=mxGetNumberOfElements(masks);if(!directions||directions>1024||mxGetNumberOfElements(amplitudes)!=directions||mxGetNumberOfElements(phases)!=directions)throw std::runtime_error("Pupil counts differ.");
 const mxArray* mask0=mxGetCell(masks,0);if(!mask0||!mxIsLogical(mask0)||mxIsSparse(mask0)||mxGetNumberOfDimensions(mask0)!=2||mxGetM(mask0)!=mxGetN(mask0))throw std::runtime_error("Square logical mask required.");
 int n=int(mxGetM(mask0));size_t pixels=size_t(n)*n;if(n<2||n>16384||nf<n||crop<1||start+crop>nf)throw std::runtime_error("Invalid diffraction geometry.");
 bool shifted=mxIsEmpty(phasor);int shift=shifted?(nf+1)/2:0;
 if(!shifted&&(!mxIsDouble(phasor)||!mxIsComplex(phasor)||mxIsSparse(phasor)||mxGetNumberOfDimensions(phasor)!=2||mxGetM(phasor)!=size_t(n)||mxGetN(phasor)!=size_t(n)))throw std::runtime_error("Invalid phasor.");
 struct Input{const double* amplitude;const double* phase;const mxLogical* mask;size_t na,np;};std::vector<Input> inputs(directions);
 for(size_t d=0;d<directions;++d){const mxArray *a=mxGetCell(amplitudes,d),*p=mxGetCell(phases,d),*m=mxGetCell(masks,d);
  if(!a||!p||!m)throw std::runtime_error("Missing pupil cell.");if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||!mxIsDouble(p)||mxIsComplex(p)||mxIsSparse(p))throw std::runtime_error("Real full double pupils required.");
  size_t na=mxGetNumberOfElements(a),np=mxGetNumberOfElements(p);
  if((na!=1&&na!=pixels)||(np!=1&&(mxGetM(p)!=size_t(n)||np!=pixels))||!mxIsLogical(m)||mxIsSparse(m)||mxGetNumberOfDimensions(m)!=2||mxGetM(m)!=size_t(n)||mxGetN(m)!=size_t(n))throw std::runtime_error("Pupil shapes differ.");
  inputs[d]={mxGetDoubles(a),mxGetDoubles(p),mxGetLogicals(m),na,np};
 }
 Current current(c.context);
 if(!c.upload)driver(cuStreamCreate(&c.upload,CU_STREAM_NON_BLOCKING));
 std::array<int,5> key={n,nf,start+shift,crop,32};auto found=c.roots.find(key);
 if(found==c.roots.end()){
  if(c.roots.size()>=16)c.roots.erase(c.roots.begin());
  auto roots=std::make_unique<Roots>();std::vector<mxComplexSingle> values(size_t(crop)*n);
  for(int x=0;x<n;++x)for(int j=0;j<crop;++j){size_t k=(size_t((start+shift+j)%nf)*x)%nf;double angle=-2*3.141592653589793238462643383279502884*double(k)/nf;values[j+crop*x]={float(std::cos(angle)),float(std::sin(angle))};}
  roots->values.ensure(values.size()*sizeof(mxComplexSingle));driver(cuMemcpyHtoD(roots->values.value,values.data(),values.size()*sizeof(mxComplexSingle)));found=c.roots.emplace(key,std::move(roots)).first;
 }
 size_t maxD=std::min(size_t(c.batchLimit),directions);
 for(auto& slot:c.pupilSlots){
  // Grow only before any submission in this call.
  slot.hostAmplitude.resize(pixels*maxD);slot.hostPhase.resize(pixels*maxD);slot.hostMask.resize(pixels*maxD);slot.hostMeta.resize(5*maxD);slot.hostPointers.resize(3*maxD);
  slot.amplitude.ensure(pixels*maxD*8);slot.phase.ensure(pixels*maxD*8);slot.mask.ensure(pixels*maxD);slot.meta.ensure(5*maxD*8);
  slot.field.ensure(pixels*maxD*sizeof(mxComplexSingle));slot.first.ensure(size_t(crop)*n*maxD*sizeof(mxComplexSingle));slot.output.ensure(size_t(crop)*crop*maxD*sizeof(mxComplexSingle));slot.pointers.ensure(3*maxD*sizeof(CUdeviceptr));
  slot.hostOutput.resize(size_t(crop)*crop*maxD,CU_MEMHOSTALLOC_PORTABLE);
  if(!slot.ready)driver(cuEventCreate(&slot.ready,CU_EVENT_DISABLE_TIMING));if(!slot.done)driver(cuEventCreate(&slot.done,CU_EVENT_DISABLE_TIMING));
 }
 if(!shifted){c.phase.ensure(pixels*sizeof(mxComplexDouble));driver(cuMemcpyHtoD(c.phase.value,mxGetComplexDoubles(phasor),pixels*sizeof(mxComplexDouble)));}
 std::unique_ptr<mxArray,decltype(&mxDestroyArray)> out(mxCreateUninitNumericMatrix(crop,size_t(crop)*directions,mxSINGLE_CLASS,mxCOMPLEX),mxDestroyArray);
 struct Pending {Context& c;~Pending(){cuStreamSynchronize(c.upload);cuStreamSynchronize(c.stream);for(auto& s:c.pupilSlots)s.pending=false;}} pending{c};
 auto finish=[&](PupilSlot& slot){if(!slot.pending)return;driver(cuEventSynchronize(slot.done));std::memcpy(mxGetComplexSingles(out.get())+slot.base*crop*crop,slot.hostOutput.data(),slot.directions*crop*crop*sizeof(mxComplexSingle));slot.pending=false;};
 size_t batch=0;
 for(size_t base=0;base<directions;base+=c.batchLimit,++batch){
  auto& slot=c.pupilSlots[batch%2];finish(slot);int d=int(std::min(size_t(c.batchLimit),directions-base));size_t ac=0,pc=0,mc=0;
  const auto& canonical=inputs[base];
  for(int j=0;j<d;++j){const auto& q=inputs[base+j];bool sameA=j&&q.na==canonical.na&&(q.amplitude==canonical.amplitude||std::memcmp(q.amplitude,canonical.amplitude,q.na*8)==0);bool sameM=j&&(q.mask==canonical.mask||std::memcmp(q.mask,canonical.mask,pixels)==0);
   auto* m=slot.hostMeta.data()+5*j;m[0]=sameA?0:ac;m[1]=pc;m[2]=sameM?0:mc;m[3]=q.na;m[4]=q.np;if(!sameA)ac+=q.na;pc+=q.np;if(!sameM)mc+=pixels;
  }
  oomaoPool().run(size_t(d),d>=4&&pixels>=65536?c.hostThreads:1,[&](size_t first,size_t last){for(size_t j=first;j<last;++j){const auto& q=inputs[base+j];const auto* m=slot.hostMeta.data()+5*j;if(j==0||m[0])std::memcpy(slot.hostAmplitude.data()+m[0],q.amplitude,q.na*8);std::memcpy(slot.hostPhase.data()+m[1],q.phase,q.np*8);if(j==0||m[2])std::memcpy(slot.hostMask.data()+m[2],q.mask,pixels);}std::atomic_thread_fence(std::memory_order_seq_cst);});
  for(int j=0;j<d;++j){slot.hostPointers.data()[j]=slot.first.value+size_t(j)*crop*n*sizeof(mxComplexSingle);slot.hostPointers.data()[d+j]=found->second->values.value;slot.hostPointers.data()[2*d+j]=slot.output.value+size_t(j)*crop*crop*sizeof(mxComplexSingle);}
  std::atomic_thread_fence(std::memory_order_seq_cst);
  driver(cuMemcpyHtoDAsync(slot.amplitude.value,slot.hostAmplitude.data(),ac*8,c.upload));driver(cuMemcpyHtoDAsync(slot.phase.value,slot.hostPhase.data(),pc*8,c.upload));driver(cuMemcpyHtoDAsync(slot.mask.value,slot.hostMask.data(),mc,c.upload));driver(cuMemcpyHtoDAsync(slot.meta.value,slot.hostMeta.data(),size_t(d)*5*8,c.upload));driver(cuMemcpyHtoDAsync(slot.pointers.value,slot.hostPointers.data(),size_t(d)*3*sizeof(CUdeviceptr),c.upload));
  driver(cuEventRecord(slot.ready,c.upload));driver(cuStreamWaitEvent(c.stream,slot.ready,0));
  int pix=int(pixels),waves=1,prepared=1;CUdeviceptr phase=shifted?0:c.phase.value;
  void* args[]={&slot.amplitude.value,&slot.phase.value,&slot.mask.value,&slot.meta.value,&slot.field.value,&pix,&d,&waves,&phase,&nf,&prepared};c.launch(c.pupil32,pixels*d,args);
  mxComplexSingle one={1,0},zero={0,0};blas(cublasCgemm_v2(c.blasHandle,0,0,crop,n*d,n,&one,reinterpret_cast<void*>(found->second->values.value),crop,reinterpret_cast<void*>(slot.field.value),n,&zero,reinterpret_cast<void*>(slot.first.value),crop));
  CUdeviceptr pointers=slot.pointers.value;
  blas(cublasCgemmBatched(c.blasHandle,0,1,crop,crop,n,&one,reinterpret_cast<void*>(pointers),crop,reinterpret_cast<void*>(pointers+size_t(d)*8),crop,&zero,reinterpret_cast<void*>(pointers+size_t(2)*d*8),crop,d));
  driver(cuMemcpyDtoHAsync(slot.hostOutput.data(),slot.output.value,size_t(crop)*crop*d*sizeof(mxComplexSingle),c.stream));driver(cuEventRecord(slot.done,c.stream));slot.pending=true;slot.base=base;slot.directions=d;
 }
 for(auto& slot:c.pupilSlots)finish(slot);
 return out.release();
}
static void dispatch(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 if(nrhs<1||!mxIsChar(in[0]))throw std::runtime_error("CUDA operation name required.");char* raw=mxArrayToString(in[0]);std::string operation(raw?raw:"");mxFree(raw);
 if(operation=="create"){
  if(nlhs!=1||nrhs<1||nrhs>3)throw std::runtime_error("create takes optional batch size/host threads and one handle output.");int batch=nrhs>=2?integer(in[1]):8,threads=nrhs==3?integer(in[2]):1;if(batch<1||batch>128||threads<1||threads>256)throw std::runtime_error("Batch size must be 1..128 and host threads 1..256.");
  auto value=std::make_unique<Context>(batch,threads);uint64_t key=nextId++;contexts.emplace(key,std::move(value));mexLock();oomaoPool();mexAtExit(cleanup);out[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(out[0]))=key;return;
 }
 if(nrhs<2)throw std::runtime_error("CUDA handle required.");auto found=contexts.find(id(in[1]));if(found==contexts.end())throw std::runtime_error("Unknown or released CUDA handle.");
 if(operation=="destroy"){
  if(nlhs!=0||nrhs!=2)throw std::runtime_error("destroy takes only a handle and no output.");contexts.erase(found);mexUnlock();return;
 }
 if(operation=="formPupil"){
  if(nlhs!=1||nrhs!=6)throw std::runtime_error("formPupil requires handle, amplitude/phase/mask cells and precision.");out[0]=formPupil(*found->second,in[2],in[3],in[4],integer(in[5]),true);return;
 }
 if(operation=="diffractPupil"){
  if(nlhs!=1||nrhs!=10)throw std::runtime_error("diffractPupil requires handle, pupil cells, phasor, FFT size, crop start/width, precision.");Context& c=*found->second;int nf=integer(in[6]),start=integer(in[7]),crop=integer(in[8]),precision=integer(in[9]),n=0,d=0;auto* firstMask=mxIsCell(in[4])&&mxGetNumberOfElements(in[4])?mxGetCell(in[4],0):nullptr;int sideHint=firstMask?int(mxGetM(firstMask)):0;
  bool prepared=precision==32&&sideHint>=64&&crop<=sideHint/2;
  if(prepared){out[0]=diffractPupilPipeline(c,in[2],in[3],in[4],in[5],nf,start,crop);return;}
  formPupil(c,in[2],in[3],in[4],precision,false,&n,&d,in[5],nf,prepared);
  out[0]=diffract(c,nullptr,in[5],nf,start,crop,c.pupilField.value,n,d,precision);return;
 }
 if(operation=="prepareReconstruction"){
  if(nlhs!=1||nrhs!=4)throw std::runtime_error("Reconstruction requires matrices/projector.");out[0]=prepareReconstruction(*found->second,in[2],in[3]);return;
 }
 if(operation=="applyReconstruction"){
  if(nlhs!=1||nrhs!=4)throw std::runtime_error("Reconstruction requires plan/slopes.");out[0]=applyReconstruction(*found->second,id(in[2]),in[3]);return;
 }
 if(operation=="releaseReconstruction"){
  if(nlhs||nrhs!=3)throw std::runtime_error("Release requires plan.");Current current(found->second->context);driver(cuStreamSynchronize(found->second->stream));if(!found->second->reconstructions.erase(id(in[2])))throw std::runtime_error("Unknown reconstruction plan.");return;
 }
 if(operation=="prepareSurface"){
  if(nlhs!=1||nrhs!=3)throw std::runtime_error("prepareSurface requires handle and sparse transpose.");out[0]=prepareSurface(*found->second,in[2]);return;
 }
 if(operation=="applySurface"){
  if(nlhs!=1||(nrhs!=4&&nrhs!=5))throw std::runtime_error("applySurface requires handle, token and coefficients, optional lane count.");out[0]=applySurface(*found->second,id(in[2]),in[3],nrhs==5?integer(in[4]):1);return;
 }
 if(operation=="releaseSurface"){
  if(nlhs||nrhs!=3)throw std::runtime_error("releaseSurface requires handle and token.");Current current(found->second->context);driver(cuStreamSynchronize(found->second->stream));if(!found->second->surfaces.erase(id(in[2])))throw std::runtime_error("Unknown surface plan.");return;
 }
 if(operation=="prepareBorder"){
  if(nlhs!=1||nrhs!=4)throw std::runtime_error("prepareBorder requires handle, A, B and one token output.");out[0]=prepareBorder(*found->second,in[2],in[3]);return;
 }
 if(operation=="releaseBorder"){
  if(nlhs||nrhs!=3)throw std::runtime_error("releaseBorder requires handle and token, no output.");Current current(found->second->context);auto& c=*found->second;driver(cuStreamSynchronize(c.stream));if(!c.borders.erase(id(in[2])))throw std::runtime_error("Unknown border plan.");return;
 }
 if(operation=="applyBorder"){
  if(nlhs!=1||nrhs!=5)throw std::runtime_error("applyBorder requires handle, token, Z and noise.");out[0]=applyBorder(*found->second,id(in[2]),in[3],in[4]);return;
 }
 if(operation=="diffractMft"){
  if(nlhs!=1||nrhs!=7)throw std::runtime_error("MFT requires handle, field, phasor, FFT size, crop start and width.");
  if(mxIsSingle(in[2]))out[0]=diffractMft<float,mxComplexSingle>(*found->second,in[2],in[3],integer(in[4]),integer(in[5]),integer(in[6]));
  else out[0]=diffractMft<double,mxComplexDouble>(*found->second,in[2],in[3],integer(in[4]),integer(in[5]),integer(in[6]));return;
 }
 if(operation=="diffract"){
  if(nlhs!=1||nrhs!=7)throw std::runtime_error("diffract takes handle, field, phasor, FFT size, crop start and crop width.");out[0]=diffract(*found->second,in[2],in[3],integer(in[4]),integer(in[5]),integer(in[6]));return;
 }
 throw std::runtime_error("Unknown CUDA operation.");
}
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){try{dispatch(nlhs,out,nrhs,in);}catch(const std::exception& e){mexErrMsgIdAndTxt("oomao:cuda:execution","%s",e.what());}}
