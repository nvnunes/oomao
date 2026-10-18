// Selected-output DFT using cached roots and native complex BLAS.
#include "mex.h"
#include "blas.h"
#include "oomao_fast_threads.hpp"
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>
template<class T,class Z>static void gemm(char ta,char tb,ptrdiff_t m,ptrdiff_t n,ptrdiff_t inner,const Z* a,ptrdiff_t lda,const Z* b,ptrdiff_t ldb,Z* c,ptrdiff_t ldc){
 const T one[2]={1,0},zero[2]={0,0};
 if constexpr(sizeof(T)==sizeof(float))cgemm(&ta,&tb,&m,&n,&inner,one,reinterpret_cast<const float*>(a),&lda,reinterpret_cast<const float*>(b),&ldb,zero,reinterpret_cast<float*>(c),&ldc);
 else zgemm(&ta,&tb,&m,&n,&inner,one,reinterpret_cast<const double*>(a),&lda,reinterpret_cast<const double*>(b),&ldb,zero,reinterpret_cast<double*>(c),&ldc);
}
static int integer(const mxArray* a){if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||mxGetNumberOfElements(a)!=1)throw std::runtime_error("Integer scalar required.");double v=mxGetScalar(a);if(!std::isfinite(v)||v<0||v>16384||v!=std::floor(v))throw std::runtime_error("Integer outside 0..16384.");return int(v);}
template<class T,class Z>static void run(mxArray*& out,const mxArray* field,const mxArray* phase,int nf,int start,int crop,size_t threads){
 const size_t n=mxGetM(field),directions=mxGetN(field)/n;bool shifted=mxIsEmpty(phase);int shift=shifted?(nf+1)/2:0;
 using Key=std::array<int,4>;static std::map<Key,std::vector<Z>> cache;Key key={int(n),nf,start+shift,crop};auto found=cache.find(key);
 if(found==cache.end()){
  if(cache.size()>=16)cache.erase(cache.begin());std::vector<Z> roots(crop*n);
  for(size_t x=0;x<n;++x)for(int j=0;j<crop;++j){size_t index=((start+shift+j)%nf*x)%nf;double angle=-2*3.141592653589793238462643383279502884*double(index)/nf;roots[j+crop*x]={T(std::cos(angle)),T(std::sin(angle))};}
  found=cache.emplace(key,std::move(roots)).first;
 }
 const Z* z=mxIsComplex(field)?static_cast<const Z*>(mxGetData(field)):nullptr;const T* real=mxIsComplex(field)?nullptr:static_cast<const T*>(mxGetData(field));
 const mxComplexDouble* ph=shifted?nullptr:mxGetComplexDoubles(phase);std::vector<Z> prepared(n*n*directions),first(crop*n*directions);
 oomaoPool().run(n*n*directions,n>64?threads:1,[&](size_t begin,size_t end){for(size_t i=begin;i<end;++i){Z a=z?z[i]:Z{real[i],0};T re=a.real/T(nf),im=a.imag/T(nf);if(shifted)prepared[i]={re,im};else{auto p=ph[i%(n*n)];T pr=T(p.real),pi=T(p.imag);prepared[i]={re*pr-im*pi,re*pi+im*pr};}}});
 out=mxCreateNumericMatrix(crop,crop*directions,mxGetClassID(field),mxCOMPLEX);Z* result=static_cast<Z*>(mxGetData(out));const Z* roots=found->second.data();
 gemm<T,Z>('N','N',crop,n*directions,n,roots,crop,prepared.data(),n,first.data(),crop);
 for(size_t d=0;d<directions;++d)gemm<T,Z>('N','T',crop,crop,n,first.data()+d*crop*n,crop,roots,crop,result+d*crop*crop,crop);
}
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 try{
  if(nlhs!=1||nrhs!=6)throw std::runtime_error("Field, phasor, FFT size, start, crop and threads required.");
  auto f=in[0];auto p=in[1];int nf=integer(in[2]),start=integer(in[3]),crop=integer(in[4]);size_t threads=oomaoThreads(in[5]);size_t n=mxGetM(f);
  if(n<2||mxGetN(f)%n||mxGetNumberOfDimensions(f)!=2||mxIsSparse(f)||!(mxIsSingle(f)||mxIsDouble(f))||nf<int(n)||crop<1||start+crop>nf)throw std::runtime_error("Invalid pupil/FFT/crop geometry.");
  if(!mxIsEmpty(p)&&(!mxIsDouble(p)||!mxIsComplex(p)||mxIsSparse(p)||mxGetM(p)!=n||mxGetN(p)!=n))throw std::runtime_error("Empty or complex double pupil phasor required.");
  if(mxIsSingle(f))run<float,mxComplexSingle>(out[0],f,p,nf,start,crop,threads);else run<double,mxComplexDouble>(out[0],f,p,nf,start,crop,threads);
 }catch(const std::exception& e){mexErrMsgIdAndTxt("oomao:mft:input","%s",e.what());}
}
