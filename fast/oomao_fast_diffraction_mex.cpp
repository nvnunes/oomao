#include <exception>
#include "mex.h"
#include "oomao_fast_threads.hpp"
#include <vector>
#include <algorithm>

static size_t integer(const mxArray* a) {
    if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("oomao:fft:integer","Real integer scalar required.");
    double x=mxGetScalar(a);
    if(!std::isfinite(x)||x<0||x!=std::floor(x))mexErrMsgIdAndTxt("oomao:fft:integer","Nonnegative integer required.");
    return size_t(x);
}
static mxArray* transform(mxArray* a,size_t n) {
    mxArray* size=mxCreateDoubleScalar(n);mxArray* axis=mxCreateDoubleScalar(1);
    mxArray* args[]={a,size,axis};mxArray* result=nullptr;
    mxArray* error=mexCallMATLABWithTrap(1,&result,3,args,"fft");
    mxDestroyArray(size);mxDestroyArray(axis);
    if(error) { mxDestroyArray(a);mxDestroyArray(error);mexErrMsgIdAndTxt("oomao:fft:library","MATLAB FFT call failed."); }
    return result;
}
template<class T,class Z> struct ComplexView {
    const Z* complexData; const T* realData;
    Z get(size_t index) const { return complexData ? complexData[index] : Z{realData[index],T(0)}; }
};
template<class T,class Z> ComplexView<T,Z> view(const mxArray* value) {
    if(mxIsComplex(value))return {static_cast<const Z*>(mxGetData(value)),nullptr};
    return {nullptr,static_cast<const T*>(mxGetData(value))};
}
template<class T,class Z> void execute(mxArray*& out,const mxArray* field,const mxArray* phasor,
        size_t nout,size_t start,size_t crop,size_t threads) {
    const size_t n=mxGetM(field),directions=mxGetN(field)/n;
    const mxClassID type=mxGetClassID(field);
    const auto wave=view<T,Z>(field);
    const bool shifted=mxIsEmpty(phasor);
    const mxComplexDouble* phase=shifted?nullptr:mxGetComplexDoubles(phasor);
    std::vector<ptrdiff_t> active(n*directions,-1);size_t count=0;
    for(size_t col=0;col<n*directions;++col) {
        for(size_t row=0;row<n;++row) if(wave.get(row+n*col).real!=0||wave.get(row+n*col).imag!=0) {
            active[col]=count++;break;
        }
    }
    out=mxCreateNumericMatrix(crop,crop*directions,type,mxCOMPLEX);
    if(count==0)return;
    mxArray* first=mxCreateNumericMatrix(n,count,type,mxCOMPLEX);
    Z* buffer=static_cast<Z*>(mxGetData(first));
    oomaoPool().run(n*directions,threads,[&](size_t begin,size_t end){
        for(size_t col=begin;col<end;++col) if(active[col]>=0)for(size_t row=0;row<n;++row) {
            const auto v=wave.get(row+n*col);const T re=v.real/T(nout),im=v.imag/T(nout);
            auto& z=buffer[row+n*active[col]];
            if(shifted) {z.real=re;z.imag=im;}
            else {const auto p=phase[row+n*(col%n)];const T pr=T(p.real),pi=T(p.imag);
                z.real=re*pr-im*pi;z.imag=re*pi+im*pr;}
        }
    });
    mxArray* firstF=transform(first,nout);mxDestroyArray(first);
    const auto firstData=view<T,Z>(firstF);
    mxArray* second=mxCreateNumericMatrix(n,crop*directions,type,mxCOMPLEX);
    Z* reordered=static_cast<Z*>(mxGetData(second));
    oomaoPool().run(crop*directions,threads,[&](size_t begin,size_t end){
        for(size_t col=begin;col<end;++col) {
            const size_t d=col%directions,y=col/directions;
            const size_t freq=(start+y+(shifted?(nout+1)/2:0))%nout;
            for(size_t x=0;x<n;++x) if(active[x+n*d]>=0)
                reordered[x+n*col]=firstData.get(freq+nout*active[x+n*d]);
        }
    });
    mxDestroyArray(firstF);
    mxArray* secondF=transform(second,nout);mxDestroyArray(second);
    const auto secondData=view<T,Z>(secondF);
    Z* result=static_cast<Z*>(mxGetData(out));
    oomaoPool().run(crop*directions,threads,[&](size_t begin,size_t end){
        for(size_t col=begin;col<end;++col) {
            const size_t d=col/crop,x=col%crop;
            const size_t freq=(start+x+(shifted?(nout+1)/2:0))%nout;
            for(size_t y=0;y<crop;++y) result[y+crop*col]=secondData.get(freq+nout*(d+directions*y));
        }
    });
    mxDestroyArray(secondF);
}
static void dispatch(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]) {
    if(nlhs!=1||nrhs!=6)mexErrMsgIdAndTxt("oomao:fft:args","field, phasor, FFT size, crop start/width and threads required.");
    const mxArray* field=in[0];const mxArray* phasor=in[1];
    const size_t n=mxGetM(field),nout=integer(in[2]),start=integer(in[3]),crop=integer(in[4]),threads=oomaoThreads(in[5]);
    if(n<2||mxGetNumberOfDimensions(field)!=2||mxGetN(field)%n||mxIsSparse(field)||
        !(mxIsSingle(field)||mxIsDouble(field))||nout<n||crop==0||start+crop>nout)
        mexErrMsgIdAndTxt("oomao:fft:shape","Real/complex square pupil pages and valid FFT/crop geometry required.");
    if(!mxIsEmpty(phasor)&&(!mxIsDouble(phasor)||!mxIsComplex(phasor)||mxIsSparse(phasor)||mxGetM(phasor)!=n||mxGetN(phasor)!=n))
        mexErrMsgIdAndTxt("oomao:fft:phasor","Empty or complex double pupil-sized phasor required.");
    if(mxIsSingle(field))execute<float,mxComplexSingle>(out[0],field,phasor,nout,start,crop,threads);
    else execute<double,mxComplexDouble>(out[0],field,phasor,nout,start,crop,threads);
}

void mexFunction(int nlhs, mxArray* out[], int nrhs, const mxArray* in[]) {
    try { dispatch(nlhs,out,nrhs,in); }
    catch(const std::exception& error) { mexErrMsgIdAndTxt("oomao:cpu:exception","%s",error.what()); }
    catch(...) { mexErrMsgIdAndTxt("oomao:cpu:exception","Native CPU execution failed."); }
}
