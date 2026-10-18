#include <exception>
#include "mex.h"
#include "blas.h"
#include "oomao_fast_threads.hpp"
#include <vector>
#include <cmath>

static const double* realFull(const mxArray* a) {
    if(!a||!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a))
        mexErrMsgIdAndTxt("oomao:cpu:type","Real full double data required.");
    return mxGetDoubles(a);
}
static double scalar(const mxArray* a) {
    const double* p=realFull(a);
    if(mxGetNumberOfElements(a)!=1) mexErrMsgIdAndTxt("oomao:cpu:scalar","Scalar required.");
    return p[0];
}
static void reconstruction(mxArray*& out,const mxArray* cells,const mxArray* projector,const mxArray* slopes) {
    if(!mxIsCell(cells)||mxGetNumberOfElements(cells)==0)
        mexErrMsgIdAndTxt("oomao:cpu:cells","Nonempty reconstructor cell required.");
    const double* p=realFull(projector);const double* s=realFull(slopes);
    const size_t m=mxGetM(projector),n=mxGetNumberOfElements(slopes),d=mxGetNumberOfElements(cells);
    if(mxGetN(projector)!=m) mexErrMsgIdAndTxt("oomao:cpu:shape","Square projector required.");
    std::vector<const double*> matrices(d);
    for(size_t i=0;i<d;++i) {
        const mxArray* a=mxGetCell(cells,i);matrices[i]=realFull(a);
        if(mxGetM(a)!=m||mxGetN(a)!=n) mexErrMsgIdAndTxt("oomao:cpu:shape","Reconstructor shape differs.");
    }
    out=mxCreateDoubleMatrix(m,d,mxREAL);double* y=mxGetDoubles(out);
    std::vector<double> work(m);
    const char trans='N';const ptrdiff_t mm=m,nn=n,stride=1;const double one=1,zero=0;
    for(size_t i=0;i<d;++i) {
        dgemv(&trans,&mm,&nn,&one,matrices[i],&mm,s,&stride,&zero,work.data(),&stride);
        dgemv(&trans,&mm,&mm,&one,p,&mm,work.data(),&stride,&zero,y+i*m,&stride);
    }
}
static void accumulate(mxArray*& out,const mxArray* buffer,const mxArray* image,size_t threads) {
    const double* a=realFull(buffer);const size_t na=mxGetNumberOfElements(buffer),n=mxGetNumberOfElements(image);
    if(mxIsComplex(image)||mxIsSparse(image)||!(mxIsDouble(image)||mxIsSingle(image))||(na!=1&&na!=n))
        mexErrMsgIdAndTxt("oomao:cpu:image","Real float image and scalar/matching double buffer required.");
    out=mxCreateNumericArray(mxGetNumberOfDimensions(image),mxGetDimensions(image),mxDOUBLE_CLASS,mxREAL);
    double* y=mxGetDoubles(out);
    if(mxIsSingle(image)) { const float* b=mxGetSingles(image);
        oomaoPool().run(n,n>=65536 ? threads:1,[&](size_t first,size_t last){for(size_t k=first;k<last;++k)y[k]=a[na==1?0:k]+double(b[k]);});
    } else { const double* b=mxGetDoubles(image);
        oomaoPool().run(n,n>=65536 ? threads:1,[&](size_t first,size_t last){for(size_t k=first;k<last;++k)y[k]=a[na==1?0:k]+b[k];});
    }
}
static double extended(const double* p,ptrdiff_t r,ptrdiff_t c,ptrdiff_t n) {
    if(c<0)return 3*extended(p,r,0,n)-3*extended(p,r,1,n)+extended(p,r,2,n);
    if(c>=n)return 3*extended(p,r,n-1,n)-3*extended(p,r,n-2,n)+extended(p,r,n-3,n);
    if(r<0)return 3*p[c*n]-3*p[1+c*n]+p[2+c*n];
    if(r>=n)return 3*p[n-1+c*n]-3*p[n-2+c*n]+p[n-3+c*n];
    return p[r+c*n];
}
static void weights(double t,double* w) {
    w[0]=(-0.5*t+t*t)-0.5*t*t*t;
    w[1]=1-2.5*t*t+1.5*t*t*t;
    w[2]=0.5*t+2*t*t-1.5*t*t*t;
    w[3]=-0.5*t*t+0.5*t*t*t;
}
static void cubicShift(mxArray*& out,const mxArray* image,const mxArray* residual,size_t threads) {
    const double* p=realFull(image);const double* delta=realFull(residual);
    const size_t side=mxGetM(image),n=side-2;
    if(side<5||mxGetN(image)!=side||mxGetNumberOfElements(residual)!=2||
            !std::isfinite(delta[0])||!std::isfinite(delta[1])||std::abs(delta[0])>=1||std::abs(delta[1])>=1)
        mexErrMsgIdAndTxt("oomao:cpu:wind","Square bordered screen and fractional displacement required.");
    std::vector<ptrdiff_t> ix(n),iy(n);std::vector<double> wx(4*n),wy(4*n);
    for(size_t k=0;k<n;++k) {
        const double x=(double(k)-delta[0])+1,y=(double(k)-delta[1])+1;
        ix[k]=ptrdiff_t(std::floor(x))-1;iy[k]=ptrdiff_t(std::floor(y))-1;
        weights(x-std::floor(x),wx.data()+4*k);weights(y-std::floor(y),wy.data()+4*k);
    }
    // Separable cubic interpolation: four vertical taps, then four horizontal.
    // Reuse C++ scratch; no MATLAB-owned pointer survives the call.
    static std::vector<double> vertical;
    vertical.resize(n*(side+2));
    oomaoPool().run(side,n>=64?threads:1,[&](size_t first,size_t last){
        for(size_t c=first;c<last;++c)for(size_t r=0;r<n;++r) {
            double value=0;
            const ptrdiff_t y=iy[r];
            if(y>=0&&y+3<ptrdiff_t(side)) {
                for(size_t k=0;k<4;++k)value+=p[y+k+c*side]*wy[4*r+k];
            } else for(size_t k=0;k<4;++k)value+=extended(p,y+k,c,side)*wy[4*r+k];
            vertical[r+n*(c+1)]=value;
        }
    });
    for(size_t r=0;r<n;++r) {
        vertical[r]=3*vertical[r+n]-3*vertical[r+2*n]+vertical[r+3*n];
        vertical[r+n*(side+1)]=3*vertical[r+n*side]-3*vertical[r+n*(side-1)]+vertical[r+n*(side-2)];
    }
    out=mxCreateDoubleMatrix(n,n,mxREAL);double* q=mxGetDoubles(out);
    oomaoPool().run(n,n>=64?threads:1,[&](size_t first,size_t last){
        for(size_t c=first;c<last;++c)for(size_t r=0;r<n;++r) {
            double value=0;
            for(size_t k=0;k<4;++k)value+=vertical[r+n*(ix[c]+1+k)]*wx[4*c+k];
            q[r+c*n]=value;
        }
    });
}

static void csrProduct(mxArray*& out,const mxArray* transpose,const mxArray* coefficients,size_t threads) {
    if(!mxIsSparse(transpose)||!mxIsDouble(transpose)||mxIsComplex(transpose)||
            mxGetM(transpose)!=mxGetM(coefficients))
        mexErrMsgIdAndTxt("oomao:cpu:csr","Transposed sparse real matrix and compatible dense coefficients required.");
    const double* b=realFull(coefficients);const double* values=mxGetDoubles(transpose);
    const mwIndex* modes=mxGetIr(transpose);const mwIndex* columns=mxGetJc(transpose);
    const size_t n=mxGetM(transpose),pixels=mxGetN(transpose),rhs=mxGetN(coefficients);
    out=mxCreateDoubleMatrix(pixels,rhs,mxREAL);double* output=mxGetDoubles(out);
    if(rhs==0)return;
    oomaoPool().run(pixels,threads,[&](size_t begin,size_t end){
        for(size_t pixel=begin;pixel<end;++pixel)for(size_t source=0;source<rhs;++source) {
            double value=0;
            for(size_t k=columns[pixel];k<columns[pixel+1];++k)value+=values[k]*b[modes[k]+source*n];
            output[pixel+pixels*source]=value;
        }
    });
}

static bool sameStorage(const mxArray* a,const mxArray* b){
 if(mxGetClassID(a)!=mxGetClassID(b)||mxIsSparse(a)!=mxIsSparse(b)||mxIsComplex(a)!=mxIsComplex(b)||mxGetNumberOfDimensions(a)!=mxGetNumberOfDimensions(b))return false;
 for(size_t k=0;k<mxGetNumberOfDimensions(a);++k)if(mxGetDimensions(a)[k]!=mxGetDimensions(b)[k])return false;
 if(mxIsComplex(a)||(!mxIsNumeric(a)&&!mxIsLogical(a)))return false;
 return mxGetData(a)==mxGetData(b)&&(!mxIsSparse(a)||(mxGetIr(a)==mxGetIr(b)&&mxGetJc(a)==mxGetJc(b)));
}
static void dispatch(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]) {
    if(nlhs==1&&nrhs==3&&scalar(in[0])==7){out[0]=mxCreateLogicalScalar(sameStorage(in[1],in[2]));return;}
    if(nlhs!=1||nrhs<4) mexErrMsgIdAndTxt("oomao:cpu:args","One output and operation/input arguments required.");
    const double op=scalar(in[0]);
    if(op==2&&nrhs==4)reconstruction(out[0],in[1],in[2],in[3]);
    else if(op==3&&nrhs==4)accumulate(out[0],in[1],in[2],oomaoThreads(in[3]));
    else if(op==6&&nrhs==4)csrProduct(out[0],in[1],in[2],oomaoThreads(in[3]));
    else if(op==5&&nrhs==4)cubicShift(out[0],in[1],in[2],oomaoThreads(in[3]));
    else mexErrMsgIdAndTxt("oomao:cpu:operation","Unknown operation or argument count.");
}

void mexFunction(int nlhs, mxArray* out[], int nrhs, const mxArray* in[]) {
    try { dispatch(nlhs,out,nrhs,in); }
    catch(const std::exception& error) { mexErrMsgIdAndTxt("oomao:cpu:exception","%s",error.what()); }
    catch(...) { mexErrMsgIdAndTxt("oomao:cpu:exception","Native CPU execution failed."); }
}
