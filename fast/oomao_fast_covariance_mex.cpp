// Double von Karman covariance, with optional ordered layer accumulation.
// Geometry, coefficients, layer order and RNG remain owned by MATLAB.
#include "mex.h"
#include "bessel_five_sixths.hpp"
#include "oomao_fast_threads.hpp"
#include <atomic>
#include <exception>
#include <limits>

struct Coordinates {
    const double* real;
    const mxComplexDouble* complex;
    size_t count;
    double re(size_t i) const {return complex ? complex[i].real:real[i];}
    double im(size_t i) const {return complex ? complex[i].imag:0;}
};
static Coordinates coordinates(const mxArray* value) {
    if(!mxIsDouble(value)||mxIsSparse(value)) mexErrMsgIdAndTxt("oomao:covariance:coordinates","Full double coordinates required.");
    Coordinates out{mxIsComplex(value)?nullptr:mxGetDoubles(value),mxIsComplex(value)?mxGetComplexDoubles(value):nullptr,mxGetNumberOfElements(value)};
    for(size_t i=0;i<out.count;++i) if(!std::isfinite(out.re(i))||!std::isfinite(out.im(i)))
        mexErrMsgIdAndTxt("oomao:covariance:coordinates","Coordinates must be finite.");
    return out;
}
static const double* coefficients(const mxArray* value,size_t count) {
    if(!mxIsDouble(value)||mxIsComplex(value)||mxIsSparse(value)||mxGetNumberOfElements(value)!=count)
        mexErrMsgIdAndTxt("oomao:covariance:coefficient","One finite real double coefficient per layer required.");
    const double* out=mxGetDoubles(value);
    for(size_t k=0;k<count;++k) if(!std::isfinite(out[k])) mexErrMsgIdAndTxt("oomao:covariance:coefficient","Coefficients must be finite.");
    return out;
}
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {
    if(nlhs!=1||(nrhs!=7&&nrhs!=8&&nrhs!=9)) mexErrMsgIdAndTxt("oomao:covariance:arguments","Use kernel(z1,z2,L0,cst,cst0,fraction,threads[,layers[,initial]]).");
    const bool fused=nrhs>7;
    size_t layers=1;
    if(fused) {
        const double n=*coefficients(prhs[7],1);
        if(n<1||n!=std::floor(n)||n>10000) mexErrMsgIdAndTxt("oomao:covariance:layers","A positive layer count up to 10000 is required.");
        layers=static_cast<size_t>(n);
        if(mxGetNumberOfDimensions(prhs[0])>2||mxGetNumberOfDimensions(prhs[1])>2||mxGetN(prhs[0])!=layers||mxGetN(prhs[1])!=layers)
            mexErrMsgIdAndTxt("oomao:covariance:shape","Coordinates must be points by layers.");
    }
    const Coordinates a=coordinates(prhs[0]),b=coordinates(prhs[1]);
    const size_t rows=a.count/layers,cols=b.count/layers;
    const double* L0=coefficients(prhs[2],layers);
    const double* constant=coefficients(prhs[3],layers);
    const double* zero=coefficients(prhs[4],layers);
    const double* fraction=coefficients(prhs[5],layers);
    for(size_t k=0;k<layers;++k) if(L0[k]<=0) mexErrMsgIdAndTxt("oomao:covariance:L0","L0 must be positive.");
    const size_t threads=oomaoThreads(prhs[6]);
    const double* initial=nullptr;
    if(nrhs==9) {
        if(mxGetNumberOfDimensions(prhs[8])>2||mxGetM(prhs[8])!=rows||mxGetN(prhs[8])!=cols)
            mexErrMsgIdAndTxt("oomao:covariance:initial","Initial covariance shape differs.");
        initial=coefficients(prhs[8],rows*cols);
    }
    if(rows && cols>std::numeric_limits<size_t>::max()/rows) mexErrMsgIdAndTxt("oomao:covariance:size","Covariance dimensions overflow.");
    plhs[0]=mxCreateUninitNumericMatrix(rows,cols,mxDOUBLE_CLASS,mxREAL);
    double* output=mxGetDoubles(plhs[0]);
    if(!rows||!cols) return;
    std::atomic<bool> failed{false};
    try {
        oomaoPool().run(cols,threads,[&](size_t first,size_t last) {
            constexpr double twoPi=2*3.141592653589793238462643383279502884;
            if(fused) for(size_t j=first;j<last;++j) for(size_t i=0;i<rows;++i) {
                const size_t index=i+rows*j;output[index]=initial?initial[index]:0;
            }
            // Traverse each layer's contiguous columns, keeping its constants
            // and coordinates hot; accumulation order is still k=0..layers-1.
            for(size_t k=0;k<layers;++k) for(size_t j=first;j<last;++j) for(size_t i=0;i<rows;++i) {
                const size_t index=i+rows*j,ai=i+rows*k,bj=j+cols*k;
                const double distance=std::hypot(a.re(ai)-b.re(bj),a.im(ai)-b.im(bj));
                double value=zero[k];
                if(distance!=0) {
                    const double u=twoPi*distance/L0[k];
                    value=constant[k]*std::pow(u,5.0/6.0)*oomaoBesselFiveSixths(u);
                }
                const double term=fraction[k]*value;
                if(fused) output[index]=output[index]+term; else output[index]=term;
                if(!std::isfinite(output[index])) failed.store(true,std::memory_order_relaxed);
            }
        });
    } catch(const std::exception& error) {mexErrMsgIdAndTxt("oomao:covariance:runtime","%s",error.what());}
    if(failed.load()) mexErrMsgIdAndTxt("oomao:covariance:numerical","Bessel evaluation did not converge to a finite covariance.");
}
