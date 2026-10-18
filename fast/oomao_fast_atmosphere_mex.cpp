#include <exception>
#include "mex.h"
#include <cmath>
#include <vector>
#include "oomao_fast_threads.hpp"

// Double-precision bilinear sampling with direct layer accumulation.
// Compile without contraction or fast-math; atmospheric evolution stays MATLAB.
static const double* real_data(const mxArray* a) {
    if (!a || !mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a))
        mexErrMsgIdAndTxt("oomao:atmos:type", "Real full double arrays required.");
    return mxGetPr(a);
}

static double scalar(const mxArray* a) {
    const double* value = real_data(a);
    if (mxGetNumberOfElements(a) != 1)
        mexErrMsgIdAndTxt("oomao:atmos:scalar", "Scalar geometry required.");
    return value[0];
}

static void dispatch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if ((nrhs != 9 && nrhs != 10) || nlhs != 1 || !mxIsCell(prhs[0]) || !mxIsCell(prhs[1]))
        mexErrMsgIdAndTxt("oomao:atmos:args", "Expected phase/sampling cells and seven geometry arrays.");
    const size_t threads = oomaoThreads(nrhs==10 ? prhs[9] : nullptr);
    const mwSize layers = mxGetNumberOfElements(prhs[0]);
    const double* heights = real_data(prhs[2]);
    const double* sampler = real_data(prhs[4]);
    const double* direction = real_data(prhs[6]);
    const double* origin = real_data(prhs[7]);
    const double size = scalar(prhs[3]);
    const double radius = scalar(prhs[5]);
    const double source_height = scalar(prhs[8]);
    if (!std::isfinite(size) || size < 1 || size != mxGetNumberOfElements(prhs[4]) ||
        !std::isfinite(radius) || radius <= 0 || !(source_height > 0))
        mexErrMsgIdAndTxt("oomao:atmos:geometry", "Positive pupil size, radius and source height required.");
    const mwSize n = static_cast<mwSize>(size);
    if (!layers || !n || mxGetNumberOfElements(prhs[1]) != layers ||
        mxGetNumberOfElements(prhs[2]) != layers || mxGetNumberOfElements(prhs[4]) != n ||
        mxGetNumberOfElements(prhs[6]) != 2 || mxGetNumberOfElements(prhs[7]) != 2)
        mexErrMsgIdAndTxt("oomao:atmos:shape", "Geometry shapes differ.");
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double* output = mxGetPr(plhs[0]);
    struct Layer { const double* values; size_t rows; bool direct;
        std::vector<mwSize> ix,iy; std::vector<double> sx,sy; };
    std::vector<Layer> prepared(layers);
    for (mwSize layer = 0; layer < layers; ++layer) {
        const mxArray* phase = mxGetCell(prhs[0], layer);
        const mxArray* sampling = mxGetCell(prhs[1], layer);
        const double* values = real_data(phase);
        const double* grid = real_data(sampling);
        const mwSize rows = mxGetM(phase), cols = mxGetN(phase);
        if (mxGetNumberOfDimensions(phase) != 2 || rows != cols || rows < 2 ||
            mxGetNumberOfElements(sampling) != rows)
            mexErrMsgIdAndTxt("oomao:atmos:grid", "Square phase screens and matching sampling required.");
        auto& item=prepared[layer]; item.values=values; item.rows=rows;
        item.direct=heights[layer]==0.0 && rows==n;
        if(item.direct) continue;
        item.ix.resize(n);item.iy.resize(n);item.sx.resize(n);item.sy.resize(n);
        auto& ix=item.ix;auto& iy=item.iy;auto& sx=item.sx;auto& sy=item.sy;
        const double layer_radius = radius * (1.0 - heights[layer]/source_height);
        const double xc = heights[layer]*direction[0], yc = heights[layer]*direction[1];
        const double extent = grid[rows-1] - grid[0];
        for (mwSize pixel = 0; pixel < n; ++pixel) {
            const double u = sampler[pixel]*layer_radius;
            const double x = (u + xc) + origin[0], y = (u + yc) + origin[1];
            const double s = 1.0 + (x-grid[0])/extent*(cols-1);
            const double t = 1.0 + (y-grid[0])/extent*(rows-1);
            double col = std::floor(s-1.0), row = std::floor(t)-1.0;
            sx[pixel] = s-std::floor(s); sy[pixel] = t-std::floor(t);
            if (s == cols) { sx[pixel] += 1.0; col -= 1.0; }
            if (t == rows) { sy[pixel] += 1.0; row -= 1.0; }
            if (!std::isfinite(s) || !std::isfinite(t) || col < 0 || row < 0 || col+1 >= cols || row+1 >= rows)
                mexErrMsgIdAndTxt("oomao:atmos:bounds", "Sample lies outside phase-screen interpolation domain.");
            ix[pixel] = static_cast<mwSize>(col); iy[pixel] = static_cast<mwSize>(row);
        }
    }
    oomaoPool().run(n,n>=64 ? threads : 1,[&](size_t begin,size_t end) {
        for(size_t col=begin;col<end;++col) for(const auto& item:prepared) {
            if(item.direct) {
                for(size_t row=0;row<n;++row) output[row+col*n]+=item.values[row+col*n];
                continue;
            }
            for(size_t row=0;row<n;++row) {
                const size_t index=item.iy[row]+item.ix[col]*item.rows;
                const double onemt=1.0-item.sy[row];
                const double sample=(item.values[index]*onemt+item.values[index+1]*item.sy[row])*(1.0-item.sx[col])+
                    (item.values[index+item.rows]*onemt+item.values[index+item.rows+1]*item.sy[row])*item.sx[col];
                output[row+col*n]+=sample;
            }
        }
    });
}

void mexFunction(int nlhs, mxArray* out[], int nrhs, const mxArray* in[]) {
    try { dispatch(nlhs,out,nrhs,in); }
    catch(const std::exception& error) { mexErrMsgIdAndTxt("oomao:cpu:exception","%s",error.what()); }
    catch(...) { mexErrMsgIdAndTxt("oomao:cpu:exception","Native CPU execution failed."); }
}
