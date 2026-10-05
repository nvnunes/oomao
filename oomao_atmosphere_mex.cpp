#include "mex.h"
#include <cmath>
#include <vector>

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

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 9 || nlhs != 1 || !mxIsCell(prhs[0]) || !mxIsCell(prhs[1]))
        mexErrMsgIdAndTxt("oomao:atmos:args", "Expected phase/sampling cells and seven geometry arrays.");
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
    std::vector<mwSize> ix(n), iy(n);
    std::vector<double> sx(n), sy(n);
    for (mwSize layer = 0; layer < layers; ++layer) {
        const mxArray* phase = mxGetCell(prhs[0], layer);
        const mxArray* sampling = mxGetCell(prhs[1], layer);
        const double* values = real_data(phase);
        const double* grid = real_data(sampling);
        const mwSize rows = mxGetM(phase), cols = mxGetN(phase);
        if (mxGetNumberOfDimensions(phase) != 2 || rows != cols || rows < 2 ||
            mxGetNumberOfElements(sampling) != rows)
            mexErrMsgIdAndTxt("oomao:atmos:grid", "Square phase screens and matching sampling required.");
        if (heights[layer] == 0.0 && rows == n) {
            for (mwSize pixel = 0; pixel < n*n; ++pixel) output[pixel] += values[pixel];
            continue;
        }
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
        for (mwSize col = 0; col < n; ++col) {
            for (mwSize row = 0; row < n; ++row) {
                const mwSize index = iy[row] + ix[col]*rows;
                const double onemt = 1.0-sy[row];
                const double sample =
                    (values[index]*onemt + values[index+1]*sy[row])*(1.0-sx[col]) +
                    (values[index+rows]*onemt + values[index+rows+1]*sy[row])*sx[col];
                output[row+col*n] += sample;
            }
        }
    }
}
