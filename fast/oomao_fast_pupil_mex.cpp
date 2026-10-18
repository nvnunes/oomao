#include <exception>
#include "mex.h"
#include <cmath>
#include <algorithm>
#include "oomao_fast_threads.hpp"

// Mask and form the same double pupil fields directly in concatenated storage.
static void dispatch(int nlhs, mxArray **out, int nrhs, const mxArray **in) {
    if ((nrhs != 3 && nrhs != 4) || nlhs != 1 || !mxIsCell(in[0]) || !mxIsCell(in[1]) || !mxIsCell(in[2]))
        mexErrMsgIdAndTxt("oomao:pupil:args", "Expected amplitude, phase and mask cells.");
    const size_t threads=oomaoThreads(nrhs==4 ? in[3] : nullptr);
    const mwSize directions = mxGetNumberOfElements(in[0]);
    if (!directions || mxGetNumberOfElements(in[1]) != directions || mxGetNumberOfElements(in[2]) != directions)
        mexErrMsgIdAndTxt("oomao:pupil:cells", "Cell counts differ.");
    const mxArray *first = mxGetCell(in[2], 0);
    if (!first || !mxIsLogical(first) || mxIsSparse(first) || mxGetNumberOfDimensions(first) != 2)
        mexErrMsgIdAndTxt("oomao:pupil:mask", "Full two-dimensional logical masks required.");
    const mwSize rows = mxGetM(first), columns = mxGetN(first), pixels = rows * columns;
    if (!pixels) mexErrMsgIdAndTxt("oomao:pupil:empty", "Masks must be initialized.");
    mwSize waves = 1;
    for (mwIndex d = 0; d < directions; ++d) {
        const mxArray *a = mxGetCell(in[0], d), *p = mxGetCell(in[1], d), *m = mxGetCell(in[2], d);
        if (!a || !p || !m || !mxIsDouble(a) || !mxIsDouble(p) || mxIsComplex(a) || mxIsComplex(p) ||
                mxIsSparse(a) || mxIsSparse(p) || !mxIsLogical(m) || mxIsSparse(m) ||
                mxGetNumberOfDimensions(m) != 2 || mxGetM(m) != rows || mxGetN(m) != columns)
            mexErrMsgIdAndTxt("oomao:pupil:type", "Real full doubles and matching logical masks required.");
        const mwSize nphase = mxGetNumberOfElements(p);
        if (nphase != 1 && (mxGetM(p) != rows || nphase % pixels))
            mexErrMsgIdAndTxt("oomao:pupil:phase", "Phase shape is incompatible with the mask.");
        waves = std::max(waves, nphase == 1 ? mwSize(1) : nphase / pixels);
    }
    for (mwIndex d = 0; d < directions; ++d) {
        const mwSize a = mxGetNumberOfElements(mxGetCell(in[0], d));
        const mwSize p = mxGetNumberOfElements(mxGetCell(in[1], d));
        if ((a != 1 && a != pixels && a != pixels * waves) ||
                (p != 1 && p != pixels && p != pixels * waves))
            mexErrMsgIdAndTxt("oomao:pupil:broadcast", "Unsupported amplitude or phase broadcasting.");
    }
    const mwSize shape[] = {rows, columns * directions, waves};
    out[0] = mxCreateNumericArray(3, shape, mxDOUBLE_CLASS, mxCOMPLEX);
    mxComplexDouble *z = mxGetComplexDoubles(out[0]);
    struct Input { const double *a,*p;const mxLogical* m;size_t na,np; };
    std::vector<Input> inputs(directions);
    for(size_t d=0;d<directions;++d) {
        inputs[d]={mxGetDoubles(mxGetCell(in[0],d)),mxGetDoubles(mxGetCell(in[1],d)),
            mxGetLogicals(mxGetCell(in[2],d)),mxGetNumberOfElements(mxGetCell(in[0],d)),
            mxGetNumberOfElements(mxGetCell(in[1],d))};
    }
    oomaoPool().run(pixels*directions*waves,pixels>=4096 ? threads : 1,[&](size_t begin,size_t end) {
        while(begin<end) {
            const size_t wave=begin/(pixels*directions), d=(begin/pixels)%directions;
            const size_t first=begin%pixels, stop=std::min(pixels,first+end-begin);
            const auto& q=inputs[d];
            for(size_t i=first;i<stop;++i) {
                const double amplitude=q.a[q.na==1 ? 0 : i+(q.na>pixels ? wave*pixels : 0)];
                const double phase=q.p[q.np==1 ? 0 : i+(q.np>pixels ? wave*pixels : 0)];
                auto& v=z[begin+(i-first)];
                if(!q.m[i] && std::isfinite(amplitude) && std::isfinite(phase)) { v.real=0;v.imag=0; }
                else { const double a=amplitude*double(q.m[i]),p=phase*double(q.m[i]);
                    v.real=a*std::cos(p);v.imag=a*std::sin(p); }
            }
            begin+=stop-first;
        }
    });
}

void mexFunction(int nlhs, mxArray* out[], int nrhs, const mxArray* in[]) {
    try { dispatch(nlhs,out,nrhs,in); }
    catch(const std::exception& error) { mexErrMsgIdAndTxt("oomao:cpu:exception","%s",error.what()); }
    catch(...) { mexErrMsgIdAndTxt("oomao:cpu:exception","Native CPU execution failed."); }
}
