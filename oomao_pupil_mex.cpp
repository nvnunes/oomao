#include "mex.h"
#include <cmath>
#include <algorithm>

// Mask and form the same double pupil fields directly in concatenated storage.
void mexFunction(int nlhs, mxArray **out, int nrhs, const mxArray **in) {
    if (nrhs != 3 || nlhs != 1 || !mxIsCell(in[0]) || !mxIsCell(in[1]) || !mxIsCell(in[2]))
        mexErrMsgIdAndTxt("oomao:pupil:args", "Expected amplitude, phase and mask cells.");
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
    for (mwIndex d = 0; d < directions; ++d) {
        const mxArray *aa = mxGetCell(in[0], d), *pp = mxGetCell(in[1], d);
        const double *a = mxGetDoubles(aa), *p = mxGetDoubles(pp);
        const mxLogical *m = mxGetLogicals(mxGetCell(in[2], d));
        const mwSize na = mxGetNumberOfElements(aa), np = mxGetNumberOfElements(pp);
        for (mwIndex w = 0; w < waves; ++w) {
            for (mwIndex i = 0; i < pixels; ++i) {
                const double amplitude = a[na == 1 ? 0 : i + (na > pixels ? w * pixels : 0)];
                const double phase = p[np == 1 ? 0 : i + (np > pixels ? w * pixels : 0)];
                mxComplexDouble &v = z[w * pixels * directions + d * pixels + i];
                if (!m[i] && std::isfinite(amplitude) && std::isfinite(phase)) {
                    v.real = 0; v.imag = 0;
                } else {
                    const double maskedAmplitude = amplitude * double(m[i]);
                    const double maskedPhase = phase * double(m[i]);
                    v.real = maskedAmplitude * std::cos(maskedPhase);
                    v.imag = maskedAmplitude * std::sin(maskedPhase);
                }
            }
        }
    }
}
