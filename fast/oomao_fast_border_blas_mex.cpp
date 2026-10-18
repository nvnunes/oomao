// Probe native BLAS triangular B versus two dense matrix-vector products.
#include "mex.h"
#include "blas.h"
#include <vector>
#include <cmath>
static const double* data(const mxArray* a){if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a))mexErrMsgIdAndTxt("oomao:border:type","Real dense double input required.");return mxGetDoubles(a);}
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 if(nlhs==1&&nrhs==1){const double* b=data(in[0]);size_t m=mxGetM(in[0]);if(mxGetN(in[0])!=m)mexErrMsgIdAndTxt("oomao:border:shape","Square matrix required.");bool lower=true;for(size_t col=0;col<m&&lower;++col)for(size_t row=0;row<col;++row)if(b[row+col*m]!=0){lower=false;break;}out[0]=mxCreateLogicalScalar(lower);return;}
 if(nlhs!=1||nrhs!=5)mexErrMsgIdAndTxt("oomao:border:args","A,Z,B,noise,mode required.");
 const double* a=data(in[0]);const double* z=data(in[1]);const double* b=data(in[2]);const double* noise=data(in[3]);const double* mode=data(in[4]);
 ptrdiff_t m=mxGetM(in[0]),n=mxGetN(in[0]),stride=1;
 if(m<1||n<1||mxGetNumberOfElements(in[1])!=size_t(n)||mxGetM(in[2])!=size_t(m)||mxGetN(in[2])!=size_t(m)||mxGetNumberOfElements(in[3])!=size_t(m)||mxGetNumberOfElements(in[4])!=1||(mode[0]!=0&&mode[0]!=1))mexErrMsgIdAndTxt("oomao:border:shape","Input shapes or mode differ.");
 out[0]=mxCreateDoubleMatrix(m,1,mxREAL);double* result=mxGetDoubles(out[0]);std::vector<double> work(m);const char trans='N',lower='L',nonunit='N';const double one=1,zero=0;
 dgemv(&trans,&m,&n,&one,a,&m,z,&stride,&zero,result,&stride);
 if(mode[0]==1){for(ptrdiff_t k=0;k<m;++k)work[k]=noise[k];dtrmv(&lower,&trans,&nonunit,&m,b,&m,work.data(),&stride);}
 else dgemv(&trans,&m,&m,&one,b,&m,noise,&stride,&zero,work.data(),&stride);
 for(ptrdiff_t k=0;k<m;++k)result[k]=result[k]+work[k];
}
