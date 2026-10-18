// Resample only phase samples used by the unchanged geometric gradient.
#include "mex.h"
#include "oomao_fast_threads.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
static const double* full(const mxArray* v) {
 if(!mxIsDouble(v)||mxIsSparse(v)||mxIsComplex(v))throw std::runtime_error("Full real double input required.");
 return mxGetDoubles(v);
}
static double extended(const double* p,ptrdiff_t r,ptrdiff_t c,ptrdiff_t n) {
 if(c<0)return 3*extended(p,r,0,n)-3*extended(p,r,1,n)+extended(p,r,2,n);
 if(c>=n)return 3*extended(p,r,n-1,n)-3*extended(p,r,n-2,n)+extended(p,r,n-3,n);
 if(r<0)return 3*p[c*n]-3*p[1+c*n]+p[2+c*n];
 if(r>=n)return 3*p[n-1+c*n]-3*p[n-2+c*n]+p[n-3+c*n];
 return p[r+c*n];
}
static void weights(double t,double* w) {
 w[0]=(-.5*t+t*t)-.5*t*t*t;w[1]=1-2.5*t*t+1.5*t*t*t;
 w[2]=.5*t+2*t*t-1.5*t*t*t;w[3]=-.5*t*t+.5*t*t*t;
}
static void run(mxArray*& output,const mxArray* in[]) {
 const mxArray* gt=in[0];
 if(!mxIsSparse(gt)||!mxIsDouble(gt)||mxIsComplex(gt))throw std::runtime_error("Transposed sparse real gradient required.");
 const double* phase=full(in[1]);const double* pixels=full(in[2]);const double* offset=full(in[3]);const double* grid=full(in[4]);
 const size_t n=mxGetM(in[1]),pupil=mxGetNumberOfElements(in[2]),rows=mxGetN(gt),threads=oomaoThreads(in[5]);
 if(n<4||mxGetN(in[1])!=n||mxGetNumberOfDimensions(in[1])!=2||mxGetM(gt)!=pupil||mxGetNumberOfElements(in[3])!=2||mxGetNumberOfElements(in[4])!=n)
  throw std::runtime_error("Gradient, square phase, pupil indices and axis shapes differ.");
 if(!std::isfinite(offset[0])||!std::isfinite(offset[1])||grid[0]!=-.5||grid[n-1]!=.5)throw std::runtime_error("Finite offsets and normalized axis required.");
 for(size_t k=0;k<pupil;++k)if(pixels[k]<1||pixels[k]>n*n||pixels[k]!=std::floor(pixels[k]))throw std::runtime_error("Pupil index outside phase.");
 const double* values=mxGetDoubles(gt);const mwIndex* columns=mxGetJc(gt);const mwIndex* indices=mxGetIr(gt);
 const bool shifted=offset[0]!=0||offset[1]!=0;
 std::vector<ptrdiff_t> ix(n),iy(n);std::vector<double> wx(4*n),wy(4*n);std::vector<bool> vx(n),vy(n);
 if(shifted)for(size_t k=0;k<n;++k){
  const double x=grid[k]-offset[0],y=grid[k]-offset[1];vx[k]=x>=grid[0]&&x<=grid[n-1];vy[k]=y>=grid[0]&&y<=grid[n-1];
  const double sx=(x-grid[0])/(grid[n-1]-grid[0])*(n-1),sy=(y-grid[0])/(grid[n-1]-grid[0])*(n-1);
  ix[k]=ptrdiff_t(std::floor(sx))-1;iy[k]=ptrdiff_t(std::floor(sy))-1;weights(sx-std::floor(sx),wx.data()+4*k);weights(sy-std::floor(sy),wy.data()+4*k);
 }
 output=mxCreateDoubleMatrix(rows,1,mxREAL);double* slopes=mxGetDoubles(output);
 oomaoPool().run(rows,threads,[&](size_t first,size_t last){
  for(size_t row=first;row<last;++row){double total=0;
   for(size_t j=columns[row];j<columns[row+1];++j){
    const size_t pixel=size_t(pixels[indices[j]])-1,r=pixel%n,c=pixel/n;double sample=phase[pixel];
    if(shifted){sample=0;if(vx[c]&&vy[r]){
     for(size_t x=0;x<4;++x){double vertical=0;for(size_t y=0;y<4;++y)vertical+=extended(phase,iy[r]+ptrdiff_t(y),ix[c]+ptrdiff_t(x),n)*wy[4*r+y];sample+=vertical*wx[4*c+x];}
    }}
    total+=values[j]*sample;
   }slopes[row]=total;
  }
 });
}
void mexFunction(int nlhs,mxArray* output[],int nrhs,const mxArray* input[]){
 if(nlhs!=1||nrhs!=6)mexErrMsgIdAndTxt("oomao:geometric:arguments","Gradient transpose, phase, pupil indices, offset, axis and threads required.");
 try{run(output[0],input);}catch(const std::exception& e){mexErrMsgIdAndTxt("oomao:geometric:input","%s",e.what());}
}
