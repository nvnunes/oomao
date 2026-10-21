#include <exception>
#include "mex.h"
#include <cmath>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "oomao_fast_threads.hpp"

// Double-precision bilinear sampling with execution-owned immutable geometry.
// Compile without contraction or fast-math; atmospheric evolution stays MATLAB.
struct AtmosphereError : std::runtime_error {
 const char* id;AtmosphereError(const char* key,const char* message):std::runtime_error(message),id(key){}
};
static const double* real_data(const mxArray* a) {
    if (!a || !mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a))
        throw AtmosphereError("oomao:atmos:type","Real full double arrays required.");
    return mxGetPr(a);
}

static double scalar(const mxArray* a) {
    const double* value = real_data(a);
    if (mxGetNumberOfElements(a) != 1)
        throw AtmosphereError("oomao:atmos:scalar","Scalar geometry required.");
    return value[0];
}

struct GeometryLayer { size_t rows; bool direct;
    std::vector<mwSize> ix,iy; std::vector<double> sx,sy; };
struct GeometryCache { std::map<std::vector<uint64_t>,std::vector<GeometryLayer>> entries; };
static std::map<uint64_t,std::unique_ptr<GeometryCache>> caches;
static uint64_t nextCache=1;
static void cleanup(){caches.clear();oomaoPoolCleanup();}
static uint64_t cacheHandle(const mxArray* a){
    if(!mxIsUint64(a)||mxGetNumberOfElements(a)!=1)throw std::runtime_error("Owned atmosphere cache handle required.");
    return *static_cast<const uint64_t*>(mxGetData(a));
}
static void append(std::vector<uint64_t>& key,const double* values,size_t count){
    size_t base=key.size();key.resize(base+count);std::memcpy(key.data()+base,values,count*8);
}
static void dispatch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if(nrhs&&mxIsChar(prhs[0])){
        char* raw=mxArrayToString(prhs[0]);std::string operation(raw?raw:"");mxFree(raw);
        if(operation=="create"&&nrhs==1&&nlhs==1){
            auto cache=std::make_unique<GeometryCache>();uint64_t id=nextCache++;
            plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(plhs[0]))=id;
            caches.emplace(id,std::move(cache));mexLock();mexAtExit(cleanup);return;
        }
        if(operation=="destroy"&&nrhs==2&&nlhs==0){
            if(!caches.erase(cacheHandle(prhs[1])))throw std::runtime_error("Unknown atmosphere cache handle.");
            mexUnlock();return;
        }
        throw std::runtime_error("Invalid atmosphere cache operation.");
    }
    GeometryCache* cache=nullptr;
    if(nrhs==11){auto found=caches.find(cacheHandle(prhs[10]));if(found==caches.end())throw std::runtime_error("Unknown atmosphere cache handle.");cache=found->second.get();}
    if ((nrhs != 9 && nrhs != 10 && nrhs != 11) || nlhs != 1 || !mxIsCell(prhs[0]) || !mxIsCell(prhs[1]))
        throw AtmosphereError("oomao:atmos:args","Expected phase/sampling cells and seven geometry arrays.");
    const size_t threads = oomaoThreads(nrhs>=10 ? prhs[9] : nullptr);
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
        throw AtmosphereError("oomao:atmos:geometry","Positive pupil size, radius and source height required.");
    const mwSize n = static_cast<mwSize>(size);
    if (!layers || !n || mxGetNumberOfElements(prhs[1]) != layers ||
        mxGetNumberOfElements(prhs[2]) != layers || mxGetNumberOfElements(prhs[4]) != n ||
        mxGetNumberOfElements(prhs[6]) != 2 || mxGetNumberOfElements(prhs[7]) != 2)
        throw AtmosphereError("oomao:atmos:shape","Geometry shapes differ.");
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double* output = mxGetPr(plhs[0]);
    std::vector<const double*> phaseValues(layers);
    std::vector<uint64_t> key;
    if(cache){
    key={uint64_t(n),uint64_t(layers)};
    append(key,&radius,1);append(key,&source_height,1);append(key,direction,2);append(key,origin,2);
    append(key,heights,layers);append(key,sampler,n);
    for(mwSize layer=0;layer<layers;++layer){
        const mxArray* phase=mxGetCell(prhs[0],layer);const mxArray* sampling=mxGetCell(prhs[1],layer);
        phaseValues[layer]=real_data(phase);const double* grid=real_data(sampling);size_t rows=mxGetM(phase);
        if(mxGetNumberOfDimensions(phase)!=2||rows!=mxGetN(phase)||rows<2||mxGetNumberOfElements(sampling)!=rows)
            throw AtmosphereError("oomao:atmos:grid","Square phase screens and matching sampling required.");
        key.push_back(rows);append(key,grid,1);append(key,grid+rows-1,1);
    }
    }
    std::vector<GeometryLayer> local;
    const std::vector<GeometryLayer>* geometry=nullptr;
    if(cache){auto found=cache->entries.find(key);if(found!=cache->entries.end())geometry=&found->second;}
    if(!geometry){
    local.resize(layers);auto& prepared=local;
    for (mwSize layer = 0; layer < layers; ++layer) {
        const mxArray* phase = mxGetCell(prhs[0], layer);
        const mxArray* sampling = mxGetCell(prhs[1], layer);
        const double* values = real_data(phase);
        const double* grid = real_data(sampling);
        const mwSize rows = mxGetM(phase), cols = mxGetN(phase);
        if (mxGetNumberOfDimensions(phase) != 2 || rows != cols || rows < 2 ||
            mxGetNumberOfElements(sampling) != rows)
            throw AtmosphereError("oomao:atmos:grid","Square phase screens and matching sampling required.");
        phaseValues[layer]=values;
        auto& item=prepared[layer]; item.rows=rows;
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
                throw AtmosphereError("oomao:atmos:bounds","Sample lies outside phase-screen interpolation domain.");
            ix[pixel] = static_cast<mwSize>(col); iy[pixel] = static_cast<mwSize>(row);
        }
    }
    if(cache){
        if(cache->entries.size()>=256)cache->entries.clear();
        geometry=&cache->entries.emplace(std::move(key),std::move(local)).first->second;
    }else geometry=&local;
    }
    oomaoPool().run(n,n>=64 ? threads : 1,[&](size_t begin,size_t end) {
        for(size_t col=begin;col<end;++col) for(size_t layer=0;layer<layers;++layer) {
            const auto& item=(*geometry)[layer];const double* values=phaseValues[layer];
            if(item.direct) {
                for(size_t row=0;row<n;++row) output[row+col*n]+=values[row+col*n];
                continue;
            }
            for(size_t row=0;row<n;++row) {
                const size_t index=item.iy[row]+item.ix[col]*item.rows;
                const double onemt=1.0-item.sy[row];
                const double sample=(values[index]*onemt+values[index+1]*item.sy[row])*(1.0-item.sx[col])+
                    (values[index+item.rows]*onemt+values[index+item.rows+1]*item.sy[row])*item.sx[col];
                output[row+col*n]+=sample;
            }
        }
    });
}

void mexFunction(int nlhs, mxArray* out[], int nrhs, const mxArray* in[]) {
    try { dispatch(nlhs,out,nrhs,in); }
    catch(const AtmosphereError& error) { mexErrMsgIdAndTxt(error.id,"%s",error.what()); }
    catch(const std::exception& error) { mexErrMsgIdAndTxt("oomao:cpu:exception","%s",error.what()); }
    catch(...) { mexErrMsgIdAndTxt("oomao:cpu:exception","Native CPU execution failed."); }
}
