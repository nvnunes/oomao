#ifndef OOMAO_FAST_THREADS_HPP
#define OOMAO_FAST_THREADS_HPP
#include "mex.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {
// One synchronous pool per MEX module. Workers touch only validated C++ data.
// MATLAB owns every output; no API calls or persistent mxArray pointers in workers.
class OomaoPool {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready, finished;
    std::function<void(size_t,size_t)> job;
    size_t generation=0, remaining=0, count=0, total=1;
    bool stopping=false;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping=true; }
        ready.notify_all();
        for(auto& worker:workers) worker.join();
        workers.clear(); stopping=false;
    }
public:
    ~OomaoPool() { stop(); }
    void run(size_t n,size_t threads,const std::function<void(size_t,size_t)>& function) {
        threads=std::min(threads,n);
        if(threads<=1) { function(0,n); return; }
        if(workers.size()!=threads-1) {
            stop();
            try {
                const size_t initialGeneration=generation;
                for(size_t id=1;id<threads;++id) workers.emplace_back([this,id,initialGeneration] {
                    size_t seen=initialGeneration;
                    for(;;) {
                        std::unique_lock<std::mutex> lock(mutex);
                        ready.wait(lock,[&]{return stopping || generation!=seen;});
                        if(stopping) return;
                        seen=generation;
                        const size_t begin=id*count/total,end=(id+1)*count/total;
                        lock.unlock(); job(begin,end); lock.lock();
                        if(--remaining==0) finished.notify_one();
                    }
                });
            } catch(...) { stop(); throw; }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            count=n; total=threads; job=function; remaining=threads-1; ++generation;
        }
        ready.notify_all();
        function(0,n/threads);
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock,[&]{return remaining==0;});
        job={};
    }
};
inline OomaoPool*& oomaoPoolPointer() { static OomaoPool* value=nullptr; return value; }
inline void oomaoPoolCleanup() { delete oomaoPoolPointer(); oomaoPoolPointer()=nullptr; }
inline OomaoPool& oomaoPool() {
    if(!oomaoPoolPointer()) { oomaoPoolPointer()=new OomaoPool; mexAtExit(oomaoPoolCleanup); }
    return *oomaoPoolPointer();
}
inline size_t oomaoThreads(const mxArray* value) {
    if(!value) return 1;
    if(!mxIsDouble(value)||mxIsComplex(value)||mxIsSparse(value)||mxGetNumberOfElements(value)!=1)
        mexErrMsgIdAndTxt("oomao:threads:type","threads must be a real double scalar.");
    const double n=mxGetScalar(value);
    if(!std::isfinite(n)||n<1||n>256||n!=std::floor(n))
        mexErrMsgIdAndTxt("oomao:threads:value","threads must be an integer from 1 to 256.");
    return static_cast<size_t>(n);
}
} // MEX-local symbols and pool state
#endif
