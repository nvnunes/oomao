#pragma once
// Stable cuFFT C ABI subset (NVIDIA cuFFT 12.x API). CUDA driver owns buffers.
#include <cstddef>
#include <cuda.h>
extern "C" {
int cufftPlan1d(int*,int,int,int);
int cufftSetStream(int,CUstream);
int cufftExecC2C(int,void*,void*,int);
int cufftExecZ2Z(int,void*,void*,int);
int cufftGetSize(int,size_t*);
int cufftDestroy(int);
}
constexpr int OOMAO_CUFFT_C2C=0x29,OOMAO_CUFFT_Z2Z=0x69;
