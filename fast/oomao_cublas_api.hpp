#pragma once
// Minimal stable cuBLAS v2 C ABI; CUDA Driver owns all memory and streams.
// Signatures: NVIDIA cuBLAS documentation, Level-2 GEMV and management API.
struct cublasContext;
using cublasHandle_t=cublasContext*;
extern "C" {
int cublasCreate_v2(cublasHandle_t*);
int cublasDestroy_v2(cublasHandle_t);
int cublasSetStream_v2(cublasHandle_t,CUstream);
int cublasSetMathMode(cublasHandle_t,int);
int cublasCgemm_v2(cublasHandle_t,int,int,int,int,int,const void*,const void*,int,const void*,int,const void*,void*,int);
int cublasZgemm_v2(cublasHandle_t,int,int,int,int,int,const void*,const void*,int,const void*,int,const void*,void*,int);
int cublasCgemmBatched(cublasHandle_t,int,int,int,int,int,const void*,const void*,int,const void*,int,const void*,void*,int,int);
int cublasZgemmBatched(cublasHandle_t,int,int,int,int,int,const void*,const void*,int,const void*,int,const void*,void*,int,int);
int cublasDgemm_v2(cublasHandle_t,int,int,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
int cublasDgemv_v2(cublasHandle_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
}
