#pragma once
// NVRTC's stable C ABI subset. Full CUDA SDK development headers are not needed.
// Signatures: NVIDIA NVRTC 12.x User Guide, compilation API.
#include <cstddef>
extern "C" {
struct _nvrtcProgram;
typedef _nvrtcProgram* nvrtcProgram;
int nvrtcCreateProgram(nvrtcProgram*,const char*,const char*,int,const char* const*,const char* const*);
int nvrtcCompileProgram(nvrtcProgram,int,const char* const*);
int nvrtcGetProgramLogSize(nvrtcProgram,size_t*);
int nvrtcGetProgramLog(nvrtcProgram,char*);
int nvrtcGetCUBINSize(nvrtcProgram,size_t*);
int nvrtcGetCUBIN(nvrtcProgram,char*);
int nvrtcDestroyProgram(nvrtcProgram*);
int nvrtcVersion(int*,int*);
const char* nvrtcGetErrorString(int);
}
