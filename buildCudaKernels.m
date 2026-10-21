function buildCudaKernels(cudaRoot,runtimeDirectory)
%BUILDCUDAKERNELS Compile the CUDA driver/NVRTC/cuFFT MEX backend on Linux.
% cudaRoot is an existing CUDA environment (for example CuPy's Conda prefix)
% containing lib/ and targets/x86_64-linux/include/. No environment is modified.
% Optional runtimeDirectory adds a deployed library location to the MEX RPATH;
% MATLAB Compiler's generated launcher replaces LD_LIBRARY_PATH.
    if ~isunix||ismac,error('oomao:UnsupportedCudaBuild','The CUDA backend builds on Linux.');end
    source=fullfile(fileparts(mfilename('fullpath')),'fast');
    lib=fullfile(cudaRoot,'lib');include=fullfile(cudaRoot,'targets','x86_64-linux','include');
    runtimePath=lib;if nargin>1,runtimePath=[runtimePath,':',runtimeDirectory];end
    flags=['LDFLAGS=$LDFLAGS -pthread -Wl,--disable-new-dtags,-rpath,',runtimePath];
    mex('-R2018a','LINKEXPORTCPP=','CXXFLAGS=$CXXFLAGS -std=c++17 -O3 -ffp-contract=off -fno-fast-math -pthread',flags, ...
        ['-I',include],['-I',source],['-L',lib],'-L/usr/lib/x86_64-linux-gnu', ...
        '-l:libcuda.so.1','-l:libnvrtc.so.12','-l:libcufft.so.11','-l:libcublas.so.12', ...
        '-outdir',source,fullfile(source,'oomao_fast_cuda_mex.cpp'));
end
