function buildMpsKernels(outputDirectory)
%BUILDMPSKERNELS Build the compiled single-precision Metal/MPS optical backend.
% No Python bridge or MATLAB GPU toolbox is used. Device creation is deferred
% until a run requests backend=mps. Output defaults to oomao/fast.
    if ~ismac,error('oomao:UnsupportedMpsBuild','Metal/MPS requires macOS.');end
    sourceDirectory=fullfile(fileparts(mfilename('fullpath')),'fast');
    if nargin==0,outputDirectory=sourceDirectory;end
    if ~isfolder(outputDirectory),mkdir(outputDirectory);end
    mex('-R2018a','LINKEXPORTCPP=', ...
        'CXXFLAGS=$CXXFLAGS -std=c++17 -O3 -fobjc-arc -ffp-contract=off -fno-fast-math -pthread', ...
        'LDFLAGS=$LDFLAGS -pthread -framework Foundation -framework Metal -framework MetalPerformanceShaders', ...
        '-outdir',outputDirectory,fullfile(sourceDirectory,'oomao_fast_mps_mex.mm'));
end
