function buildFastKernels(outputDirectory)
%BUILDFASTKERNELS Build fast CPU kernels with the configured C++ compiler.
% Defaults to writing MEX files in oomao/fast; an explicit output folder
% must be added to the MATLAB path. Supports MATLAB on macOS and Linux.
    sourceDirectory=fullfile(fileparts(mfilename('fullpath')),'fast');
    if nargin==0,outputDirectory=sourceDirectory;end
    if ~isunix,error('oomao:UnsupportedCompiler','macOS or Linux is required.');end
    if ~isfolder(outputDirectory),mkdir(outputDirectory);end
    flags='CXXFLAGS=$CXXFLAGS -std=c++17 -O3 -ffp-contract=off -fno-fast-math -pthread';
    link='LDFLAGS=$LDFLAGS -pthread';
    digestLibraries={};if ~ismac,digestLibraries={'-ldl'};end
    mex('-R2018a','LINKEXPORTCPP=',flags,link,'-outdir',outputDirectory, ...
        fullfile(sourceDirectory,'oomao_fast_digest_mex.cpp'),digestLibraries{:});
    for kernel={'oomao_fast_atmosphere_mex','oomao_fast_pupil_mex','oomao_fast_cpu_ops_mex','oomao_fast_diffraction_mex','oomao_fast_covariance_mex','oomao_fast_geometric_mex','oomao_fast_mft_mex','oomao_fast_border_blas_mex'}
        mex('-R2018a','LINKEXPORTCPP=',flags,link,'-outdir',outputDirectory, ...
            fullfile(sourceDirectory,[kernel{1},'.cpp']),'-lmwblas');
    end
end
