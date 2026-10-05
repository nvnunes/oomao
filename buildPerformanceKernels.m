function buildPerformanceKernels(outputDirectory)
%BUILDPERFORMANCEKERNELS Compile the optional kernels before running simulations.
% Uses the configured MATLAB C++ compiler on macOS or Linux. With no argument,
% writes platform-specific MEX files beside the sources; binaries are ignored
% by Git. A supplied output directory must be added to the MATLAB path.
    sourceDirectory = fileparts(mfilename('fullpath'));
    if nargin == 0
        outputDirectory = sourceDirectory;
    end
    if ~isunix
        error('oomao:UnsupportedCompiler', 'This build supports macOS and Linux C++ compilers.');
    end
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    flags = 'CXXFLAGS=$CXXFLAGS -O3 -ffp-contract=off -fno-fast-math';
    for kernel = {'oomao_atmosphere_mex', 'oomao_pupil_mex'}
        mex('-R2018a', 'LINKEXPORTCPP=', flags, '-outdir', outputDirectory, ...
            fullfile(sourceDirectory, [kernel{1} '.cpp']));
    end
end
