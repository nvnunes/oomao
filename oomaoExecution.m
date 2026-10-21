classdef oomaoExecution < handle
%OOMAOEXECUTION Per-run compiled execution and science-diffraction precision.
% fast selects C++ kernels, sparse DM layout and reconstruction
% regrouping. precision (32/64) sets science diffraction only; calibration,
% atmosphere, control, detector noise and PSF accumulation stay double.
% Omitted precision selects single science diffraction for fast execution.
    properties (SetAccess=private)
        Fast
        Backend
        Precision
        Threads
    end
    properties (Access=private)
        AtmosphereGeometry = []
        Cuda = []
        Mps = []
        AtmosphereDirectory = ''
        AtmospherePrefix = ''
        AtmosphereEvents = {}
        FittingDirectory = ''
        FittingInputs = struct()
        FittingEvents = {}
        GroundEntries = {}
        GroundBytes = 0
        GroundHits = 0
        GroundMisses = 0
        GroundLimit = 128*1024^2
    end
    methods
        function clearCovarianceCache(obj,resetCounters)
% Cache lifetime is one HO build, never the subsequent simulation loop.
            obj.GroundEntries={};obj.GroundBytes=0;
            if nargin>1 && resetCounters,obj.GroundHits=0;obj.GroundMisses=0;end
        end
        function stats = covarianceCacheStats(obj)
            stats=struct('hits',obj.GroundHits,'misses',obj.GroundMisses, ...
                'bytes',obj.GroundBytes,'limitBytes',obj.GroundLimit);
        end
        function obj = oomaoExecution(fast, backend, precision, threads)
            if nargin < 1, fast = true; end
            if nargin < 2, backend = 'cpu'; end
            if nargin < 3, precision = []; end
            if nargin < 4, threads = -1; end
            if (ischar(fast) && isrow(fast)) || (isstring(fast) && isscalar(fast))
                value = lower(string(fast));
                if ~any(value == ["true","false","1","0","on","off"])
                    error('oomao:InvalidFast','fast must be true/false or on/off.');
                end
                fast = any(value == ["true","1","on"]);
            end
            if ~((islogical(fast) || isnumeric(fast)) && isscalar(fast) ...
                    && isreal(fast) && isfinite(fast) && any(fast == [0,1]))
                error('oomao:InvalidFast','fast must be a scalar boolean.');
            end
            if ~((ischar(backend) && isrow(backend)) || (isstring(backend) && isscalar(backend)))
                error('oomao:InvalidBackend','backend must be cpu, cuda or mps.');
            end
            backend = lower(string(backend));
            if ~any(backend == ["cpu","cuda","mps"])
                error('oomao:BackendUnavailable','backend must be cpu, cuda or mps.');
            end
            if backend ~= "cpu" && ~fast
                error('oomao:InvalidBackend','A non-CPU backend requires fast=true.');
            end
            if isempty(precision)
                precision = 64 - 32*double(logical(fast));
            elseif (ischar(precision) && isrow(precision)) || (isstring(precision) && isscalar(precision))
                precision = str2double(precision);
            end
            if ~(isnumeric(precision) && isreal(precision) && isscalar(precision) ...
                    && isfinite(precision) && any(precision == [32,64]))
                error('oomao:InvalidPrecision','precision must be 32 or 64.');
            end
            if backend=="mps" && precision~=32
                error('oomao:MpsPrecision','MPS science diffraction requires precision=32.');
            end
            if (ischar(threads) && isrow(threads)) || (isstring(threads) && isscalar(threads))
                threads = str2double(threads);
            end
            if ~(isnumeric(threads) && isreal(threads) && isscalar(threads) ...
                    && isfinite(threads) && (threads == -1 || (threads >= 1 && threads <= 256 && threads == fix(threads))))
                error('oomao:InvalidThreads','threads must be an integer from 1 to 256, or -1 to inherit MATLAB.');
            end
            if threads == -1, threads = maxNumCompThreads; end
            obj.Threads = double(threads);
            obj.Fast = logical(fast);
            obj.Backend = char(backend);
            obj.Precision = double(precision);
        end
        function close(obj)
            if ~isempty(obj.AtmosphereGeometry)
                token=obj.AtmosphereGeometry;obj.AtmosphereGeometry=[];
                oomao_fast_atmosphere_mex('destroy',token);
            end
            if ~isempty(obj.Mps)
                backend=obj.Mps;obj.Mps=[];delete(backend);
            end
            if ~isempty(obj.Cuda)
                backend=obj.Cuda;obj.Cuda=[];delete(backend);
            end
            obj.clearCovarianceCache(false);
        end
        function delete(obj)
            obj.close();
        end
        function requireKernels(obj)
            if obj.Fast
                for name = {'oomao_fast_atmosphere_mex','oomao_fast_pupil_mex','oomao_fast_cpu_ops_mex','oomao_fast_diffraction_mex','oomao_fast_covariance_mex','oomao_fast_geometric_mex','oomao_fast_mft_mex','oomao_fast_border_blas_mex'}
                    if exist(name{1},'file') ~= 3
                        error('oomao:MissingPerformanceKernel', ...
                            'Missing %s. Run buildFastKernels before fast=true.',name{1});
                    end
                end
                if strcmp(obj.Backend,'cuda') && isempty(obj.Cuda)
                    obj.Cuda=oomaoCudaBackend(obj.Threads);
                end
                if strcmp(obj.Backend,'mps') && isempty(obj.Mps)
                    obj.Mps=oomaoMpsBackend(obj.Threads);
                end
            end
        end
        function out = projectAtmosphere(obj, varargin)
            if ~obj.Fast
                error('oomao:InvalidDispatch','Baseline atmosphere uses the MATLAB relay.');
            end
            if strcmp(obj.Backend,'cuda')
                if isempty(obj.AtmosphereGeometry)
                    obj.AtmosphereGeometry=oomao_fast_atmosphere_mex('create');
                end
                out=oomao_fast_atmosphere_mex(varargin{:},obj.Threads,obj.AtmosphereGeometry);
            else
                out=oomao_fast_atmosphere_mex(varargin{:},obj.Threads);
            end
        end
        function out = formPupil(obj, amplitudes, phases, masks)
            if ~obj.Fast
                error('oomao:InvalidDispatch','Baseline pupil uses the MATLAB wave getter.');
            end
            out = oomao_fast_pupil_mex(amplitudes, phases, masks,obj.Threads);
        end
        function out=diffractPupil(obj,amplitudes,phases,masks,phasor,nout,crop,start,precision)
            if strcmp(obj.Backend,'cuda')
                out=obj.Cuda.diffractPupil(amplitudes,phases,masks,phasor,nout,crop,start,precision);
                return
            end
            if strcmp(obj.Backend,'mps')&&precision==32&&size(masks{1},1)>=64&&crop<=size(masks{1},1)/2
                out=obj.Mps.diffractPupil(amplitudes,phases,masks,phasor,nout,crop,start);
                return
            end
            field=obj.prepareDiffraction(obj.formPupil(amplitudes,phases,masks),precision);
            out=obj.diffract(field,phasor,nout,crop,start);
        end
        function out = diffract(obj,field,phasor,nout,crop,start)
            if strcmp(obj.Backend,'cuda')
                out=obj.Cuda.diffract(field,phasor,nout,start,crop);
                return
            end
            if strcmp(obj.Backend,'mps') && isa(field,'single') && size(field,1)>=64 && crop<=size(field,1)/2
                out=obj.Mps.diffract(field,phasor,nout,start,crop);
                return
            end
            if size(field,1)>=64 && crop<=size(field,1)/2
                out=oomao_fast_mft_mex(field,phasor,double(nout),double(start),double(crop),obj.Threads);
            else
                out=oomao_fast_diffraction_mex(field,phasor,double(nout),double(start),double(crop),obj.Threads);
            end
        end
        function out = geometricSlopes(obj,transpose,phase,pixels,offset,axis)
            out=oomao_fast_geometric_mex(transpose,phase,pixels,offset,axis,obj.Threads);
        end
        function same=sameArrayStorage(~,left,right)
% MATLAB snapshots own shared data; mutations detach through copy-on-write.
% False only triggers the existing value comparison, never invalidates data.
            same=oomao_fast_cpu_ops_mex(7,left,right);
        end
        function plan = prepareSparseSurface(obj,transpose)
            plan=[];
            if strcmp(obj.Backend,'cuda'),plan=obj.Cuda.prepareSurface(transpose);end
        end
        function current=isCurrentSparseSurface(obj,plan)
            current=~isempty(obj.Cuda)&&~isempty(plan)&&isvalid(plan)&&plan.isCurrent(obj.Cuda);
        end
        function out = sparseSurface(obj,transpose,coefs,plan)
            if nargin>3 && ~isempty(plan),out=plan.apply(coefs);return;end
            out = oomao_fast_cpu_ops_mex(6,transpose,coefs,obj.Threads);
        end
        function out = shiftScreen(obj,image,residual)
            out = oomao_fast_cpu_ops_mex(5,image,residual,obj.Threads);
        end
        function out = reconstruct(obj,matrices,projector,slopes)
% Apply each reconstructor followed by its command projector, in native order.
            if strcmp(obj.Backend,'cuda')
                out=obj.Cuda.reconstruct(matrices,projector,slopes);
            else
                out=oomao_fast_cpu_ops_mex(2,matrices,projector,slopes);
            end
        end
        function out = accumulate(obj,buffer,image)
% Sum into a double detector buffer, converting science float32 imagelets.
            out=oomao_fast_cpu_ops_mex(3,buffer,image,obj.Threads);
        end
        function out = multiply(~,left,right)
% CPU backend uses MATLAB BLAS; preserve the caller's multiplication order.
            out=left*right;
        end
        function plan=prepareScreenBorder(~,A,B)
            plan=oomaoCpuBorderPlan(A,B);
        end
        function out = extendScreen(obj,A,Z,B,noise,plan)
% Random draws are supplied by the caller using the native MATLAB stream.
            if nargin>5&&~isempty(plan)
                out=plan.apply(Z,noise);
            else
                out=obj.multiply(A,Z)+obj.multiply(B,noise);
            end
        end
        function out = fourier(~,field,n,axis)
% WFS/science Fourier primitive; the input retains its existing precision.
            out=fft(field,n,axis);
        end
        function out = normalDraw(~,varargin)
% Preserve native MATLAB RNG/stream consumption, including detector noise.
            out=randn(varargin{:});
        end
        function out = poissonDraw(~,rate)
            out=poissrnd(rate);
        end
        function configureAtmosphereCache(obj,directory,release,prefix)
            if ~obj.Fast,return;end
            if ~(ischar(directory)&&isrow(directory)&&~isempty(directory))
                error('oomao:AtmosphereCacheDirectory','A nonempty cache directory is required.');
            end
            if isempty(regexp(prefix,'^[a-zA-Z0-9_-]+$','once'))
                error('oomao:AtmosphereCacheName','A readable atmosphere configuration name is required.');
            end
            obj.AtmospherePrefix=prefix;
            obj.AtmosphereDirectory=oomaoFastCacheDirectory(directory,release);
        end
        function enabled=hasAtmosphereCache(obj)
            enabled=obj.Fast&&~isempty(obj.AtmosphereDirectory);
        end
        function [A,B]=atmosphereMatrices(obj,inner,outer,atm,layer,geometry)
            if ~obj.hasAtmosphereCache(),error('oomao:AtmosphereCacheNotConfigured','Configure atmosphere cache storage first.');end
            name=sprintf('%s_layer_%02d',obj.AtmospherePrefix,layer);
            [A,B,event]=oomaoFastAtmosphereMatrices(inner,outer,atm,obj.AtmosphereDirectory,obj.Threads,name,geometry);
            obj.AtmosphereEvents{end+1}=event;
        end
        function events=atmosphereCacheStats(obj)
            events=obj.AtmosphereEvents;
        end
        function configureFittingCache(obj,directory,release,inputs)
% Optional persistent storage is separate from every existing HO cache.
            if ~obj.Fast,return;end
            if ~(ischar(directory)&&isrow(directory)&&~isempty(directory))
                error('oomao:FittingCacheDirectory','A nonempty cache directory is required.');
            end
            obj.FittingDirectory=oomaoFastCacheDirectory(directory,release);
            obj.FittingInputs=inputs;
        end
        function out = fittingInverse(obj,name,matrix,varargin)
% This is still native pinv, with its exact default or explicit tolerance.
            if ~obj.Fast || isempty(obj.FittingDirectory)
                out=pinv(matrix,varargin{:});return
            end
            if isempty(regexp(name,'^[a-zA-Z0-9_-]+$','once'))
                error('oomao:FittingCacheName','A readable fitting-matrix role is required.');
            end
            if ~isfield(obj.FittingInputs,name)
                error('oomao:FittingCacheMetadata','Missing scientific inputs for %s.',name);
            end
            [out,event]=oomaoFastFittingInverse(matrix,varargin,obj.FittingDirectory,obj.Threads,name,obj.FittingInputs.(name));
            obj.FittingEvents{end+1}=event;
        end
        function events = fittingCacheStats(obj)
            events=obj.FittingEvents;
        end
        function out = pseudoInverse(~,matrix,varargin)
% Retain MATLAB SVD, default rank tolerance and explicit fitting tolerances.
            out=pinv(matrix,varargin{:});
        end
        function out = truncatedInverse(~,matrix,nFiltered)
            out=tsvd(matrix,nFiltered);
        end
        function out = irregularCovariance(obj,sampling,diameter,wfs,atm,lgs,varargin)
% Native spatio-angular geometry; only covariance evaluation is dispatched.
            out=phaseStats.spatioAngularIrregularCovarianceMatrix( ...
                sampling,diameter,wfs,atm,lgs,varargin{:},'execution',obj);
        end
        function out = covarianceMatrix(obj,rho1,rho2,atm)
% Fused double-precision CPU covariance. Source geometry and layer weighting
% retain the native phaseStats definition; fast=false calls MATLAB directly.
            if ~obj.Fast
                out=phaseStats.covarianceMatrix(rho1,rho2,atm);
                return
            end
            [L0,cst,cst0,fraction]=obj.covarianceParameters(atm);
            layers=atm.layer;
            ground=isscalar(layers) && layers.altitude==0;
            if ground
                key={rho1,rho2,L0,cst,cst0,fraction};
                for k=1:numel(obj.GroundEntries)
                    entry=obj.GroundEntries{k};
                    if isequal(entry.key,key)
                        out=entry.value;obj.GroundHits=obj.GroundHits+1;
                        obj.GroundEntries(k)=[];obj.GroundEntries{end+1}=entry;return
                    end
                end
                obj.GroundMisses=obj.GroundMisses+1;
            end
            out=oomao_fast_covariance_mex(rho1,rho2,L0,cst,cst0,fraction,obj.Threads);
            if ground
                info=whos('key','out');bytes=sum([info.bytes]);
                if bytes<=obj.GroundLimit
                    while obj.GroundBytes+bytes>obj.GroundLimit
                        old=obj.GroundEntries{1};obj.GroundBytes=obj.GroundBytes-old.bytes;
                        obj.GroundEntries(1)=[];
                    end
                    obj.GroundEntries{end+1}=struct('key',{key},'value',out,'bytes',bytes);
                    obj.GroundBytes=obj.GroundBytes+bytes;
                end
            end
        end
        function out = layerCovariance(obj,left,right,atm)
% Sum layers in native order. Ground entries reuse the exact single-layer
% evaluator; consecutive elevated layers share one output allocation.
            out=[];first=1;
            while first<=atm.nLayer
                if atm.layer(first).altitude==0
                    term=obj.covarianceMatrix(left(:,first),right(:,first),slab(atm,first));
                    if isempty(out),out=0+term;else,out=out+term;end
                    first=first+1;
                else
                    last=first;
                    while last<atm.nLayer && atm.layer(last+1).altitude~=0,last=last+1;end
                    indices=first:last;count=numel(indices);
                    L0=zeros(1,count);cst=L0;cst0=L0;fraction=L0;
                    for k=1:count
                        [L0(k),cst(k),cst0(k),fraction(k)]= ...
                            obj.covarianceParameters(slab(atm,indices(k)));
                    end
                    args={left(:,indices),right(:,indices),L0,cst,cst0,fraction,obj.Threads,double(count)};
                    if isempty(out),out=oomao_fast_covariance_mex(args{:});
                    else,out=oomao_fast_covariance_mex(args{:},out);end
                    first=last+1;
                end
            end
        end
        function out = prepareDiffraction(obj, field, precision)
            if nargin<3,precision=obj.Precision;end
            if precision == 32
                out = single(field);
            else
                out = double(field);
            end
        end
    end
    methods (Static,Access=private)
        function [L0,cst,cst0,fraction]=covarianceParameters(atm)
% Keep the native phaseStats constants in one backend-owned implementation.
            ratio=(atm.L0./atm.r0).^(5./3);
            cst=double((24.*gamma(6./5)./5).^(5./6).* ...
                (gamma(11./6)./(2.^(5./6).*pi.^(8./3))).*ratio);
            cst0=double((24.*gamma(6./5)./5).^(5./6).* ...
                (gamma(11./6).*gamma(5./6)./(2.*pi.^(8./3))).*ratio);
            L0=double(atm.L0);layers=atm.layer;
            fraction=double(sum([layers.fractionnalR0]));
        end
    end
end
