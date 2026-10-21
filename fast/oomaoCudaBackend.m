classdef oomaoCudaBackend < handle
%OOMAO CUDA execution through owned compiled MEX state.
% No GPU toolbox or Python interpreter is loaded. Device buffers/plans belong
% to this instance; all returned arrays retain the CPU method's MATLAB layout.
    properties (Access=private)
        Handle=[]
        Reconstruction=[]
        ReconstructionMatrices={}
        ReconstructionProjector=[]
    end
    methods
        function obj=oomaoCudaBackend(threads)
            if nargin<1,threads=maxNumCompThreads;end
            if exist('oomao_fast_cuda_mex','file')~=3
                error('oomao:MissingCudaKernel','Run buildCudaKernels(cudaRoot) before selecting CUDA.');
            end
            obj.Handle=oomao_fast_cuda_mex('create',32,double(threads));
        end
        function delete(obj)
            if ~isempty(obj.Handle)
                handle=obj.Handle;obj.Handle=[];
                oomao_fast_cuda_mex('destroy',handle);
            end
        end
        function out=reconstruct(obj,matrices,projector,slopes)
% Snapshots retain MATLAB copy-on-write ownership; storage changes trigger values checks.
            same=~isempty(obj.Reconstruction) ...
                && (oomao_fast_cpu_ops_mex(7,matrices,obj.ReconstructionMatrices) ...
                    || isequaln(matrices,obj.ReconstructionMatrices)) ...
                && (oomao_fast_cpu_ops_mex(7,projector,obj.ReconstructionProjector) ...
                    || isequaln(projector,obj.ReconstructionProjector));
            if ~same
                next=oomao_fast_cuda_mex('prepareReconstruction',obj.Handle,matrices,projector);
                previous=obj.Reconstruction;obj.Reconstruction=next;
                obj.ReconstructionMatrices=matrices;obj.ReconstructionProjector=projector;
                if ~isempty(previous),oomao_fast_cuda_mex('releaseReconstruction',obj.Handle,previous);end
            end
            out=oomao_fast_cuda_mex('applyReconstruction',obj.Handle,obj.Reconstruction,slopes);
        end
        function plan=prepareBorder(obj,A,B)
            token=oomao_fast_cuda_mex('prepareBorder',obj.Handle,A,B);
            plan=oomaoCudaBorderPlan(obj,token);
        end
        function releaseBorder(obj,token)
            oomao_fast_cuda_mex('releaseBorder',obj.Handle,token);
        end
        function out=applyBorder(obj,token,Z,noise)
            out=oomao_fast_cuda_mex('applyBorder',obj.Handle,token,Z,noise);
        end
        function out=formPupil(obj,amplitudes,phases,masks)
            out=oomao_fast_cuda_mex('formPupil',obj.Handle,amplitudes,phases,masks,64);
        end
        function out=diffractPupil(obj,amplitudes,phases,masks,phasor,nout,crop,start,precision)
            out=oomao_fast_cuda_mex('diffractPupil',obj.Handle,amplitudes,phases,masks,phasor,double(nout),double(start),double(crop),double(precision));
        end
        function plan=prepareSurface(obj,transpose)
            token=oomao_fast_cuda_mex('prepareSurface',obj.Handle,transpose);
            plan=oomaoCudaSurfacePlan(obj,token);
        end
        function releaseSurface(obj,token)
            oomao_fast_cuda_mex('releaseSurface',obj.Handle,token);
        end
        function out=applySurface(obj,token,coefficients)
            mode=8;if size(coefficients,2)>=32,mode=0;end
            out=oomao_fast_cuda_mex('applySurface',obj.Handle,token,coefficients,mode);
        end
        function out=diffract(obj,field,phasor,nout,start,crop)
            operation='diffract';
            if isa(field,'single')&&size(field,1)>=64&&crop<=size(field,1)/2,operation='diffractMft';end
            out=oomao_fast_cuda_mex(operation,obj.Handle,field,phasor,double(nout),double(start),double(crop));
        end
    end
end
