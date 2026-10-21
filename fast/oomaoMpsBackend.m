classdef oomaoMpsBackend < handle
% Own a compiled Metal/MPS context for cropped single science diffraction.
% All returned arrays are ordinary MATLAB arrays; calls wait for completion.
    properties (Access=private)
        Handle=[]
    end
    methods
        function obj=oomaoMpsBackend(threads)
            if ~ismac,error('oomao:MpsUnavailable','Metal/MPS requires macOS.');end
            if exist('oomao_fast_mps_mex','file')~=3
                error('oomao:MissingMpsKernel','Run buildMpsKernels before selecting MPS.');
            end
            obj.Handle=oomao_fast_mps_mex('create',double(threads));
        end
        function delete(obj)
            if ~isempty(obj.Handle)
                handle=obj.Handle;obj.Handle=[];oomao_fast_mps_mex('destroy',handle);
            end
        end
        function out=diffractPupil(obj,amplitudes,phases,masks,phasor,nout,crop,start)
            out=oomao_fast_mps_mex('diffractPupil',obj.Handle,amplitudes,phases,masks,phasor,double(nout),double(start),double(crop),32);
        end
        function out=diffract(obj,field,phasor,nout,start,crop)
            out=oomao_fast_mps_mex('diffract',obj.Handle,field,phasor,double(nout),double(start),double(crop));
        end
    end
end
