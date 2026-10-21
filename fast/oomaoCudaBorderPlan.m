classdef oomaoCudaBorderPlan < handle
% Own an immutable pair of border matrices in the compiled CUDA context.
    properties (Access=private)
        Backend
        Token
    end
    methods
        function obj=oomaoCudaBorderPlan(backend,token)
            obj.Backend=backend;obj.Token=token;
        end
        function out=apply(obj,Z,noise)
            if ~isvalid(obj.Backend),error('oomao:ReleasedCudaBorder','CUDA backend has been closed.');end
            out=obj.Backend.applyBorder(obj.Token,Z,noise);
        end
        function delete(obj)
            if isvalid(obj.Backend),obj.Backend.releaseBorder(obj.Token);end
        end
    end
end
