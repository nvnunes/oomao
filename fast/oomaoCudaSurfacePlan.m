classdef oomaoCudaSurfacePlan < handle
% Own an immutable sparse DM matrix in the compiled CUDA context.
    properties (Access=private)
        Backend
        Token
    end
    methods
        function obj=oomaoCudaSurfacePlan(backend,token)
            obj.Backend=backend;obj.Token=token;
        end
        function current=isCurrent(obj,backend)
            current=isvalid(obj.Backend)&&obj.Backend==backend;
        end
        function out=apply(obj,coefficients)
            if ~isvalid(obj.Backend),error('oomao:ReleasedCudaSurface','CUDA backend has been closed.');end
            out=obj.Backend.applySurface(obj.Token,coefficients);
        end
        function delete(obj)
            if isvalid(obj.Backend),obj.Backend.releaseSurface(obj.Token);end
        end
    end
end
