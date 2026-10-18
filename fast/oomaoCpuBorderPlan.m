classdef oomaoCpuBorderPlan < handle
% Own immutable CPU border coefficients and their verified triangular layout.
    properties (Access=private)
        A
        B
        Mode
    end
    methods
        function obj=oomaoCpuBorderPlan(A,B)
            obj.A=A;obj.B=B;
            obj.Mode=double(oomao_fast_border_blas_mex(B));
        end
        function out=apply(obj,Z,noise)
            out=oomao_fast_border_blas_mex(obj.A,Z,obj.B,noise,obj.Mode);
        end
    end
end
