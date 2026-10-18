classdef stochasticWave < handle
    % STOCHASTICWAVE Create a stochasticWave object

    properties
        % mask to retain amplitude and phase values used for statistics
        % moments evaluation
        mask;
        % Form concatenated pupils with the double-precision native kernel.
        useNativePupil = false;
        % Run-owned numerical backend; geometry and source state stay native.
        execution = [];
        % true if the phase is a stochastic property
        stochasticPhase = true;
        % true if the amplitude is a stochastic property
        stochasticAmplitude = false;
        nSample;
        samples;
        saveImage;
        % image 
        image
        % integratedImage
        cumImage=0;
        % phase variance buffer
        phaseVar;
        % logging flag
        logging = false;
    end
    
    properties (Dependent)
        % wave amplitude
        amplitude;
        % wave phase        
        phase;
        % unmasked wave phase
        phaseUnmasked;
        % reset the wave phase 
        %resetPhase
        % intensity
        intensity;
        % buffer sequence
        bufSeq
    end
    
    properties (Dependent,SetAccess=private)
        % the wave
        wave;
        % mean value removed phase map
        meanRmPhase; 
    end
    
    properties (Access=private)
        fastAmplitudeExecution=[];
        fastAmplitudeSnapshot=[];
        fastMaskSnapshot=[];
        fastMaskedAmplitude=[];
        fastAmplitudeValid=false;
        p_bufSeq;
        bufferLength = 10000;
        kBufSeq;
    end
    properties (Access=protected)
        p_amplitude = 1;
        p_phase = 0;
    end
    
    methods
        
        %% Constructor
        function obj = stochasticWave
        end
        
        %% Get/Set bufSeq
        function out = get.bufSeq(obj)
            out = obj.p_bufSeq;
        end
        function set.bufSeq(obj,val)
            obj.p_bufSeq = val;
            obj.kBufSeq = 1;
            obj.samples  = zeros(sum(obj.p_bufSeq),obj.bufferLength);
            obj.nSample  = 0;
        end
        
        %% Get/Set amplitude
        function out = get.amplitude(obj)
            eligible=~isempty(obj.execution)&&obj.execution.Fast&& ...
                isa(obj.p_amplitude,'double')&&isreal(obj.p_amplitude)&&~issparse(obj.p_amplitude)&& ...
                islogical(obj.mask)&&~issparse(obj.mask)&&ndims(obj.mask)==2;
            if eligible
                same=obj.fastAmplitudeValid&&isequal(obj.fastAmplitudeExecution,obj.execution)&& ...
                    obj.execution.sameArrayStorage(obj.p_amplitude,obj.fastAmplitudeSnapshot)&& ...
                    obj.execution.sameArrayStorage(obj.mask,obj.fastMaskSnapshot);
                if ~same
                    out=bsxfun(@times,obj.p_amplitude,obj.mask);
                    if isequal(out,obj.p_amplitude),out=obj.p_amplitude;end
                    obj.fastAmplitudeExecution=obj.execution;obj.fastAmplitudeSnapshot=obj.p_amplitude;obj.fastMaskSnapshot=obj.mask;
                    obj.fastMaskedAmplitude=out;obj.fastAmplitudeValid=true;
                end
                out=obj.fastMaskedAmplitude;
            else
                obj.fastAmplitudeValid=false;
                obj.fastAmplitudeExecution=[];obj.fastAmplitudeSnapshot=[];obj.fastMaskSnapshot=[];obj.fastMaskedAmplitude=[];
                out=bsxfun(@times,obj.p_amplitude,obj.mask);
            end
        end
        function set.amplitude(obj,val)
            fast=~isempty(obj.execution)&&obj.execution.Fast&& ...
                isa(obj.p_amplitude,'double')&&isreal(obj.p_amplitude)&&~issparse(obj.p_amplitude)&& ...
                isa(val,'double')&&isreal(val)&&~issparse(val);
            if fast&&isscalar(val)&&val==1,return;end
            if fast&&isscalar(obj.p_amplitude)&&obj.p_amplitude==1
                obj.p_amplitude=val;
            else
                obj.p_amplitude=bsxfun(@times,obj.p_amplitude,val);
            end
        end
    
        %% Get/Set phaseUnmasked
        function out = get.phaseUnmasked(obj)
            out = obj.p_phase;
        end
        %% Get/Set phase
        function out = get.phase(obj)
            out = obj.p_phase;
            out = bsxfun(@times,out,obj.mask);
        end
        function set.phase(obj,val)
            fast=~isempty(obj.execution)&&obj.execution.Fast&& ...
                isa(obj.p_phase,'double')&&isreal(obj.p_phase)&&~issparse(obj.p_phase)&& ...
                isa(val,'double')&&isreal(val)&&~issparse(val);
            if fast&&isscalar(obj.p_phase)&&obj.p_phase==0
                obj.p_phase=val;
            elseif ~(fast&&isscalar(val)&&val==0)
                obj.p_phase=bsxfun(@plus,obj.p_phase,val);
            end
            if obj.logging
                buf = stats(obj,@var);
                if isempty(obj.phaseVar)
                    obj.phaseVar = zeros(obj.bufferLength,length(buf));
                    obj.kBufSeq = 0;
                end
                if length(buf)>size(obj.phaseVar,2)
                    obj.phaseVar = repmat( obj.phaseVar , 1 , length(buf));
                end
                obj.kBufSeq = obj.kBufSeq + 1;
                obj.phaseVar(obj.kBufSeq,:) = buf;
            end
            %             if ~isempty(obj.p_bufSeq)
%                 if obj.p_bufSeq(obj.kBufSeq)
%                     obj.nSample = obj.nSample + 1;
%                     obj.samples(obj.nSample) = std(obj);
%                 end
%                 obj.kBufSeq = obj.kBufSeq + 1;
%                 if obj.kBufSeq>length(obj.p_bufSeq)
%                     obj.kBufSeq = 1;
%                 end
%             end
%             if obj.saveImage
%                 a = obj.wave;
%                 [n,m] = size(a);
%                 obj.image = abs( fft2( a , 2*n , 2*m ) ).^2;
%                 obj.cumImage = obj.cumImage + obj.image;
%             end
        end
        
        %% reset phase to predefined value val
        function resetPhase(obj,val)
            obj.p_phase = val;
        end
        
        %% Get the wave
        function out = get.wave(obj)
            out = bsxfun( @times , obj.amplitude , exp(1i.*obj.phase) );
        end
        
        %% Get intensity
        function out = get.intensity(obj)
            out = obj.wave.*conj(obj.wave);
        end
        
        %% Get the meanRmPhase propertye
        function meanRmPhase = get.meanRmPhase(obj)
            buffer = obj.stochasticAmplitude;
            obj.stochasticAmplitude = false; % Set to false in order for the mean method to return only phase value
            meanPhase   = mean(obj); % Set the mask if empty
            buf = utilities.toggleFrame(obj.phase,2);
            meanRmPhase = zeros( size( buf ) );
            meanRmPhase(obj.mask,:) ...
                        = bsxfun( @minus, buf(obj.mask) , meanPhase );
            meanRmPhase = tools.toggleFrame(meanRmPhase,3);
            obj.stochasticAmplitude = buffer;
        end
        
        function varargout = reset(obj)
            %% RESET Reset wave properties
            %
            % reset(obj) resets the mask to [], the amplitude to 1 and the
            % phase to 0
            %
            % obj = reset(obj) resets and returns the resetted object
            
            for kObj = 1:numel(obj);
                obj(kObj).mask        = [];
                obj(kObj).p_amplitude = 1;
                obj(kObj).p_phase     = 0;
            end
            if nargout>0
                varargout{1} = obj;
            end
        end
        
        function out = catAmplitude(obj)
            %% CATAMPLITUDE Concatenate amplitudes
            %
            % out = catAmplitude(obj) concatenates the amplitudes of an
            % array of stochasticWave objects
            
            out = cat( ndims(obj) , obj.amplitude);
        end
        function out = catPhase(obj)
            %% CATPHASE Concatenate phases
            %
            % out = catPhase(obj) concatenates the phases of an array of
            % stochasticWave objects
            
            out = cat( ndims(obj) , obj.phase);
        end
        function out = phaseVector(obj)
            %% PHASEVECTOR Concatenate phases in 1 single vector
            %
            % out = phaseVector(obj) concatenates phase(s) values within
            % the mask in 1 single vector 
            
            nObj = length(obj);
            out = cell(nObj,1);
            for kSrc=1:nObj
%                 m_mask = repmat(obj(kSrc).mask,[1,1,size(obj(kSrc).phase,3)]);
                buf = utilities.toggleFrame(obj(kSrc).phase,2);
                out{kSrc} = buf(obj(kSrc).mask,:);
            end
            out = cell2mat(out);
        end
        function out = catMeanRmPhase(obj)
            %% CATMEANRMPHASE Concatenate mean removed phases
            %
            % out = catMeanRmPhase(obj) concatenates the mean removed
            % phases of an array of stochasticWave objects
            
            out = cat( ndims(obj) , obj.meanRmPhase);
        end
        function supported=hasSinglePupil(obj,n)
% Eligibility only; default/uninitialized and multi-wave sources use catWave.
            supported=~isempty(obj)&&ndims(obj)==2&&isscalar(n)&&n>=2;
            if ~supported,return;end
            for k=1:numel(obj)
                a=obj(k).p_amplitude;p=obj(k).p_phase;m=obj(k).mask;
                if ~isa(a,'double')||~isreal(a)||issparse(a)||~isa(p,'double')||~isreal(p)||issparse(p)|| ...
                  ~islogical(m)||issparse(m)||~isequal(size(m),[n,n])|| ...
                  ~(isscalar(a)||isequal(size(a),[n,n]))||~(isscalar(p)||isequal(size(p),[n,n]))
                    supported=false;return
                end
            end
        end
        function out=diffractPupil(obj,execution,phasor,nout,crop,start,precision)
            out=execution.diffractPupil({obj.p_amplitude},{obj.p_phase},{obj.mask},phasor,nout,crop,start,precision);
        end
        function out = catWave(obj)
            %% CATWAVE Concatenate waves
            %
            % out = catWave(obj) concatenates the waves of an array of
            % stochasticWave objects
            if ~isempty(obj) && ndims(obj)==2 && ~isempty(obj(1).execution) && obj(1).execution.Fast
                out = obj(1).execution.formPupil({obj.p_amplitude},{obj.p_phase},{obj.mask});
            elseif ~isempty(obj) && ndims(obj)==2 && all([obj.useNativePupil])
                out = oomao_pupil_mex({obj.p_amplitude},{obj.p_phase},{obj.mask});
            else
                out = cat( ndims(obj) , obj.wave);
            end
        end
        
        function out = mean(obj)
            %% MEAN Average or mean value
            %
            % out = mean(obj) computes the average or mean value of the
            % amplitude and/or the phase within the mask. If either
            % amplitude or phase is a stochastic property out is a single
            % value; if both amplitude and phase are stochastic out is a 2
            % element vector with amplitude and phase results in this order

            out = stats(obj,@mean);
        end
        
        function out = var(obj)
            %% VAR Variance
            %
            % out = var(obj) computes the variance of the amplitude and/or
            % the phase within the mask. If either amplitude or phase is a
            % stochastic property out is a single value; if both amplitude
            % and phase are stochastic out is a 2 element vector with
            % amplitude and phase results in this order

            out = stats(obj,@var);
       end
        
        function out = std(obj)
            %% STD Standard deviation
            %
            % out = std(obj) computes the standard deviation of the
            % amplitude and/or the phase within the mask. If either
            % amplitude or phase is a stochastic property out is a single
            % value; if both amplitude and phase are stochastic out is a 2
            % element vector with amplitude and phase results in this order
            
            out = stats(obj,@std);
        end
        
        function out = reSample(obj,newResolution)
            nSrc = obj(1).nSrc;
            n    = length(obj(1).p_phase);
            out  = zeros(newResolution,newResolution,nSrc);
            [xi,yi] = meshgrid( 1 + (0:newResolution-1)*(n-1)/(newResolution-1) );
            for kSrc=1:nSrc
                out(:,:,kSrc) = interp2(obj(kSrc).p_phase,xi,yi,'spline');
            end
        end
        
    end
        
    methods (Access=private)
        
        function out = stats(obj,fun)
            if numel(obj)>1
                out = cellfun( @(x) stats(x,fun), num2cell(obj) );
            else
                if isempty(obj.mask)
                    disp(' @(stochasticWave)> Setting the mask!')
                    obj.mask = ones(size(obj.amplitude));
                end
%                 k = 1;
%                 if obj.stochasticAmplitude
%                     out(k) = fun(obj.amplitude(obj.mask));
%                     k = k+1;
%                 end
                if obj.stochasticPhase
                    buf = utilities.toggleFrame(obj.phase,2);
                    out = fun(buf(obj.mask,:));
                end
            end
        end
        
    end
    
end