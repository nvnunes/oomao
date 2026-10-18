function [A,B,event]=oomaoFastAtmosphereMatrices(inner,outer,atm,directory,threads,name)
% Cache deterministic native stationary-atmosphere border matrices, not screens.
% Geometry and arithmetic are unchanged; producer environments are provenance.
identity=struct('schema',1,'algorithm','native-atmosphere-border', ...
 'innerShape',size(inner),'outerShape',size(outer),'matlab',version, ...
 'release',version('-release'),'computer',computer,'blas',version('-blas'), ...
 'lapack',version('-lapack'),'threads',threads,'matlabThreads',maxNumCompThreads);
layers=atm.layer;parameters=[atm.L0;atm.r0;atm.wavelength;[layers.fractionnalR0].'];
keyInput=[parameters;real(inner(:));imag(inner(:));real(outer(:));imag(outer(:))];
t=tic;key=oomaoFastArrayDigest(keyInput,jsonencode(identity));hashSeconds=toc(t);
file=fullfile(directory,[name,'.mat']);
[value,event]=oomaoFastCachedRecord(file,@produce,@validate,'oomao:AtmosphereCacheCorrupt');
A=value.A;B=value.B;event.key=key;event.hashSeconds=hashSeconds;event.totalSeconds=event.totalSeconds+hashSeconds;
    function record=produce()
        ZZt=phaseStats.covarianceMatrix(inner,atm);
        ZXt=phaseStats.covarianceMatrix(inner,outer,atm);
        a=ZXt'/ZZt;clear ZZt
        XXt=phaseStats.covarianceMatrix(outer,atm);
        BBt=XXt-a*ZXt;clear XXt ZXt
        bb=chol(BBt,'lower');
        record=struct('key',key,'identity',identity,'value',struct('A',a,'B',bb), ...
            'digestA',oomaoFastArrayDigest(a,[key,'A']),'digestB',oomaoFastArrayDigest(bb,[key,'B']));
    end
    function value=validate(record)
        fields={'key','identity','value','digestA','digestB'};
        if ~isstruct(record)||~isscalar(record)||~all(isfield(record,fields))|| ...
          ~isstruct(record.identity)||~isscalar(record.identity)||~all(isfield(record.identity,fieldnames(identity)))|| ...
          ~isequal(record.identity.schema,identity.schema)||~isequal(record.identity.algorithm,identity.algorithm)|| ...
          ~isequal(record.identity.innerShape,size(inner))||~isequal(record.identity.outerShape,size(outer))|| ...
          ~isstruct(record.value)||~all(isfield(record.value,{'A','B'}))
            error('oomao:AtmosphereCacheCorrupt','Invalid atmosphere cache %s; preserve and inspect it.',file);
        end
        if ~isequal(record.identity,identity)
            t=tic;key=oomaoFastArrayDigest(keyInput,jsonencode(record.identity));hashSeconds=hashSeconds+toc(t);
        end
        if ~strcmp(record.key,key)
            error('oomao:AtmosphereCacheCorrupt','Atmosphere inputs differ in %s; preserve and inspect it.',file);
        end
        value=record.value;a=value.A;bb=value.B;
        if ~isa(a,'double')||~isreal(a)||issparse(a)||~isequal(size(a),[numel(outer),numel(inner)])|| ...
          ~isa(bb,'double')||~isreal(bb)||issparse(bb)||~isequal(size(bb),[numel(outer),numel(outer)])|| ...
          any(~isfinite(a(:)))||any(~isfinite(bb(:)))|| ...
          ~strcmp(record.digestA,oomaoFastArrayDigest(a,[key,'A']))||~strcmp(record.digestB,oomaoFastArrayDigest(bb,[key,'B']))
            error('oomao:AtmosphereCacheCorrupt','Invalid matrix bytes in %s; preserve and inspect it.',file);
        end
    end
end
