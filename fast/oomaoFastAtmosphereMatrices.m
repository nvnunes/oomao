function [A,B,event]=oomaoFastAtmosphereMatrices(inner,outer,atm,directory,threads,name,geometry)
% Cache native stationary-atmosphere border matrices by scientific metadata.
identity=struct('schema',2,'algorithm','native-atmosphere-border', ...
 'innerShape',size(inner),'outerShape',size(outer),'matlab',version, ...
 'release',version('-release'),'computer',computer,'blas',version('-blas'), ...
 'lapack',version('-lapack'),'threads',threads,'matlabThreads',maxNumCompThreads);
layers=atm.layer;
configuration=struct('geometry',geometry,'outerScale',atm.L0,'r0',atm.r0,...
 'wavelength',atm.wavelength,'fractionalR0',[layers.fractionnalR0]);
model='native-atmosphere-border/v1';
file=oomaoFastCacheFile(directory,name,identity.algorithm);
[value,event]=oomaoFastCachedRecord(file,@produce,@validate,'oomao:AtmosphereCacheCorrupt');
A=value.A;B=value.B;event.name=name;
    function record=produce()
        ZZt=phaseStats.covarianceMatrix(inner,atm);
        ZXt=phaseStats.covarianceMatrix(inner,outer,atm);
        a=ZXt'/ZZt;clear ZZt
        XXt=phaseStats.covarianceMatrix(outer,atm);
        BBt=XXt-a*ZXt;clear XXt ZXt
        bb=chol(BBt,'lower');
        record=struct('identity',identity,'scientific_model',model,'configuration',configuration,...
            'value',struct('A',a,'B',bb),...
            'digestA',oomaoFastArrayDigest(a,[name,'A']),'digestB',oomaoFastArrayDigest(bb,[name,'B']));
    end
    function value=validate(record)
        fields={'identity','value','digestA','digestB'};
        if ~isstruct(record)||~isscalar(record)||~all(isfield(record,fields))|| ...
          ~isstruct(record.identity)||~isscalar(record.identity)||~all(isfield(record.identity,{'schema','algorithm','innerShape','outerShape'}))|| ...
          ~any(record.identity.schema==[1 2])||~strcmp(record.identity.algorithm,identity.algorithm)|| ...
          ~isequal(record.identity.innerShape,size(inner))||~isequal(record.identity.outerShape,size(outer))|| ...
          ~isstruct(record.value)||~all(isfield(record.value,{'A','B'}))
            error('oomao:AtmosphereCacheCorrupt','Invalid atmosphere cache %s; preserve and inspect it.',file);
        end
        if record.identity.schema==1
            oomaoMatrixCacheMetadata(file,model,configuration);
            if ~isfield(record,'key'),error('oomao:AtmosphereCacheCorrupt','Missing original payload context.');end
            context=record.key;
        else
            if ~all(isfield(record,{'scientific_model','configuration'}))||~strcmp(record.scientific_model,model)||...
                    ~isequaln(record.configuration,configuration)
                error('oomao:AtmosphereCacheIncompatible','Scientific inputs differ in %s.',file);
            end
            context=name;
        end
        value=record.value;a=value.A;bb=value.B;
        if ~isa(a,'double')||~isreal(a)||issparse(a)||~isequal(size(a),[numel(outer),numel(inner)])|| ...
          ~isa(bb,'double')||~isreal(bb)||issparse(bb)||~isequal(size(bb),[numel(outer),numel(outer)])|| ...
          any(~isfinite(a(:)))||any(~isfinite(bb(:)))|| ...
          ~strcmp(record.digestA,oomaoFastArrayDigest(a,[context,'A']))||~strcmp(record.digestB,oomaoFastArrayDigest(bb,[context,'B']))
            error('oomao:AtmosphereCacheCorrupt','Invalid matrix bytes in %s; preserve and inspect it.',file);
        end
    end
end
