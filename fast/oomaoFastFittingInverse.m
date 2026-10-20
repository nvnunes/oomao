function [out,event]=oomaoFastFittingInverse(matrix,tolerance,directory,threads,name,configuration)
% Native pinv; scientific metadata decides reuse, checksums protect payloads.
identity=struct('schema',2,'algorithm','MATLAB-pinv','tolerance',{tolerance}, ...
 'matlab',version,'release',version('-release'),'computer',computer, ...
 'blas',version('-blas'),'lapack',version('-lapack'),'threads',threads, ...
 'matlabThreads',maxNumCompThreads);
model='native-gaussian-dm-fitting/v1';
file=oomaoFastCacheFile(directory,name,identity.algorithm);nativeSeconds=0;
[out,event]=oomaoFastCachedRecord(file,@produce,@validate,'oomao:FittingCacheCorrupt');
event.computeSeconds=nativeSeconds;event.name=name;event.shape=size(matrix);
    function record=produce()
        t=tic;inverse=pinv(matrix,tolerance{:});nativeSeconds=toc(t);
        record=struct('inputShape',size(matrix),'inputClass',class(matrix),...
            'identity',identity,'scientific_model',model,'configuration',configuration,...
            'value',inverse,'digest',oomaoFastArrayDigest(inverse,name));
    end
    function value=validate(record)
        fields={'inputShape','inputClass','value','digest','identity'};
        if ~isstruct(record)||~isscalar(record)||~all(isfield(record,fields))|| ...
         ~isstruct(record.identity)||~isscalar(record.identity)||~all(isfield(record.identity,{'schema','algorithm','tolerance'}))|| ...
         ~any(record.identity.schema==[1 2])||~strcmp(record.identity.algorithm,identity.algorithm)|| ...
         ~isequal(record.identity.tolerance,tolerance)
            error('oomao:FittingCacheCorrupt','Invalid cache metadata in %s; preserve and inspect it.',file);
        end
        if record.identity.schema==1
            oomaoMatrixCacheMetadata(file,model,configuration);
            if ~isfield(record,'key'),error('oomao:FittingCacheCorrupt','Missing original payload context.');end
            context=record.key; % Original checksum context; never compared with current inputs.
        else
            if ~all(isfield(record,{'scientific_model','configuration'}))||~strcmp(record.scientific_model,model)||...
                    ~isequaln(record.configuration,configuration)
                error('oomao:FittingCacheIncompatible','Scientific inputs differ in %s.',file);
            end
            context=name;
        end
        if ~isequal(record.inputShape,size(matrix))||~strcmp(record.inputClass,class(matrix))|| ...
         ~isequal(size(record.value),fliplr(size(matrix)))||~strcmp(class(record.value),class(matrix))|| ...
         any(~isfinite(record.value(:)))||~strcmp(record.digest,oomaoFastArrayDigest(record.value,context))
            error('oomao:FittingCacheCorrupt','Invalid cache payload in %s; preserve and inspect it.',file);
        end
        value=record.value;
    end
end
