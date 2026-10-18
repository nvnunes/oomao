function [out,event]=oomaoFastFittingInverse(matrix,tolerance,directory,threads,name)
% Native pinv; environment fields record provenance, not reuse restrictions.
clock=tic;
identity=struct('schema',1,'algorithm','MATLAB-pinv','tolerance',{tolerance}, ...
 'matlab',version,'release',version('-release'),'computer',computer, ...
 'blas',version('-blas'),'lapack',version('-lapack'),'threads',threads, ...
 'matlabThreads',maxNumCompThreads);
key=oomaoFastArrayDigest(matrix,jsonencode(identity));hashSeconds=toc(clock);
file=fullfile(directory,[name,'.mat']);nativeSeconds=0;
[out,event]=oomaoFastCachedRecord(file,@produce,@validate,'oomao:FittingCacheCorrupt');
event.computeSeconds=nativeSeconds;event.key=key;event.shape=size(matrix);
event.hashSeconds=hashSeconds;event.totalSeconds=toc(clock);
    function record=produce()
        t=tic;inverse=pinv(matrix,tolerance{:});nativeSeconds=toc(t);
        record=struct('key',key,'inputShape',size(matrix),'inputClass',class(matrix), ...
            'identity',identity,'value',inverse,'digest',oomaoFastArrayDigest(inverse,key));
    end
    function value=validate(record)
        fields={'key','inputShape','inputClass','value','digest','identity'};
        if ~isstruct(record)||~isscalar(record)||~all(isfield(record,fields))|| ...
         ~isstruct(record.identity)||~isscalar(record.identity)||~all(isfield(record.identity,fieldnames(identity)))|| ...
         ~isequal(record.identity.schema,identity.schema)||~isequal(record.identity.algorithm,identity.algorithm)|| ...
         ~isequal(record.identity.tolerance,tolerance)
            error('oomao:FittingCacheCorrupt','Invalid cache metadata in %s; preserve and inspect it.',file);
        end
        % Recheck current inputs using the stored producer identity, preserving
        % the existing file's input and output digests across environments.
        if ~isequal(record.identity,identity)
            t=tic;key=oomaoFastArrayDigest(matrix,jsonencode(record.identity));hashSeconds=hashSeconds+toc(t);
        end
        if ~strcmp(record.key,key)||~isequal(record.inputShape,size(matrix))||~strcmp(record.inputClass,class(matrix))|| ...
         ~isequal(size(record.value),fliplr(size(matrix)))||~strcmp(class(record.value),class(matrix))|| ...
         any(~isfinite(record.value(:)))||~strcmp(record.digest,oomaoFastArrayDigest(record.value,key))
            error('oomao:FittingCacheCorrupt','Invalid cache entry %s; preserve and inspect it.',key);
        end
        value=record.value;
    end
end
