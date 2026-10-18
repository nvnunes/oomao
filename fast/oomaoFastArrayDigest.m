function hex = oomaoFastArrayDigest(value,context)
% Hash complete numeric content in bounded chunks, without consuming RNG.
% Dimensions, class and real/complex layout are part of the identity.
if ~(isnumeric(value)&&~issparse(value)),error('oomao:DigestInput','A full numeric array is required.');end
header=jsonencode(struct('context',context,'class',class(value),'size',size(value),'real',isreal(value)));
headerBytes=unicode2native(header,'UTF-8');
if isreal(value)&&exist('oomao_fast_digest_mex','file')==3
 bytes=oomao_fast_digest_mex(value,headerBytes);
 hex=lower(reshape(dec2hex(bytes,2).',1,[]));return
end
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(headerBytes,'int8'));
chunk=262144;
for first=1:chunk:numel(value)
 part=value(first:min(first+chunk-1,numel(value)));
 md.update(typecast(real(part(:)),'int8'));
 if ~isreal(value),md.update(typecast(imag(part(:)),'int8'));end
end
hex=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
