function directory=oomaoFastCacheDirectory(root,release)
% Readable release namespace; producer environments remain in matrix records.
if isempty(regexp(release,'^\d+\.\d+$','once'))
    error('oomao:CacheRelease','An exact OOMAO major.minor release is required.');
end
directory=fullfile(root,['v',release]);
if ~isfolder(directory)
    [ok,message]=mkdir(directory);if ~ok,error('oomao:CacheDirectory','%s',message);end
end
end
