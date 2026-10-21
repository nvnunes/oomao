function directory=oomaoFastCacheDirectory(root,release)
% Locate a readable release namespace without creating unused directories.
if isempty(regexp(release,'^\d+\.\d+$','once'))
    error('oomao:CacheRelease','An exact OOMAO major.minor release is required.');
end
directory=fullfile(root,['v',release]);
end
