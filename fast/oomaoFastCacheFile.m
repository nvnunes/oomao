function file=oomaoFastCacheFile(directory,name,algorithm)
% Prefer the requested release, then reuse another release in place.
% Release folders retain provenance; the caller validates the scientific model,
% parameters and payload integrity. Version numbers do not prove compatibility.
if ~any(strcmp(algorithm,{'MATLAB-pinv','native-atmosphere-border'}))
    error('oomao:CacheScientificContract','Unsupported matrix-cache contract: %s.',algorithm);
end
file=fullfile(directory,[name,'.mat']);
if isfile(file),return;end % Invalid current work must fail validation, not fall back.
[root,release,extension]=fileparts(directory);release=[release,extension];
if isempty(regexp(release,'^v[0-9]+\.[0-9]+$','once')),return;end
folders=dir(fullfile(root,'v*'));
for i=1:numel(folders)
    if ~folders(i).isdir || isempty(regexp(folders(i).name,'^v[0-9]+\.[0-9]+$','once'))
        continue
    end
    candidate=fullfile(root,folders(i).name,[name,'.mat']);
    if isfile(candidate),file=candidate;return;end
end
end
