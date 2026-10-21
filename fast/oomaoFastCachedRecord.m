function [value,event]=oomaoFastCachedRecord(file,producer,validator,errorId)
% Verified immutable cache publication. The owner defines record science/schema.
% A hard link publishes a complete file atomically and never replaces an entry.
clock=tic;event=struct('hit',false,'computeSeconds',0,'ioSeconds',0,'totalSeconds',0);
if isfile(file)
 t=tic;value=validator(readRecord(file,errorId));event.hit=true;
 event.ioSeconds=toc(t);event.totalSeconds=toc(clock);return
end
if ~isempty(getenv('OOMAO_BENCHMARK_FILE')) && ~strcmp(getenv('OOMAO_BENCHMARK_PREPARE'),'true')
 error('oomao:BenchmarkPreparation','Required matrix was not prepared: %s.',file);
end
directory=fileparts(file);
if ~isfolder(directory)
 [ok,message]=mkdir(directory);if ~ok,error('oomao:CacheDirectory','%s',message);end
end
t=tic;record=producer();value=validator(record);event.computeSeconds=toc(t);
temporary=fullfile(fileparts(file),['pending-',char(java.util.UUID.randomUUID),'.mat']);
cleanup=onCleanup(@()removeTemporary(temporary));t=tic;info=whos('record');
if info.bytes<1.8e9,save(temporary,'record','-v7');else,save(temporary,'record','-v7.3');end
try
 java.nio.file.Files.createLink(java.io.File(file).toPath,java.io.File(temporary).toPath);
catch problem
 if ~isfile(file),rethrow(problem);end
 previous=validator(readRecord(file,errorId));
 if ~isequal(previous,value),error('oomao:MatrixCacheConflict','Concurrent results differ for %s.',file);end
end
event.ioSeconds=toc(t);event.totalSeconds=toc(clock);
end
function record=readRecord(file,errorId)
try,stored=load(file,'record');
catch problem,error(errorId,'Cannot read cache %s (%s); preserve and inspect it.',file,problem.message);end
if ~isfield(stored,'record'),error(errorId,'Missing record in %s; preserve and inspect it.',file);end
record=stored.record;
end
function removeTemporary(file)
if isfile(file),delete(file);end
end
