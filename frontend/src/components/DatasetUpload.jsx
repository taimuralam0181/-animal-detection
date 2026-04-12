import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Database, FileArchive, Loader2, Play, Upload } from 'lucide-react';

function DatasetUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploadMessage, setUploadMessage] = useState('');
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [status, setStatus] = useState(null);
  const [training, setTraining] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let ignore = false;

    const loadStatus = async () => {
      try {
        const [datasetResponse, trainingResponse] = await Promise.all([
          axios.get('/api/datasets/status'),
          axios.get('/api/training/status'),
        ]);

        if (!ignore) {
          setStatus(datasetResponse.data);
          setTraining(trainingResponse.data.training);
        }
      } catch (requestError) {
        if (!ignore) {
          setError(requestError.response?.data?.error || 'Failed to load dataset status');
        }
      }
    };

    loadStatus();
    const intervalId = setInterval(loadStatus, 5000);

    return () => {
      ignore = true;
      clearInterval(intervalId);
    };
  }, []);

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setIsUploading(true);
    setUploadPercent(0);
    setUploadMessage('Uploading dataset...');
    setError(null);
    setUploadResult(null);

    const formData = new FormData();
    formData.append('dataset', file);

    try {
      const response = await axios.post('/api/datasets/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (event) => {
          if (event.total) {
            setUploadPercent(Math.round((event.loaded / event.total) * 100));
          }
        },
      });

      setUploadResult(response.data);
      setUploadMessage(response.data.message || 'Dataset imported successfully.');
      setStatus({
        success: true,
        dataset: response.data.active_dataset,
        target_classes: status?.target_classes || [],
        message: response.data.message,
      });
    } catch (requestError) {
      setError(requestError.response?.data?.error || requestError.message || 'Dataset upload failed');
      setUploadMessage('');
    } finally {
      setIsUploading(false);
      setUploadPercent(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleStartTraining = async () => {
    setIsStartingTraining(true);
    setTrainingError(null);

    try {
      const response = await axios.post('/api/training/start');
      setTraining(response.data.training);
    } catch (requestError) {
      setTrainingError(
        requestError.response?.data?.error || requestError.message || 'Failed to start training',
      );
    } finally {
      setIsStartingTraining(false);
    }
  };

  const splitRows = status?.dataset?.splits
    ? Object.entries(status.dataset.splits)
    : [];
  const trainingStatus = training?.status || 'idle';
  const canStartTraining = training?.dataset_ready && trainingStatus !== 'running';
  const trainingStatusClass = {
    idle: 'text-slate-300',
    running: 'text-amber-300',
    completed: 'text-green-300',
    failed: 'text-red-300',
  }[trainingStatus] || 'text-slate-300';

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-slate-100 font-semibold flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-400" />
            Add Training Dataset
          </h3>
          <div className="text-slate-400 text-sm mt-2 space-y-1">
            <p>Upload a YOLO dataset ZIP. Matching image/label pairs will be merged into `datasets/animals10`.</p>
            {isUploading && uploadPercent > 0 && (
              <p className="text-blue-300 text-xs uppercase tracking-wide">
                Uploading... {uploadPercent}%
              </p>
            )}
            {!isUploading && uploadMessage && (
              <p className="text-green-300 text-xs uppercase tracking-wide">
                {uploadMessage}
              </p>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-500/60 text-slate-950 font-medium px-4 py-2 transition-colors"
        >
          {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          Upload ZIP
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,application/zip"
        onChange={handleUpload}
        className="hidden"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {splitRows.map(([splitName, splitData]) => (
          <div
            key={splitName}
            className="rounded-lg border border-slate-700 bg-slate-900/50 p-4"
          >
            <p className="text-slate-200 font-medium capitalize">{splitName}</p>
            <p className="text-slate-400 text-sm mt-2">
              Images: {splitData.images}
            </p>
            <p className="text-slate-400 text-sm">
              Labels: {splitData.labels}
            </p>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
        <p className="text-slate-300 text-sm font-medium flex items-center gap-2">
          <FileArchive className="w-4 h-4 text-sky-400" />
          ZIP layout
        </p>
        <p className="text-slate-400 text-sm mt-2">
          Required: `images/train`, `images/val`, `labels/train`, `labels/val`
        </p>
        <p className="text-slate-500 text-xs mt-2">
          Optional: `images/test` and `labels/test`
        </p>
        <p className="text-slate-500 text-xs mt-2">
          Default dataset ZIP upload limit: 12 GB
        </p>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-slate-100 font-medium">Train Custom Model</p>
            <p className="text-slate-400 text-sm mt-1">
              Start YOLO training with the dataset currently inside `datasets/animals10`.
            </p>
            <p className={`text-sm mt-2 capitalize ${trainingStatusClass}`}>
              Status: {trainingStatus}
            </p>
          </div>
          <button
            type="button"
            onClick={handleStartTraining}
            disabled={!canStartTraining || isStartingTraining}
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:bg-sky-500/50 text-slate-950 font-medium px-4 py-2 transition-colors"
          >
            {isStartingTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Start Training
          </button>
        </div>

        {!training?.dataset_ready && (
          <p className="text-amber-300 text-sm">
            Training disabled until both `train` and `val` splits have image/label pairs.
          </p>
        )}

        {training?.started_at && (
          <p className="text-slate-400 text-sm">
            Started: {training.started_at}
            {training.finished_at ? ` | Finished: ${training.finished_at}` : ''}
          </p>
        )}

        {training?.log_tail?.length > 0 && (
          <div className="rounded-lg bg-slate-950 border border-slate-700 p-3">
            <p className="text-slate-300 text-xs uppercase tracking-wide mb-2">
              Training Log
            </p>
            <pre className="text-xs text-slate-400 whitespace-pre-wrap break-words max-h-52 overflow-auto">
              {training.log_tail.join('\n')}
            </pre>
          </div>
        )}

        <p className="text-slate-500 text-xs">
          CPU training can take a long time. Keep the backend running while the job is active.
        </p>
      </div>

      {uploadResult && (
        <div className="rounded-lg border border-green-500/30 bg-green-500/10 p-4">
          <p className="text-green-300 font-medium">{uploadResult.message}</p>
          <p className="text-slate-300 text-sm mt-2">
            Imported from: {uploadResult.dataset_name}
          </p>
          <p className="text-slate-400 text-sm mt-2">
            Train: {uploadResult.imported_summary.imported.train} imported, {uploadResult.imported_summary.skipped_without_label.train} skipped
          </p>
          <p className="text-slate-400 text-sm">
            Val: {uploadResult.imported_summary.imported.val} imported, {uploadResult.imported_summary.skipped_without_label.val} skipped
          </p>
          <p className="text-slate-400 text-sm">
            Test: {uploadResult.imported_summary.imported.test} imported, {uploadResult.imported_summary.skipped_without_label.test} skipped
          </p>
        </div>
      )}

      {trainingError && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-red-300 font-medium">Training error</p>
          <p className="text-slate-300 text-sm mt-1">{trainingError}</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-red-300 font-medium">Dataset upload error</p>
          <p className="text-slate-300 text-sm mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}

export default DatasetUpload;
