import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Brain,
  CheckCircle2,
  ImageIcon,
  Loader2,
  PawPrint,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  Layers3,
} from 'lucide-react';

const TRAINING_ACTIVE_STATES = new Set(['preparing', 'training']);
const SELECTION_MODES = [
  { id: 'default', label: 'Default only' },
  { id: 'single', label: 'Single model' },
  { id: 'selected', label: 'Selected models' },
  { id: 'all', label: 'All models' },
];

function CustomModelManager({
  models,
  training,
  clipEnabled,
  selectionMode,
  selectedModelIds,
  onSelectionModeChange,
  onSelectedModelIdsChange,
  onRefresh,
}) {
  const [animalName, setAnimalName] = useState('');
  const [trainingFiles, setTrainingFiles] = useState([]);
  const [targetModelId, setTargetModelId] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [deletingModelId, setDeletingModelId] = useState(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

  const completedModels = useMemo(
    () => models.filter((model) => model.status === 'completed'),
    [models],
  );
  const trainingActive = TRAINING_ACTIVE_STATES.has(training?.status);
  const targetModel = useMemo(
    () => models.find((model) => model.id === targetModelId) || null,
    [models, targetModelId],
  );

  useEffect(() => {
    if (targetModelId && !targetModel) {
      setTargetModelId(null);
      setAnimalName('');
    }
  }, [targetModel, targetModelId]);

  const handleFiles = (event) => {
    const files = Array.from(event.target.files || []);
    setTrainingFiles(files);
    setError(null);
    setMessage('');
  };

  const handleStartTraining = async () => {
    if (!animalName.trim() && !targetModelId) {
      setError('Custom animal name is required.');
      return;
    }
    if (trainingFiles.length === 0) {
      setError('Select custom animal images first.');
      return;
    }

    setIsSubmitting(true);
    setUploadPercent(0);
    setError(null);
    setMessage('');

    const formData = new FormData();
    formData.append('name', animalName.trim());
    if (targetModelId) {
      formData.append('existing_model_id', targetModelId);
    }
    trainingFiles.forEach((file) => {
      formData.append('images', file);
    });

    try {
      const response = await axios.post('/api/custom-models/train', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (event) => {
          if (event.total) {
            setUploadPercent(Math.round((event.loaded / event.total) * 100));
          }
        },
      });

      setMessage(response.data.message || 'Custom animal training started.');
      setAnimalName('');
      setTrainingFiles([]);
      setTargetModelId(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      onRefresh();
    } catch (requestError) {
      setError(requestError.response?.data?.error || 'Failed to start custom training.');
    } finally {
      setIsSubmitting(false);
      setUploadPercent(0);
    }
  };

  const prepareRetrain = (model) => {
    setTargetModelId(model.id);
    setAnimalName(model.name);
    setTrainingFiles([]);
    setError(null);
    setMessage(`Selected ${model.name}. Choose more images to append, then train again.`);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const resetToNewModel = () => {
    setTargetModelId(null);
    setAnimalName('');
    setTrainingFiles([]);
    setError(null);
    setMessage('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDeleteModel = async (model) => {
    setDeletingModelId(model.id);
    setError(null);
    setMessage('');

    try {
      const response = await axios.delete(`/api/custom-models/${model.id}`);
      if (targetModelId === model.id) {
        resetToNewModel();
      }
      onSelectedModelIdsChange(selectedModelIds.filter((id) => id !== model.id));
      setMessage(response.data.message || `${model.name} deleted.`);
      await onRefresh();
    } catch (requestError) {
      setError(requestError.response?.data?.error || 'Failed to delete custom model.');
    } finally {
      setDeletingModelId(null);
    }
  };

  const handleModeChange = (mode) => {
    onSelectionModeChange(mode);

    if (mode === 'default' || mode === 'all') {
      onSelectedModelIdsChange([]);
      return;
    }

    if (mode === 'single') {
      const firstAvailableId = completedModels[0]?.id;
      onSelectedModelIdsChange(firstAvailableId ? [firstAvailableId] : []);
    }
  };

  const toggleSelectedModel = (modelId) => {
    if (selectedModelIds.includes(modelId)) {
      onSelectedModelIdsChange(selectedModelIds.filter((id) => id !== modelId));
      return;
    }

    onSelectedModelIdsChange([...selectedModelIds, modelId]);
  };

  return (
    <div className="dashboard-panel dashboard-panel-spacious space-y-6">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <h3 className="text-slate-100 font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5 text-amber-300" />
            Custom Animal Models
          </h3>
          <p className="text-slate-400 text-sm mt-2 max-w-2xl">
            Add one animal name, upload many same-type images, and train a custom recognizer. Later you can run one model, many selected models, or all trained models together.
          </p>
        </div>
        <div className="dashboard-subpanel rounded-[18px] px-4 py-3 min-w-52">
          <p className="text-slate-300 text-sm font-medium">Current custom mode</p>
          <p className="text-amber-300 text-sm mt-1 capitalize">{selectionMode}</p>
          <p className="text-slate-500 text-xs mt-2">
            Completed models: {completedModels.length}
          </p>
        </div>
      </div>

      <div className="grid items-start gap-5 xl:grid-cols-12">
        <div className="dashboard-subpanel rounded-[22px] p-4 space-y-4 xl:col-span-7">
          <div>
            <p className="text-slate-100 font-medium flex items-center gap-2">
              <PawPrint className="w-4 h-4 text-emerald-300" />
              {targetModel ? `Add Images To ${targetModel.name}` : 'Add Custom Animal'}
            </p>
            <p className="text-slate-400 text-sm mt-2">
              {targetModel
                ? 'New images will be saved in the same model folder and then the model will retrain without overwriting the old images.'
                : 'Example: Tiger, Panda, Black Panther. More images usually mean better recognition.'}
            </p>
          </div>

          <div className="space-y-3">
            <input
              type="text"
              value={animalName}
              onChange={(event) => setAnimalName(event.target.value)}
              placeholder="Animal name"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-400/50"
            />

            {targetModel && (
              <button
                type="button"
                onClick={resetToNewModel}
                className="inline-flex items-center gap-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 text-sm transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Create New Model Instead
              </button>
            )}

            <div className="rounded-[18px] border border-dashed border-slate-600 bg-slate-950/60 p-4">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <div>
                  <p className="text-slate-200 text-sm font-medium">Training images</p>
                  <p className="text-slate-500 text-xs mt-1">
                    Upload at least 5 images. More images will help the custom model.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isSubmitting || trainingActive}
                  className="inline-flex items-center gap-2 rounded-lg bg-amber-400 hover:bg-amber-300 disabled:bg-amber-400/50 text-slate-950 font-medium px-4 py-2 transition-colors"
                >
                  <Upload className="w-4 h-4" />
                  Choose Images
                </button>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.avif,.bmp,.gif,.heic,.heif,.jfif,.jpeg,.jpg,.png,.tif,.tiff,.webp"
                multiple
                onChange={handleFiles}
                className="hidden"
              />

              <div className="mt-3 text-sm text-slate-400 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-slate-500" />
                {trainingFiles.length > 0
                  ? `${trainingFiles.length} image${trainingFiles.length !== 1 ? 's' : ''} selected`
                  : 'No images selected yet'}
              </div>

              {isSubmitting && uploadPercent > 0 && (
                <div className="mt-3">
                  <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                    <span>Uploading custom images</span>
                    <span>{uploadPercent}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-amber-400 transition-all"
                      style={{ width: `${uploadPercent}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={handleStartTraining}
            disabled={!clipEnabled || isSubmitting || trainingActive}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-500/50 text-slate-950 font-medium px-4 py-3 transition-colors"
          >
            {isSubmitting || trainingActive ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {targetModel ? 'Append Images And Retrain' : 'Train Custom Animal'}
          </button>

          {!clipEnabled && (
            <p className="text-red-300 text-sm">
              CLIP is not available, so custom animal training is disabled on this machine.
            </p>
          )}

          {message && (
            <div className="status-surface status-success text-sm text-green-300">
              {message}
            </div>
          )}

          {error && (
            <div className="status-surface status-error text-sm text-red-300">
              {error}
            </div>
          )}
        </div>

        <div className="dashboard-subpanel rounded-[22px] p-4 space-y-4 xl:col-span-5">
          <div>
            <p className="text-slate-100 font-medium">Training Status</p>
            <p className="text-slate-400 text-sm mt-2 capitalize">
              Status: {training?.status || 'idle'}
            </p>
            {training?.model_name && (
              <p className="text-slate-500 text-xs mt-1">
                Active model: {training.model_name}
              </p>
            )}
          </div>

          {training?.total_images > 0 && (
            <div>
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>
                  {training.processed_images || 0} / {training.total_images} images
                </span>
                <span>{training.progress || 0}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-sky-400 transition-all"
                  style={{ width: `${training.progress || 0}%` }}
                />
              </div>
            </div>
          )}

          {training?.message && (
            <p className="text-slate-300 text-sm">{training.message}</p>
          )}

          {training?.error && (
            <p className="text-red-300 text-sm">{training.error}</p>
          )}

          <p className="text-slate-500 text-xs">
            Custom animal training builds CLIP embeddings from the uploaded animal images.
          </p>
        </div>
      </div>

      <div className="dashboard-subpanel rounded-[22px] p-4 space-y-4">
        <div>
          <p className="text-slate-100 font-medium">Detection Model Selection</p>
          <p className="text-slate-400 text-sm mt-2">
            Choose how custom models will be used when a new image is detected.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {SELECTION_MODES.map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={() => handleModeChange(mode.id)}
              className={`rounded-full px-4 py-2 text-sm transition-colors ${
                selectionMode === mode.id
                  ? 'bg-sky-400 text-slate-950 font-medium'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        {selectionMode === 'single' && (
          <select
            value={selectedModelIds[0] || ''}
            onChange={(event) => onSelectedModelIdsChange(event.target.value ? [event.target.value] : [])}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-400/50"
          >
            <option value="">Select one custom model</option>
            {completedModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        )}

        {selectionMode === 'selected' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {completedModels.map((model) => (
              <label
                key={model.id}
                className="flex items-center gap-3 rounded-lg border border-white/10 bg-slate-950/60 px-4 py-3 text-slate-300"
              >
                <input
                  type="checkbox"
                  checked={selectedModelIds.includes(model.id)}
                  onChange={() => toggleSelectedModel(model.id)}
                  className="rounded border-slate-600 bg-slate-900 text-sky-400 focus:ring-sky-400"
                />
                <span>{model.name}</span>
              </label>
            ))}
            {completedModels.length === 0 && (
              <p className="text-slate-500 text-sm">No completed custom models yet.</p>
            )}
          </div>
        )}
      </div>

      <div className="space-y-4">
        <div>
          <p className="text-slate-100 font-medium flex items-center gap-2">
            <Layers3 className="w-4 h-4 text-sky-300" />
            Animal Model Library
          </p>
          <p className="text-slate-400 text-sm mt-2">
            Manage trained animal profiles, add more images, or remove models you no longer need.
          </p>
        </div>

        <div className="grid auto-rows-fr gap-4 lg:grid-cols-2">
        {models.map((model) => {
          const isSelected = selectedModelIds.includes(model.id);
          return (
            <div
              key={model.id}
              className={`dashboard-subpanel rounded-[22px] border p-4 h-full flex flex-col ${
                isSelected
                  ? 'border-sky-400/60 bg-sky-500/10 shadow-[0_0_0_1px_rgba(56,189,248,0.18)]'
                  : 'border-slate-700 bg-slate-900/40'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 min-w-0">
                  <div className="h-11 w-11 rounded-2xl bg-slate-950/80 border border-white/10 flex items-center justify-center shrink-0">
                    <PawPrint className="w-5 h-5 text-emerald-300" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-slate-100 font-medium truncate">{model.name}</p>
                    <p className="text-slate-500 text-xs mt-1 truncate">{model.id}</p>
                  </div>
                </div>
                <span
                  className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide ${
                    model.status === 'completed'
                      ? 'bg-emerald-500/12 text-green-300'
                      : model.status === 'failed'
                        ? 'bg-red-500/12 text-red-300'
                        : 'bg-amber-500/12 text-amber-300'
                  }`}
                >
                  {model.status}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-2xl border border-white/8 bg-slate-950/55 px-3 py-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Images</p>
                  <p className="mt-2 text-lg font-semibold text-slate-100">{model.image_count || 0}</p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-slate-950/55 px-3 py-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Threshold</p>
                  <p className="mt-2 text-lg font-semibold text-slate-100">{model.threshold || '-'}</p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-slate-950/55 px-3 py-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Trained</p>
                  <p className="mt-2 text-sm font-medium text-slate-200">{model.trained_at || 'Not yet'}</p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-slate-950/55 px-3 py-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Last Update</p>
                  <p className="mt-2 text-sm font-medium text-slate-200">{model.last_appended_at || 'Initial set'}</p>
                </div>
              </div>

              {model.status === 'completed' && (
                <div className="mt-4 inline-flex items-center gap-2 rounded-full bg-emerald-500/10 px-3 py-1.5 text-xs text-green-300 w-fit">
                  <CheckCircle2 className="w-4 h-4" />
                  Ready for detection
                </div>
              )}

              {model.error && (
                <div className="status-surface status-error mt-4 text-sm text-red-300">
                  Error: {model.error}
                </div>
              )}

              <div className="mt-auto pt-5 grid grid-cols-1 sm:grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => prepareRetrain(model)}
                  disabled={trainingActive}
                  className="inline-flex w-full justify-center items-center gap-2 rounded-xl bg-amber-400 hover:bg-amber-300 disabled:bg-amber-400/50 text-slate-950 px-3 py-2.5 text-sm font-medium transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Add More Images
                </button>
                <button
                  type="button"
                  onClick={() => handleDeleteModel(model)}
                  disabled={deletingModelId === model.id || trainingActive}
                  className="inline-flex w-full justify-center items-center gap-2 rounded-xl bg-red-500/80 hover:bg-red-400 disabled:bg-red-500/40 text-white px-3 py-2.5 text-sm font-medium transition-colors"
                >
                  {deletingModelId === model.id ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                  Delete
                </button>
              </div>
            </div>
          );
        })}
        </div>

        {models.length === 0 && (
          <div className="dashboard-subpanel rounded-[22px] border border-white/10 p-8 text-sm text-slate-500 text-center">
            No custom animal models yet. Add one above, upload same-type images, then train it.
          </div>
        )}
      </div>
    </div>
  );
}

export default CustomModelManager;
