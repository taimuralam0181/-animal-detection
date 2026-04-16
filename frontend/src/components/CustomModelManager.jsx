import { useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Brain,
  CheckCircle2,
  ImageIcon,
  Loader2,
  PawPrint,
  Sparkles,
  Upload,
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
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

  const completedModels = useMemo(
    () => models.filter((model) => model.status === 'completed'),
    [models],
  );
  const trainingActive = TRAINING_ACTIVE_STATES.has(training?.status);

  const handleFiles = (event) => {
    const files = Array.from(event.target.files || []);
    setTrainingFiles(files);
    setError(null);
    setMessage('');
  };

  const handleStartTraining = async () => {
    if (!animalName.trim()) {
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
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 space-y-6">
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
        <div className="rounded-lg border border-slate-700 bg-slate-900/60 px-4 py-3 min-w-52">
          <p className="text-slate-300 text-sm font-medium">Current custom mode</p>
          <p className="text-amber-300 text-sm mt-1 capitalize">{selectionMode}</p>
          <p className="text-slate-500 text-xs mt-2">
            Completed models: {completedModels.length}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-5">
        <div className="rounded-xl border border-slate-700 bg-slate-900/40 p-4 space-y-4">
          <div>
            <p className="text-slate-100 font-medium flex items-center gap-2">
              <PawPrint className="w-4 h-4 text-emerald-300" />
              Add Custom Animal
            </p>
            <p className="text-slate-400 text-sm mt-2">
              Example: Tiger, Panda, Black Panther. More images usually mean better recognition.
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

            <div className="rounded-lg border border-dashed border-slate-600 bg-slate-950/60 p-4">
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
            Train Custom Animal
          </button>

          {!clipEnabled && (
            <p className="text-red-300 text-sm">
              CLIP is not available, so custom animal training is disabled on this machine.
            </p>
          )}

          {message && (
            <div className="rounded-lg border border-green-500/30 bg-green-500/10 p-3 text-sm text-green-300">
              {message}
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900/40 p-4 space-y-4">
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

      <div className="rounded-xl border border-slate-700 bg-slate-900/40 p-4 space-y-4">
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
                className="flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-950/60 px-4 py-3 text-slate-300"
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((model) => {
          const isSelected = selectedModelIds.includes(model.id);
          return (
            <div
              key={model.id}
              className={`rounded-xl border p-4 ${
                isSelected
                  ? 'border-sky-400/60 bg-sky-500/10'
                  : 'border-slate-700 bg-slate-900/40'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-slate-100 font-medium">{model.name}</p>
                  <p className="text-slate-500 text-xs mt-1">{model.id}</p>
                </div>
                <span
                  className={`text-xs uppercase tracking-wide ${
                    model.status === 'completed'
                      ? 'text-green-300'
                      : model.status === 'failed'
                        ? 'text-red-300'
                        : 'text-amber-300'
                  }`}
                >
                  {model.status}
                </span>
              </div>

              <div className="mt-3 text-sm text-slate-400 space-y-1">
                <p>Images: {model.image_count || 0}</p>
                {model.threshold && <p>Match threshold: {model.threshold}</p>}
                {model.trained_at && <p>Trained: {model.trained_at}</p>}
                {model.error && <p className="text-red-300">Error: {model.error}</p>}
              </div>

              {model.status === 'completed' && (
                <div className="mt-3 inline-flex items-center gap-2 text-xs text-green-300">
                  <CheckCircle2 className="w-4 h-4" />
                  Ready for detection
                </div>
              )}
            </div>
          );
        })}

        {models.length === 0 && (
          <div className="rounded-xl border border-slate-700 bg-slate-900/40 p-6 text-sm text-slate-500">
            No custom animal models yet. Add one above, upload same-type images, then train it.
          </div>
        )}
      </div>
    </div>
  );
}

export default CustomModelManager;
