import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DetectionResults from './components/DetectionResults';
import DatasetUpload from './components/DatasetUpload';
import CustomModelManager from './components/CustomModelManager';

const LOCAL_PROVIDER = 'local';
const LOCAL_PROVIDER_LABEL = 'Local YOLO + CLIP';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [customModels, setCustomModels] = useState([]);
  const [customTraining, setCustomTraining] = useState(null);
  const [clipEnabled, setClipEnabled] = useState(false);
  const [customSelectionMode, setCustomSelectionMode] = useState('default');
  const [selectedCustomModelIds, setSelectedCustomModelIds] = useState([]);
  const provider = LOCAL_PROVIDER;

  const refreshCustomModels = async () => {
    const response = await axios.get('/api/custom-models/status');
    setCustomModels(response.data.models || []);
    setCustomTraining(response.data.training || null);
    setClipEnabled(Boolean(response.data.clip_enabled));
  };

  useEffect(() => {
    let ignore = false;

    const loadCustomModels = async () => {
      try {
        const response = await axios.get('/api/custom-models/status');
        if (!ignore) {
          setCustomModels(response.data.models || []);
          setCustomTraining(response.data.training || null);
          setClipEnabled(Boolean(response.data.clip_enabled));
        }
      } catch (_error) {
        if (!ignore) {
          setClipEnabled(false);
        }
      }
    };

    loadCustomModels();
    const intervalId = setInterval(loadCustomModels, 5000);

    return () => {
      ignore = true;
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    const completedIds = customModels
      .filter((model) => model.status === 'completed')
      .map((model) => model.id);

    if (customSelectionMode === 'single') {
      const firstValidId = selectedCustomModelIds.find((id) => completedIds.includes(id));
      if (firstValidId) {
        if (selectedCustomModelIds.length !== 1 || selectedCustomModelIds[0] !== firstValidId) {
          setSelectedCustomModelIds([firstValidId]);
        }
      } else if (completedIds[0]) {
        setSelectedCustomModelIds([completedIds[0]]);
      }
      return;
    }

    if (customSelectionMode === 'selected') {
      const validIds = selectedCustomModelIds.filter((id) => completedIds.includes(id));
      if (validIds.length !== selectedCustomModelIds.length) {
        setSelectedCustomModelIds(validIds);
      }
      return;
    }

    if (selectedCustomModelIds.length > 0) {
      setSelectedCustomModelIds([]);
    }
  }, [customModels, customSelectionMode, selectedCustomModelIds]);

  const customSelectionLabel = useMemo(() => {
    if (customSelectionMode === 'default') {
      return 'No custom model';
    }
    if (customSelectionMode === 'all') {
      return `All models (${customModels.filter((model) => model.status === 'completed').length})`;
    }
    if (selectedCustomModelIds.length === 0) {
      return 'No custom model selected';
    }

    const selectedNames = customModels
      .filter((model) => selectedCustomModelIds.includes(model.id))
      .map((model) => model.name);

    return selectedNames.join(', ');
  }, [customModels, customSelectionMode, selectedCustomModelIds]);

  const handleImageSelect = async (file) => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('provider', provider);
    formData.append('custom_mode', customSelectionMode);
    formData.append('custom_model_ids', JSON.stringify(selectedCustomModelIds));

    try {
      const response = await axios.post('/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || 'Detection failed');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Failed to connect to server';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900">
      <Header />

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Upload Section */}
          <section>
            <h2 className="text-lg font-semibold text-slate-200 mb-4">
              Upload Image
            </h2>
            <ImageUpload
              onImageSelect={handleImageSelect}
              isLoading={isLoading}
              providerLabel={LOCAL_PROVIDER_LABEL}
              customSelectionLabel={customSelectionLabel}
            />
          </section>

          <section>
            <h2 className="text-lg font-semibold text-slate-200 mb-4">
              Custom Animal Training
            </h2>
            <CustomModelManager
              models={customModels}
              training={customTraining}
              clipEnabled={clipEnabled}
              selectionMode={customSelectionMode}
              selectedModelIds={selectedCustomModelIds}
              onSelectionModeChange={setCustomSelectionMode}
              onSelectedModelIdsChange={setSelectedCustomModelIds}
              onRefresh={refreshCustomModels}
            />
          </section>

          {/* Results Section */}
          {(results || error) && (
            <section>
              <h2 className="text-lg font-semibold text-slate-200 mb-4">
                Results
              </h2>
              <DetectionResults results={results} error={error} provider={provider} />
            </section>
          )}

          <section>
            <h2 className="text-lg font-semibold text-slate-200 mb-4">
              Dataset
            </h2>
            <DatasetUpload />
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700/50 py-6 mt-12">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-slate-500 text-sm">
            Animal Detection using local YOLO + CLIP - Built with React + Flask
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
