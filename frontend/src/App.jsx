import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  Brain,
  CheckCircle2,
  Database,
  ImageIcon,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DetectionResults from './components/DetectionResults';
import DatasetUpload from './components/DatasetUpload';
import CustomModelManager from './components/CustomModelManager';
import DetectionHistory from './components/DetectionHistory';
import WorkflowSteps from './components/WorkflowSteps';

const LOCAL_PROVIDER = 'local';
const LOCAL_PROVIDER_LABEL = 'Local YOLO + CLIP';
const DETECTION_HISTORY_STORAGE_KEY = 'animal-dashboard-detection-history';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [customModels, setCustomModels] = useState([]);
  const [customTraining, setCustomTraining] = useState(null);
  const [datasetStatus, setDatasetStatus] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [clipEnabled, setClipEnabled] = useState(false);
  const [customSelectionMode, setCustomSelectionMode] = useState('default');
  const [selectedCustomModelIds, setSelectedCustomModelIds] = useState([]);
  const [detectionHistory, setDetectionHistory] = useState(() => {
    if (typeof window === 'undefined') {
      return [];
    }

    try {
      const rawHistory = window.localStorage.getItem(DETECTION_HISTORY_STORAGE_KEY);
      return rawHistory ? JSON.parse(rawHistory) : [];
    } catch (_error) {
      return [];
    }
  });
  const provider = LOCAL_PROVIDER;

  const refreshDashboard = async () => {
    const [customResponse, datasetResponse, trainingResponse, healthResponse] = await Promise.all([
      axios.get('/api/custom-models/status'),
      axios.get('/api/datasets/status'),
      axios.get('/api/training/status'),
      axios.get('/api/health'),
    ]);

    setCustomModels(customResponse.data.models || []);
    setCustomTraining(customResponse.data.training || null);
    setClipEnabled(Boolean(customResponse.data.clip_enabled));
    setDatasetStatus(datasetResponse.data || null);
    setTrainingStatus(trainingResponse.data.training || null);
    setHealthStatus(healthResponse.data || null);
  };

  useEffect(() => {
    let ignore = false;

    const loadDashboardState = async () => {
      try {
        const [customResponse, datasetResponse, trainingResponse, healthResponse] = await Promise.all([
          axios.get('/api/custom-models/status'),
          axios.get('/api/datasets/status'),
          axios.get('/api/training/status'),
          axios.get('/api/health'),
        ]);

        if (!ignore) {
          setCustomModels(customResponse.data.models || []);
          setCustomTraining(customResponse.data.training || null);
          setClipEnabled(Boolean(customResponse.data.clip_enabled));
          setDatasetStatus(datasetResponse.data || null);
          setTrainingStatus(trainingResponse.data.training || null);
          setHealthStatus(healthResponse.data || null);
        }
      } catch (_error) {
        if (!ignore) {
          setClipEnabled(false);
        }
      }
    };

    loadDashboardState();
    const intervalId = setInterval(loadDashboardState, 5000);

    return () => {
      ignore = true;
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    window.localStorage.setItem(
      DETECTION_HISTORY_STORAGE_KEY,
      JSON.stringify(detectionHistory.slice(0, 10)),
    );
  }, [detectionHistory]);

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

  const completedModels = useMemo(
    () => customModels.filter((model) => model.status === 'completed'),
    [customModels],
  );

  const datasetTotals = useMemo(() => {
    const splits = datasetStatus?.dataset?.splits || {};
    return Object.values(splits).reduce(
      (accumulator, split) => ({
        images: accumulator.images + (split.images || 0),
        labels: accumulator.labels + (split.labels || 0),
      }),
      { images: 0, labels: 0 },
    );
  }, [datasetStatus]);

  const activeSelectionCount = useMemo(() => {
    if (customSelectionMode === 'default') {
      return 0;
    }
    if (customSelectionMode === 'all') {
      return completedModels.length;
    }
    return selectedCustomModelIds.length;
  }, [completedModels.length, customSelectionMode, selectedCustomModelIds.length]);

  const trainingLifecycle = trainingStatus?.status || 'idle';
  const detectionTargetCount = healthStatus?.target_classes?.length || 0;
  const trainingTargets = datasetStatus?.target_classes?.length || 0;
  const datasetReady = Boolean(trainingStatus?.dataset_ready);
  const selectedModeLabel = customSelectionMode === 'default'
    ? 'Default detector'
    : `${customSelectionMode} custom mode`;

  const overviewCards = [
    {
      title: 'Detection Targets',
      value: detectionTargetCount,
      detail: `${trainingTargets} classes prepared for YOLO dataset training`,
      icon: Activity,
      accent: 'sky',
    },
    {
      title: 'Custom Model Library',
      value: completedModels.length,
      detail: `${customModels.length} total custom model profiles`,
      icon: Brain,
      accent: 'amber',
    },
    {
      title: 'Dataset Capacity',
      value: datasetTotals.images,
      detail: `${datasetTotals.labels} label files inside active training dataset`,
      icon: Database,
      accent: 'emerald',
    },
    {
      title: 'Current Inference Mode',
      value: activeSelectionCount,
      detail: `${selectedModeLabel} with ${activeSelectionCount} active custom model${activeSelectionCount === 1 ? '' : 's'}`,
      icon: Sparkles,
      accent: 'violet',
    },
  ];

  const handleImageSelect = async (file) => {
    if (!file) return;

    setIsLoading(true);
    setUploadProgress(0);
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
        onUploadProgress: (event) => {
          if (event.total) {
            setUploadProgress(Math.round((event.loaded / event.total) * 100));
          }
        },
      });

      if (response.data.success) {
        setResults(response.data);
        setDetectionHistory((previousHistory) => {
          const labels = (response.data.detections || []).map(
            (detection) => detection.resolved_class || detection.class,
          );
          const uniqueLabels = [...new Set(labels)];
          const now = new Date();
          const historyItem = {
            id: `${now.getTime()}-${Math.random().toString(16).slice(2, 8)}`,
            createdAt: now.toISOString(),
            timeLabel: now.toLocaleString(),
            providerLabel: response.data.provider_label || LOCAL_PROVIDER_LABEL,
            selectionMode: customSelectionMode,
            totalDetections: response.data.total_detections || 0,
            labels: uniqueLabels,
            imageUrl: response.data.image_url || '',
            summary: uniqueLabels.length > 0 ? uniqueLabels.join(', ') : 'No animals detected',
          };

          return [historyItem, ...previousHistory].slice(0, 10);
        });
      } else {
        setError(response.data.error || 'Detection failed');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Failed to connect to server';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  const clearHistory = () => {
    setDetectionHistory([]);
  };

  return (
    <div className="min-h-screen dashboard-shell">
      <Header
        detectionTargetCount={detectionTargetCount}
        readyModelCount={completedModels.length}
        datasetImageCount={datasetTotals.images}
        trainingStatus={trainingLifecycle}
      />

      <main className="max-w-7xl mx-auto px-4 pb-12 pt-8">
        <WorkflowSteps />

        {/* Top summary cards */}
        <section className="mt-6 grid auto-rows-fr gap-6 md:grid-cols-2 xl:grid-cols-4">
          {overviewCards.map((card) => {
            const Icon = card.icon;
            return (
              <article
                key={card.title}
                className={`overview-card overview-card-${card.accent} h-full rounded-2xl`}
              >
                <div className="overview-card-top">
                  <span className="overview-card-label">{card.title}</span>
                  <div className="overview-card-icon">
                    <Icon className="h-4 w-4" />
                  </div>
                </div>
                <p className="overview-card-value">{card.value}</p>
                <p className="overview-card-detail">{card.detail}</p>
              </article>
            );
          })}
        </section>

        {/* Main responsive dashboard: 2 columns main content, 1 column sticky sidebar */}
        <section className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {/* Main content */}
          <div className="space-y-6 lg:col-span-2 xl:col-span-2">
            {/* Upload section */}
            <section id="detect-workspace" className="workspace-panel rounded-2xl p-6 md:p-8">
              <div className="workspace-heading">
                <div>
                  <p className="workspace-kicker">Live Detection</p>
                  <h2 className="workspace-title">Upload And Inspect</h2>
                  <p className="workspace-subtitle">
                    Run the local detector, mix in custom animal models, and review the full inference output in one place.
                  </p>
                </div>
                <div className="surface-pill">
                  <ImageIcon className="h-4 w-4" />
                  {customSelectionLabel}
                </div>
              </div>

              <ImageUpload
                onImageSelect={handleImageSelect}
                isLoading={isLoading}
                uploadProgress={uploadProgress}
                providerLabel={LOCAL_PROVIDER_LABEL}
                customSelectionLabel={customSelectionLabel}
              />
            </section>

            {/* Prediction workspace */}
            <section className="workspace-panel rounded-2xl p-6 md:p-8">
              <div className="workspace-heading">
                <div>
                  <p className="workspace-kicker">Detection Results</p>
                  <h2 className="workspace-title">Prediction Workspace</h2>
                  <p className="workspace-subtitle">
                    Bounding-box previews, confidence scores, and custom model matches are shown here after each image run.
                  </p>
                </div>
                <div className={`surface-pill ${datasetReady ? 'surface-pill-success' : 'surface-pill-warning'}`}>
                  <CheckCircle2 className="h-4 w-4" />
                  {datasetReady ? 'Dataset ready' : 'Dataset incomplete'}
                </div>
              </div>

              <div className="min-h-[440px]">
                {(results || error) ? (
                  <DetectionResults results={results} error={error} provider={provider} />
                ) : (
                  <div className="result-placeholder min-h-[440px] rounded-2xl">
                    <ImageIcon className="h-10 w-10 text-slate-500" />
                    <h3 className="text-lg font-semibold text-slate-200">Results will appear here</h3>
                    <p className="max-w-xl text-center text-sm leading-6 text-slate-400">
                      Upload an image to generate bounding boxes, confidence scores, and any matching custom-animal signals from your selected model mode.
                    </p>
                  </div>
                )}
              </div>
            </section>

            {/* Full-width custom model management */}
            <section className="space-y-4">
              <div className="section-header mb-0">
                <div>
                  <p className="workspace-kicker">Custom Recognition</p>
                  <h2 className="workspace-title">Animal Model Library</h2>
                  <p className="workspace-subtitle">
                    Train and manage custom animal recognizers in a full-width workspace instead of a cramped sidebar.
                  </p>
                </div>
              </div>

              <CustomModelManager
                models={customModels}
                training={customTraining}
                clipEnabled={clipEnabled}
                selectionMode={customSelectionMode}
                selectedModelIds={selectedCustomModelIds}
                onSelectionModeChange={setCustomSelectionMode}
                onSelectedModelIdsChange={setSelectedCustomModelIds}
                onRefresh={refreshDashboard}
              />
            </section>
          </div>

          {/* Sticky sidebar */}
          <aside className="space-y-6 xl:sticky xl:top-6 self-start">
            <section className="workspace-panel workspace-rail rounded-2xl p-6">
              <div className="workspace-heading">
                <div>
                  <p className="workspace-kicker">Operations Summary</p>
                  <h2 className="workspace-title">Runtime Overview</h2>
                </div>
              </div>

              <div className="space-y-3">
                <div className="rail-row">
                  <div className="rail-icon rail-icon-emerald">
                    <ShieldCheck className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="rail-label">Local inference stack</p>
                    <p className="rail-value">YOLO + CLIP on this machine</p>
                  </div>
                </div>
                <div className="rail-row">
                  <div className="rail-icon rail-icon-sky">
                    <Database className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="rail-label">Dataset state</p>
                    <p className="rail-value">
                      {datasetReady ? 'Ready for training' : 'Need train + val image/label pairs'}
                    </p>
                  </div>
                </div>
                <div className="rail-row">
                  <div className="rail-icon rail-icon-amber">
                    <Brain className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="rail-label">Custom model engine</p>
                    <p className="rail-value">
                      {clipEnabled ? 'Enabled and ready for embedding-based custom recognition' : 'CLIP unavailable on this machine'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="rail-divider" />

              <div className="grid gap-3">
                <div className="mini-status-card">
                  <span className="mini-status-label">Training lifecycle</span>
                  <span className="mini-status-value capitalize">{trainingLifecycle}</span>
                </div>
                <div className="mini-status-card">
                  <span className="mini-status-label">Selected mode</span>
                  <span className="mini-status-value capitalize">{customSelectionMode}</span>
                </div>
              </div>
            </section>

            <section className="space-y-4">
              <div className="section-header mb-0">
                <div>
                  <p className="workspace-kicker">Dataset Pipeline</p>
                  <h2 className="workspace-title">Upload And Train</h2>
                </div>
              </div>
              <DatasetUpload />
            </section>

            <DetectionHistory items={detectionHistory} onClear={clearHistory} compact />
          </aside>
        </section>
      </main>

      <footer className="mt-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          Custom animal training, dataset ingestion, and local detection are available from the same dashboard workflow.
        </div>
      </footer>
    </div>
  );
}

export default App;
