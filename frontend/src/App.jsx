import { useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DetectionResults from './components/DetectionResults';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = async (file) => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('image', file);

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
            />
          </section>

          {/* Results Section */}
          {(results || error) && (
            <section>
              <h2 className="text-lg font-semibold text-slate-200 mb-4">
                Results
              </h2>
              <DetectionResults results={results} error={error} />
            </section>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700/50 py-6 mt-12">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-slate-500 text-sm">
            Animal Detection using YOLOv8 - Built with React + Flask
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
