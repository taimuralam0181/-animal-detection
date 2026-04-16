import { useState, useRef } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';

const IMAGE_EXTENSIONS = [
  '.avif',
  '.bmp',
  '.gif',
  '.heic',
  '.heif',
  '.jfif',
  '.jpeg',
  '.jpg',
  '.png',
  '.tif',
  '.tiff',
  '.webp',
];

function ImageUpload({
  onImageSelect,
  isLoading,
  uploadProgress,
  providerLabel,
  customSelectionLabel,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const isLikelyImageFile = (file) => {
    if (!file) {
      return false;
    }

    if (file.type?.startsWith('image/')) {
      return true;
    }

    const lowerName = file.name.toLowerCase();
    return IMAGE_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (isLikelyImageFile(file)) {
      handleFile(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (isLikelyImageFile(file)) {
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);
    onImageSelect(file);
  };

  const handleClear = () => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      {!preview ? (
        <div
          className={`upload-zone premium-upload-zone rounded-[24px] p-8 md:p-12 text-center cursor-pointer ${
            dragOver ? 'dragover' : ''
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.avif,.bmp,.gif,.heic,.heif,.jfif,.jpeg,.jpg,.png,.tif,.tiff,.webp"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="flex flex-col items-center gap-4">
            <div className="relative z-10 p-4 bg-slate-900/70 rounded-full border border-white/10 shadow-lg">
              <Upload className="w-8 h-8 text-green-500" />
            </div>
            <div className="relative z-10">
              <p className="text-xl font-semibold text-slate-100">
                Drop your image here
              </p>
              <p className="text-slate-300 text-sm mt-2 leading-6">
                Click or drag any supported image into this panel. The detector will run locally with your current model selection.
              </p>
              <p className="text-slate-400 text-xs mt-4 uppercase tracking-[0.2em]">
                Current source: {providerLabel}
              </p>
              <p className="text-slate-500 text-xs mt-2 uppercase tracking-[0.2em]">
                Custom models: {customSelectionLabel}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="dashboard-subpanel relative rounded-[24px] overflow-hidden border border-white/10">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-72 md:h-[25rem] object-contain bg-slate-950/40"
          />
          <button
            onClick={handleClear}
            disabled={isLoading}
            className="absolute top-3 right-3 p-2 bg-slate-950/80 hover:bg-slate-950 rounded-full transition-colors border border-white/10"
          >
            <X className="w-5 h-5 text-slate-300" />
          </button>
          {isLoading && (
            <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-10 h-10 text-green-500 animate-spin" />
                <p className="text-slate-300 font-medium">
                  Detecting with {providerLabel}...
                </p>
                <div className="w-64 max-w-[80vw]">
                  <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-slate-400">
                    <span>Upload progress</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full progress-bar transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
