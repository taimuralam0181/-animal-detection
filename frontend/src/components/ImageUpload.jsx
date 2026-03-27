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

function ImageUpload({ onImageSelect, isLoading }) {
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
          className={`upload-zone rounded-xl p-8 md:p-12 text-center cursor-pointer ${
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
            <div className="p-4 bg-slate-800 rounded-full">
              <Upload className="w-8 h-8 text-green-500" />
            </div>
            <div>
              <p className="text-lg font-medium text-slate-200">
                Drop your image here
              </p>
              <p className="text-slate-400 text-sm mt-1">
                or click to browse - most image formats are accepted
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl overflow-hidden bg-slate-800 border border-slate-700">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-64 md:h-80 object-contain"
          />
          <button
            onClick={handleClear}
            disabled={isLoading}
            className="absolute top-3 right-3 p-2 bg-slate-900/80 hover:bg-slate-900 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-slate-300" />
          </button>
          {isLoading && (
            <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-10 h-10 text-green-500 animate-spin" />
                <p className="text-slate-300 font-medium">Detecting animals...</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
