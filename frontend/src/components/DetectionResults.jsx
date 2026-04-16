import { CheckCircle, AlertCircle, Box, ExternalLink, Server } from 'lucide-react';

function DetectionResults({ results, error, provider }) {
  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 flex items-center gap-4">
        <AlertCircle className="w-8 h-8 text-red-500 flex-shrink-0" />
        <div>
          <h3 className="text-red-400 font-semibold">Error</h3>
          <p className="text-slate-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  const {
    detections,
    image_url,
    total_detections,
    provider_label,
    provider_mode,
    message,
    external_url,
    custom_models_used,
    custom_selection_mode,
  } = results;

  return (
    <div className="space-y-6">
      {/* Image with bounding boxes */}
      <div className="bg-slate-800 rounded-xl overflow-hidden border border-slate-700">
        <div className="p-4 border-b border-slate-700 flex items-center justify-between">
          <h3 className="font-semibold text-slate-200 flex items-center gap-2">
            <Box className="w-5 h-5 text-green-500" />
            Detection Result
          </h3>
          <div className="text-right">
            <div className="text-sm text-slate-400">
              {total_detections} animal{total_detections !== 1 ? 's' : ''} found
            </div>
            <div className="text-xs text-slate-500 flex items-center justify-end gap-1 mt-1">
              <Server className="w-3.5 h-3.5" />
              {provider_label || provider}
              {provider_mode ? ` - ${provider_mode}` : ''}
            </div>
          </div>
        </div>
        <div className="p-4">
          <img
            src={image_url}
            alt="Detection result"
            className="w-full h-auto rounded-lg"
          />
        </div>
      </div>

      {custom_selection_mode && custom_selection_mode !== 'default' && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
          <p className="text-amber-300 font-medium">Custom model mode</p>
          <p className="text-slate-300 text-sm mt-1 capitalize">
            {custom_selection_mode}
          </p>
          <p className="text-slate-400 text-sm mt-2">
            {custom_models_used?.length > 0
              ? custom_models_used.map((model) => model.name).join(', ')
              : 'No completed custom models were available for this request.'}
          </p>
        </div>
      )}

      {message && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-blue-300 font-medium">Provider message</p>
              <p className="text-slate-300 text-sm mt-1">{message}</p>
            </div>
            {external_url && (
              <a
                href={external_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 text-sm text-blue-300 hover:text-blue-200 whitespace-nowrap"
              >
                Open
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
          </div>
        </div>
      )}

      {/* Detection cards */}
      {detections && detections.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {detections.map((detection, index) => (
            <div
              key={index}
              className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-green-500/50 transition-colors animate-fadeIn"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="font-medium text-slate-200 capitalize">
                    {detection.resolved_class || detection.class}
                  </span>
                </div>
                <span className="text-sm text-slate-400">
                  {((detection.resolved_confidence ?? detection.confidence) * 100).toFixed(1)}%
                </span>
              </div>

              {detection.custom_match?.accepted && (
                <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2">
                  <p className="text-amber-300 text-sm font-medium">
                    Custom model matched: {detection.custom_match.name}
                  </p>
                  <p className="text-slate-400 text-xs mt-1">
                    Base detector: {detection.class} ({(detection.confidence * 100).toFixed(1)}%)
                  </p>
                </div>
              )}
              
              {/* Confidence bar */}
              <div className="h-2 bg-slate-700 rounded-full overflow-hidden mb-3">
                <div
                  className="h-full progress-bar rounded-full transition-all duration-500"
                  style={{ width: `${(detection.resolved_confidence ?? detection.confidence) * 100}%` }}
                />
              </div>

              {detection.custom_matches?.length > 0 && (
                <div className="mb-3 text-xs text-slate-400 space-y-1">
                  <p className="text-slate-300">Custom matches:</p>
                  {detection.custom_matches.slice(0, 3).map((match) => (
                    <p key={match.id}>
                      {match.name}: {(match.score * 100).toFixed(1)}%
                      {match.accepted ? ' accepted' : ''}
                    </p>
                  ))}
                </div>
              )}
              
              {/* Bounding box coordinates */}
              <div className="text-xs text-slate-500">
                <span className="text-slate-400">BBox: </span>
                [
                {detection.bbox.map((coord, i) => (
                  <span key={i}>
                    {coord.toFixed(0)}
                    {i < 3 ? ', ' : ''}
                  </span>
                ))}
                ]
              </div>
            </div>
          ))}
        </div>
      )}

      {detections && detections.length === 0 && (
        <div className="bg-slate-800/50 rounded-xl p-8 text-center border border-slate-700">
          <AlertCircle className="w-12 h-12 text-slate-500 mx-auto mb-3" />
          <p className="text-slate-400">No animals detected in this image</p>
          <p className="text-slate-500 text-sm mt-1">
            Try uploading an image with visible animals
          </p>
        </div>
      )}
    </div>
  );
}

export default DetectionResults;
