import { Clock3, History, Layers3, Trash2 } from 'lucide-react';

function DetectionHistory({ items, onClear, compact = false }) {
  return (
    <div id="history-section" className="dashboard-panel dashboard-panel-spacious space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="workspace-kicker">Detection Timeline</p>
          <h2 className="workspace-title">History</h2>
          <p className="workspace-subtitle">
            Recent detections are saved locally on this device so you can review previous runs quickly.
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          disabled={items.length === 0}
          className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-slate-950/50 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-red-400/30 hover:text-red-200 disabled:opacity-40"
        >
          <Trash2 className="h-4 w-4" />
          Clear
        </button>
      </div>

      {items.length === 0 ? (
        <div className={`result-placeholder ${compact ? 'min-h-[12rem]' : 'min-h-[16rem]'}`}>
          <History className="h-10 w-10 text-slate-500" />
          <h3 className="text-lg font-semibold text-slate-200">No detection history yet</h3>
          <p className="max-w-lg text-center text-sm leading-6 text-slate-400">
            Run a detection and the result summary will appear here with time, labels, and preview image.
          </p>
        </div>
      ) : (
        <div className={`grid gap-4 ${compact ? 'grid-cols-1' : 'lg:grid-cols-2'}`}>
          {items.map((item) => (
            <article
              key={item.id}
              className={`history-card ${compact ? 'history-card-compact' : ''}`}
            >
              <div className="history-preview-wrap">
                {item.imageUrl ? (
                  <img
                    src={item.imageUrl}
                    alt={item.summary}
                    className="history-preview"
                  />
                ) : (
                  <div className="history-preview history-preview-empty">
                    <History className="h-8 w-8 text-slate-500" />
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-base font-semibold text-slate-100 truncate">{item.summary}</p>
                    <p className="mt-1 text-sm text-slate-400 truncate">{item.providerLabel}</p>
                  </div>
                  <span className="rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium uppercase tracking-wide text-emerald-300">
                    {item.totalDetections} found
                  </span>
                </div>

                <div className={`grid gap-3 ${compact ? 'grid-cols-1' : 'grid-cols-2'}`}>
                  <div className="history-meta-card">
                    <Clock3 className="h-4 w-4 text-sky-300" />
                    <div>
                      <p className="history-meta-label">Time</p>
                      <p className="history-meta-value">{item.timeLabel}</p>
                    </div>
                  </div>
                  <div className="history-meta-card">
                    <Layers3 className="h-4 w-4 text-amber-300" />
                    <div>
                      <p className="history-meta-label">Mode</p>
                      <p className="history-meta-value capitalize">{item.selectionMode}</p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {item.labels.length > 0 ? (
                    item.labels.slice(0, 5).map((label) => (
                      <span key={`${item.id}-${label}`} className="history-chip">
                        {label}
                      </span>
                    ))
                  ) : (
                    <span className="history-chip history-chip-muted">No animals detected</span>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

export default DetectionHistory;
