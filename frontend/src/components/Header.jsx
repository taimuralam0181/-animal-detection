import { Activity, ArrowRight, Brain, Database, History, PawPrint, Sparkles } from 'lucide-react';

function Header({
  detectionTargetCount,
  readyModelCount,
  datasetImageCount,
  trainingStatus,
}) {
  return (
    <header className="px-4 pt-6 md:pt-8">
      <div className="max-w-7xl mx-auto">
        <div className="hero-surface overflow-hidden rounded-[28px] border border-white/10 px-6 py-8 md:px-8 md:py-10">
          <div className="flex flex-col gap-8 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-emerald-200">
                <Sparkles className="h-3.5 w-3.5" />
                Premium Animal Dashboard
              </div>
              <div className="mt-5 flex items-center gap-4">
                <div className="hero-icon-shell">
                  <PawPrint className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight text-white md:text-5xl">
                    Animal Detection Control Center
                  </h1>
                  <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300 md:text-base">
                    Detect animals, train custom recognizers, manage datasets, and monitor the full workflow from one dashboard.
                  </p>
                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <a href="#detect-workspace" className="cta-primary">
                      Start Detecting
                      <ArrowRight className="h-4 w-4" />
                    </a>
                    <a href="#history-section" className="cta-secondary">
                      <History className="h-4 w-4" />
                      View History
                    </a>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:min-w-[420px]">
              <div className="hero-mini-card">
                <Activity className="h-4 w-4 text-sky-300" />
                <p className="hero-mini-value">{detectionTargetCount}</p>
                <p className="hero-mini-label">Detect Targets</p>
              </div>
              <div className="hero-mini-card">
                <Brain className="h-4 w-4 text-amber-300" />
                <p className="hero-mini-value">{readyModelCount}</p>
                <p className="hero-mini-label">Ready Models</p>
              </div>
              <div className="hero-mini-card">
                <Database className="h-4 w-4 text-emerald-300" />
                <p className="hero-mini-value">{datasetImageCount}</p>
                <p className="hero-mini-label">Dataset Images</p>
              </div>
              <div className="hero-mini-card">
                <Sparkles className="h-4 w-4 text-fuchsia-300" />
                <p className="hero-mini-value capitalize">{trainingStatus}</p>
                <p className="hero-mini-label">Training Status</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
