import { Brain, Database, SearchCheck, UploadCloud } from 'lucide-react';

const steps = [
  {
    id: '01',
    title: 'Upload Image',
    description: 'Drop an animal image and preview it before sending it to the detector.',
    icon: UploadCloud,
    accent: 'emerald',
  },
  {
    id: '02',
    title: 'Run Detection',
    description: 'The local YOLO + CLIP stack analyzes the image and scores the best matches.',
    icon: SearchCheck,
    accent: 'sky',
  },
  {
    id: '03',
    title: 'Review Results',
    description: 'Inspect bounding boxes, confidence, and custom-model overrides from one panel.',
    icon: Brain,
    accent: 'amber',
  },
  {
    id: '04',
    title: 'Improve Dataset',
    description: 'Upload training zips or custom-animal images to keep improving the system.',
    icon: Database,
    accent: 'violet',
  },
];

function WorkflowSteps() {
  return (
    <section className="dashboard-panel dashboard-panel-spacious space-y-5">
      <div>
        <p className="workspace-kicker">Workflow</p>
        <h2 className="workspace-title">Step-by-step Pipeline</h2>
        <p className="workspace-subtitle">
          The dashboard is organized around a simple cycle: detect first, then refine models and datasets when needed.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        {steps.map((step) => {
          const Icon = step.icon;
          return (
            <article key={step.id} className={`workflow-step workflow-step-${step.accent}`}>
              <div className="workflow-step-top">
                <span className="workflow-step-id">{step.id}</span>
                <div className="workflow-step-icon">
                  <Icon className="h-4 w-4" />
                </div>
              </div>
              <h3 className="workflow-step-title">{step.title}</h3>
              <p className="workflow-step-description">{step.description}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export default WorkflowSteps;
