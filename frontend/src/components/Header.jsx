import { PawPrint } from 'lucide-react';

function Header() {
  return (
    <header className="py-6 px-4 border-b border-slate-700/50">
      <div className="max-w-6xl mx-auto flex items-center justify-center gap-3">
        <div className="p-2 bg-gradient-to-br from-green-500 to-blue-500 rounded-xl">
          <PawPrint className="w-8 h-8 text-white" />
        </div>
        <div className="text-center">
          <h1 className="text-2xl md:text-3xl font-bold">
            <span className="gradient-text">Animal Detection</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">Powered by YOLOv8</p>
        </div>
      </div>
    </header>
  );
}

export default Header;