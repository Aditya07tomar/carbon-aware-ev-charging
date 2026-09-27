/**
 * Header — Top navigation bar with brand identity.
 */
import { useState } from 'react';
import { connectSmartcar } from '../services/api';

export default function Header({ isConnected, isDemoMode }) {
  const [connecting, setConnecting] = useState(false);

  const handleConnect = () => {
    setConnecting(true);
    connectSmartcar();
  };

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-slate-200/50">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800 leading-tight">Carbon-Aware</h1>
            <p className="text-xs text-slate-500 font-medium -mt-0.5">EV Charging Optimizer</p>
          </div>
        </div>

        {/* Connection status + Smartcar button */}
        <div className="flex items-center gap-4">
          {isDemoMode && (
            <div 
              className="px-2 py-1 rounded-md text-xs font-semibold text-amber-700 bg-amber-100 border border-amber-200"
              title="Smartcar API keys are missing. Showing mock data."
            >
              Demo Mode
            </div>
          )}

          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-brand-500 animate-pulse-slow' : 'bg-slate-300'}`} />
            <span className="text-sm text-slate-500">
              {isConnected ? 'Vehicle Connected' : 'No Vehicle'}
            </span>
          </div>

          {isDemoMode && (
            <button
              onClick={handleConnect}
              disabled={connecting}
              className="px-4 py-2 rounded-lg text-sm font-medium text-brand-700 bg-brand-50 border border-brand-200 hover:bg-brand-100 transition-colors disabled:opacity-50"
            >
              {connecting ? 'Redirecting…' : 'Connect Vehicle'}
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
