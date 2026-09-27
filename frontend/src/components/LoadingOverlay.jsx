/**
 * LoadingOverlay — Full-page loading state during schedule optimization.
 * Shows an animated molecule/grid icon with progress messaging.
 */

const LOADING_MESSAGES = [
  'Fetching real-time grid carbon intensity…',
  'Analyzing marginal emissions data…',
  'Running dynamic programming optimizer…',
  'Evaluating battery degradation penalties…',
  'Minimizing multi-objective cost function…',
  'Building optimal charging blocks…',
];

import { useState, useEffect } from 'react';

export default function LoadingOverlay() {
  const [messageIdx, setMessageIdx] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setMessageIdx((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="metric-card py-12 text-center animate-fade-in">
      {/* Animated rings */}
      <div className="relative w-20 h-20 mx-auto mb-6">
        <div className="absolute inset-0 rounded-full border-4 border-brand-100 animate-ping opacity-20" />
        <div className="absolute inset-2 rounded-full border-4 border-brand-200 animate-ping opacity-30" style={{ animationDelay: '0.3s' }} />
        <div className="absolute inset-4 rounded-full border-4 border-brand-300 animate-pulse" />
        <div className="absolute inset-0 flex items-center justify-center">
          <svg className="w-8 h-8 text-brand-600 animate-spin" style={{ animationDuration: '3s' }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      </div>

      <h3 className="text-lg font-semibold text-slate-700 mb-2">
        Optimizing Your Schedule
      </h3>

      <p className="text-sm text-slate-500 h-5 transition-opacity duration-300" key={messageIdx}>
        {LOADING_MESSAGES[messageIdx]}
      </p>

      {/* Progress bar */}
      <div className="mt-6 mx-auto max-w-xs">
        <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600 animate-[shimmer_2s_ease-in-out_infinite]"
               style={{ width: '60%', animation: 'progress 3s ease-in-out infinite' }} />
        </div>
      </div>

      <style>{`
        @keyframes progress {
          0%   { width: 10%; margin-left: 0; }
          50%  { width: 60%; margin-left: 20%; }
          100% { width: 10%; margin-left: 90%; }
        }
      `}</style>
    </div>
  );
}
