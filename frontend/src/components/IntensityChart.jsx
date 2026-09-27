import React from 'react';

/**
 * IntensityChart — Renders a bar chart showing the Carbon Intensity (MOER) over time
 * for the generated schedule slots.
 * 
 * @param {Object} props
 * @param {Array} props.slots - The array of scheduled slots from the backend/mock
 */
export default function IntensityChart({ slots }) {
  if (!slots || slots.length === 0) return null;

  // Find min and max MOER for scaling the bars
  const maxMoer = Math.max(...slots.map(s => s.moer_value || 0));
  const minMoer = Math.min(...slots.map(s => s.moer_value || 0));
  const range = maxMoer - minMoer || 1; // avoid div by 0

  return (
    <div className="metric-card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-700">Grid Carbon Intensity Forecast</h3>
        <span className="text-xs text-slate-500 font-medium">lbs CO₂ / MWh</span>
      </div>
      
      <div className="relative h-32 flex items-end gap-1 w-full mt-2">
        {slots.map((slot, idx) => {
          const val = slot.moer_value || 0;
          // Scale height between 20% and 100% based on value
          const heightPercent = 20 + ((val - minMoer) / range) * 80;
          
          // Color coding: green for low carbon, yellow for medium, red for high
          let colorClass = "bg-green-400";
          if (val > minMoer + range * 0.6) colorClass = "bg-red-400";
          else if (val > minMoer + range * 0.3) colorClass = "bg-yellow-400";

          // If charging, make the bar solid. If idle, make it slightly faded/gray
          if (!slot.is_charging) {
            colorClass = "bg-slate-300";
          }

          const timeStr = new Date(slot.slot_start).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

          return (
            <div 
              key={idx} 
              className="group relative flex-1 flex flex-col justify-end items-center h-full"
            >
              <div 
                className={`w-full rounded-t-sm transition-all duration-300 ${colorClass} ${slot.is_charging ? 'opacity-100' : 'opacity-60'}`}
                style={{ height: `${heightPercent}%` }}
              ></div>
              
              {/* Tooltip */}
              <div className="absolute bottom-full mb-2 hidden group-hover:flex flex-col items-center z-10">
                <div className="bg-slate-800 text-white text-xs rounded px-2 py-1 whitespace-nowrap shadow-lg">
                  <div className="font-bold">{Math.round(val)} lbs</div>
                  <div className="text-slate-300 text-[10px]">{timeStr}</div>
                  <div className="text-[10px] mt-0.5 font-semibold text-brand-300">
                    {slot.is_charging ? 'CHARGING' : 'IDLE'}
                  </div>
                </div>
                <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[4px] border-t-slate-800"></div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex justify-between text-[10px] text-slate-400 font-medium px-1">
        <span>{new Date(slots[0].slot_start).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>
        <span>{new Date(slots[slots.length - 1].slot_end).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>
      </div>
    </div>
  );
}
