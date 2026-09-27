/**
 * ChargingForm — User inputs for charging constraints.
 * 
 * Inputs:
 *   - Target Departure Time (datetime-local picker)
 *   - Target Charge Limit % (range slider, 50–100%)
 *   - Charger Power kW (select dropdown)
 */
import { useState, useMemo } from 'react';

const CHARGER_OPTIONS = [
  { value: 3.6,  label: '3.6 kW — Level 1' },
  { value: 7.2,  label: '7.2 kW — Level 2' },
  { value: 11,   label: '11 kW — Level 2 (High)' },
  { value: 22,   label: '22 kW — Level 2 (Max)' },
];

function getDefaultDeparture() {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  tomorrow.setHours(8, 0, 0, 0);
  // Format for datetime-local input
  const pad = (n) => String(n).padStart(2, '0');
  return `${tomorrow.getFullYear()}-${pad(tomorrow.getMonth() + 1)}-${pad(tomorrow.getDate())}T${pad(tomorrow.getHours())}:${pad(tomorrow.getMinutes())}`;
}

export default function ChargingForm({ onSubmit, isLoading, disabled }) {
  const [departureTime, setDepartureTime] = useState(getDefaultDeparture);
  const [targetLimit, setTargetLimit] = useState(80);
  const [chargerPower, setChargerPower] = useState(7.2);

  // Compute minimum departure (now + 1 hour)
  const minDeparture = useMemo(() => {
    const min = new Date();
    min.setHours(min.getHours() + 1);
    const pad = (n) => String(n).padStart(2, '0');
    return `${min.getFullYear()}-${pad(min.getMonth() + 1)}-${pad(min.getDate())}T${pad(min.getHours())}:${pad(min.getMinutes())}`;
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      departureTime: new Date(departureTime).toISOString(),
      targetLimit,
      chargerPower,
    });
  };

  // Slider gradient: fill up to the current value
  const sliderPercent = ((targetLimit - 50) / 50) * 100;

  return (
    <form onSubmit={handleSubmit} className="metric-card space-y-6">
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
        Charging Constraints
      </h3>

      {/* Departure Time */}
      <div>
        <label htmlFor="departure-time" className="block text-sm font-medium text-slate-700 mb-2">
          Target Departure Time
        </label>
        <input
          id="departure-time"
          type="datetime-local"
          value={departureTime}
          min={minDeparture}
          onChange={(e) => setDepartureTime(e.target.value)}
          className="input-field"
          required
        />
        <p className="mt-1.5 text-xs text-slate-400">
          When do you need your vehicle ready?
        </p>
      </div>

      {/* Target Charge Limit Slider */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label htmlFor="target-limit" className="text-sm font-medium text-slate-700">
            Target Charge Limit
          </label>
          <div className="flex items-baseline gap-1">
            <span className="text-xl font-bold text-brand-600">{targetLimit}</span>
            <span className="text-sm text-slate-500">%</span>
          </div>
        </div>
        
        <div className="relative">
          <input
            id="target-limit"
            type="range"
            min={50}
            max={100}
            step={5}
            value={targetLimit}
            onChange={(e) => setTargetLimit(Number(e.target.value))}
            className="w-full h-2 rounded-full appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-brand-500
                       [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-brand-500/30
                       [&::-webkit-slider-thumb]:cursor-pointer
                       [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white
                       [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150
                       [&::-webkit-slider-thumb]:hover:scale-110"
            style={{
              background: `linear-gradient(to right, #22c55e ${sliderPercent}%, #e2e8f0 ${sliderPercent}%)`,
            }}
          />
          <div className="flex justify-between mt-1 text-xs text-slate-400">
            <span>50%</span>
            <span>75%</span>
            <span>100%</span>
          </div>
        </div>
        <p className="mt-1.5 text-xs text-slate-400">
          80% recommended for lithium-ion longevity
        </p>
      </div>

      {/* Charger Power */}
      <div>
        <label htmlFor="charger-power" className="block text-sm font-medium text-slate-700 mb-2">
          Charger Power
        </label>
        <select
          id="charger-power"
          value={chargerPower}
          onChange={(e) => setChargerPower(Number(e.target.value))}
          className="select-field"
        >
          {CHARGER_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading || disabled}
        className="btn-brand w-full flex items-center justify-center gap-2"
        id="generate-schedule-btn"
      >
        {isLoading ? (
          <>
            <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Optimizing Schedule…</span>
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Generate Smart Schedule</span>
          </>
        )}
      </button>
    </form>
  );
}
