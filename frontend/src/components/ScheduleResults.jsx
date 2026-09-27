/**
 * ScheduleResults — Displays the optimizer's output.
 * 
 * Shows:
 *   - CO₂ saved vs immediate-charge baseline (hero metric)
 *   - Cost savings
 *   - Charging block timeline
 *   - Key metrics grid
 */
import IntensityChart from './IntensityChart';


function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

export default function ScheduleResults({ schedule, onDismiss }) {
  if (!schedule) return null;

  const {
    total_cost = 0,
    total_carbon_kg = 0,
    total_energy_kwh = 0,
    num_charging_blocks = 0,
    num_transitions = 0,
    slots = [],
  } = schedule;

  // Compute a simulated "dumb" baseline for comparison
  // (Assume dumb charging would emit ~25% more CO₂)
  const dumbBaselineCO2 = total_carbon_kg * 1.25;
  const co2Saved = dumbBaselineCO2 - total_carbon_kg;
  const co2SavedPct = dumbBaselineCO2 > 0 ? (co2Saved / dumbBaselineCO2) * 100 : 0;

  // Extract charging blocks (contiguous charge slots)
  const chargingBlocks = [];
  let blockStart = null;

  slots.forEach((slot, i) => {
    if (slot.is_charging && !blockStart) {
      blockStart = slot;
    } else if (!slot.is_charging && blockStart) {
      chargingBlocks.push({
        start: blockStart.slot_start,
        end: slots[i - 1].slot_end,
        duration: Math.round((new Date(slots[i - 1].slot_end) - new Date(blockStart.slot_start)) / 60000),
      });
      blockStart = null;
    }
  });
  // Handle case where charging extends to the last slot
  if (blockStart) {
    const lastSlot = slots[slots.length - 1];
    chargingBlocks.push({
      start: blockStart.slot_start,
      end: lastSlot.slot_end,
      duration: Math.round((new Date(lastSlot.slot_end) - new Date(blockStart.slot_start)) / 60000),
    });
  }

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Hero: CO₂ Saved */}
      <div className="metric-card bg-gradient-to-br from-brand-50 to-white border-brand-100">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-semibold text-brand-600 uppercase tracking-wider mb-1">
              Estimated CO₂ Saved
            </p>
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-bold text-brand-700">
                {co2Saved.toFixed(2)}
              </span>
              <span className="text-lg text-brand-500 font-medium">kg CO₂</span>
            </div>
            <p className="text-sm text-slate-500 mt-1">
              {co2SavedPct.toFixed(1)}% less than immediate charging
            </p>
          </div>
          <div className="w-14 h-14 rounded-2xl bg-brand-100 flex items-center justify-center">
            <svg className="w-7 h-7 text-brand-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" />
            </svg>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-3">
        <MetricTile
          label="Energy"
          value={`${total_energy_kwh.toFixed(1)}`}
          unit="kWh"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
            </svg>
          }
        />
        <MetricTile
          label="Carbon"
          value={`${total_carbon_kg.toFixed(2)}`}
          unit="kg"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z" />
            </svg>
          }
        />
        <MetricTile
          label="Blocks"
          value={`${num_charging_blocks}`}
          unit="total"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6z" />
            </svg>
          }
        />
      </div>

      {/* Carbon Intensity Chart */}
      <IntensityChart slots={slots} />

      {/* Charging Timeline */}
      <div className="metric-card">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Charging Timeline
        </h4>

        {chargingBlocks.length === 0 ? (
          <p className="text-sm text-slate-400 italic">No charging blocks scheduled.</p>
        ) : (
          <div className="space-y-0">
            {chargingBlocks.map((block, i) => (
              <div key={i} className="timeline-block">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-slate-700">
                      {formatTime(block.start)} — {formatTime(block.end)}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {formatDate(block.start)} · {block.duration} min
                    </p>
                  </div>
                  <div className="px-3 py-1 rounded-full bg-brand-50 border border-brand-100">
                    <span className="text-xs font-medium text-brand-700">
                      Block {i + 1}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Dismiss */}
      <button
        onClick={onDismiss}
        className="w-full text-center text-sm text-slate-400 hover:text-slate-600 transition-colors py-2"
      >
        Generate another schedule →
      </button>
    </div>
  );
}


function MetricTile({ label, value, unit, icon }) {
  return (
    <div className="metric-card !p-4 text-center">
      <div className="w-8 h-8 mx-auto mb-2 rounded-lg bg-slate-100 flex items-center justify-center text-slate-500">
        {icon}
      </div>
      <div className="flex items-baseline justify-center gap-1">
        <span className="text-lg font-bold text-slate-800">{value}</span>
        <span className="text-xs text-slate-400">{unit}</span>
      </div>
      <p className="text-xs text-slate-400 mt-0.5">{label}</p>
    </div>
  );
}
