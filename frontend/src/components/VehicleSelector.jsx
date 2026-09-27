/**
 * VehicleSelector — Dropdown to select the active EV.
 * Displays car model name + live battery percentage.
 */

export default function VehicleSelector({ vehicles, selectedId, onSelect }) {
  if (!vehicles || vehicles.length === 0) {
    return (
      <div className="metric-card text-center py-8">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h1.125c.621 0 1.129-.504 1.09-1.124a17.902 17.902 0 00-3.213-9.193 2.056 2.056 0 00-1.58-.86H14.25M16.5 18.75h-2.25m0-11.177v-.958c0-.568-.422-1.048-.987-1.106a48.554 48.554 0 00-10.026 0 1.106 1.106 0 00-.987 1.106v7.635m12-6.677v6.677m0 4.5v-4.5m0 0h-12" />
          </svg>
        </div>
        <p className="text-slate-500 text-sm">No vehicles found</p>
        <p className="text-slate-400 text-xs mt-1">Connect your EV via Smartcar to get started</p>
      </div>
    );
  }

  const selected = vehicles.find(v => v.id === selectedId) || vehicles[0];

  return (
    <div className="metric-card">
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
        Select Vehicle
      </label>

      <select
        value={selected.id}
        onChange={(e) => onSelect(e.target.value)}
        className="select-field text-base font-medium"
        id="vehicle-selector"
      >
        {vehicles.map((vehicle) => (
          <option key={vehicle.id} value={vehicle.id}>
            {vehicle.year} {vehicle.make} {vehicle.model}
          </option>
        ))}
      </select>

      {/* Battery Status Card */}
      <div className="mt-4 flex items-center gap-4 p-4 rounded-xl bg-slate-50 border border-slate-100">
        {/* Battery icon with fill level */}
        <div className="relative w-14 h-8 rounded-md border-2 border-slate-300 overflow-hidden">
          <div className="absolute right-[-4px] top-1/2 -translate-y-1/2 w-1.5 h-3 bg-slate-300 rounded-r-sm" />
          <div
            className={`absolute left-0 top-0 bottom-0 rounded-sm transition-all duration-700 ${
              selected.battery_level > 60 ? 'bg-brand-500' :
              selected.battery_level > 25 ? 'bg-amber-400' : 'bg-red-400'
            }`}
            style={{ width: `${Math.min(100, selected.battery_level)}%` }}
          />
        </div>

        <div className="flex-1">
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-slate-800">{selected.battery_level}</span>
            <span className="text-sm font-medium text-slate-500">%</span>
          </div>
          <p className="text-xs text-slate-400">
            {selected.battery_capacity_kwh ? `${(selected.battery_capacity_kwh * selected.battery_level / 100).toFixed(1)} / ${selected.battery_capacity_kwh} kWh` : 'Battery Level'}
          </p>
        </div>

        {/* Charging indicator */}
        {selected.is_charging && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-brand-50 border border-brand-200">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
            <span className="text-xs font-medium text-brand-700">Charging</span>
          </div>
        )}
      </div>
    </div>
  );
}
