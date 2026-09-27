/**
 * App.jsx — Carbon-Aware EV Charging Dashboard
 * 
 * Main application component that orchestrates:
 *   1. Vehicle selection (via Smartcar)
 *   2. Charging constraint input
 *   3. Schedule generation (Celery task)
 *   4. Results display
 */

import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import VehicleSelector from './components/VehicleSelector';
import ChargingForm from './components/ChargingForm';
import ScheduleResults from './components/ScheduleResults';
import LoadingOverlay from './components/LoadingOverlay';
import {
  fetchVehicles,
  generateSchedule,
  pollTaskStatus,
  fetchSchedule,
} from './services/api';


// ── Demo / Mock Data ───────────────────────────────────────────────────────
// Used when the backend is not running, so the UI is always demonstrable.

const MOCK_VEHICLES = [
  {
    id: 'demo-vehicle-001',
    make: 'Tesla',
    model: 'Model 3',
    year: 2024,
    battery_level: 42,
    battery_capacity_kwh: 60,
    is_charging: false,
  },
  {
    id: 'demo-vehicle-002',
    make: 'Chevrolet',
    model: 'Bolt EUV',
    year: 2023,
    battery_level: 67,
    battery_capacity_kwh: 65,
    is_charging: true,
  },
  {
    id: 'demo-vehicle-003',
    make: 'Hyundai',
    model: 'Ioniq 5',
    year: 2023,
    battery_level: 25,
    battery_capacity_kwh: 77.4,
    is_charging: false,
  },
  {
    id: 'demo-vehicle-004',
    make: 'Ford',
    model: 'Mustang Mach-E',
    year: 2022,
    battery_level: 85,
    battery_capacity_kwh: 91,
    is_charging: false,
  },
  {
    id: 'demo-vehicle-005',
    make: 'Rivian',
    model: 'R1T',
    year: 2024,
    battery_level: 15,
    battery_capacity_kwh: 135,
    is_charging: false,
  },
];

function generateMockSchedule(departureTimeStr) {
  const departure = new Date(departureTimeStr).getTime();
  const now = Date.now();
  
  if (departure <= now) {
    throw new Error('Departure time must be in the future.');
  }

  const durationMs = departure - now;
  const numSlots = Math.floor(durationMs / (15 * 60 * 1000));
  
  // We need to charge 2 blocks of 8 and 6 slots, but bounded by numSlots
  const maxChargeSlots = Math.min(numSlots, 14);
  const block1Slots = Math.min(maxChargeSlots, 8);
  const block2Slots = maxChargeSlots - block1Slots;

  const slots = [];
  
  // Generate block 1 at beginning
  for (let i = 0; i < block1Slots; i++) {
    slots.push({
      slot_start: new Date(now + i * 900000).toISOString(),
      slot_end: new Date(now + (i + 1) * 900000).toISOString(),
      is_charging: true,
      moer_value: 180 + Math.random() * 40,
      price_value: 0.10,
      slot_cost: 0.8 + Math.random() * 0.3,
    });
  }

  // Generate idle slots for the rest, except the very end for block 2
  const idleSlots = numSlots - block1Slots - block2Slots;
  for (let i = 0; i < idleSlots; i++) {
    slots.push({
      slot_start: new Date(now + (block1Slots + i) * 900000).toISOString(),
      slot_end: new Date(now + (block1Slots + i + 1) * 900000).toISOString(),
      is_charging: false,
      moer_value: 650 + Math.random() * 100,
      price_value: 0.22,
      slot_cost: 0,
    });
  }

  // Generate block 2 at the very end
  for (let i = 0; i < block2Slots; i++) {
    slots.push({
      slot_start: new Date(now + (block1Slots + idleSlots + i) * 900000).toISOString(),
      slot_end: new Date(now + (block1Slots + idleSlots + i + 1) * 900000).toISOString(),
      is_charging: true,
      moer_value: 200 + Math.random() * 50,
      price_value: 0.10,
      slot_cost: 0.7 + Math.random() * 0.2,
    });
  }

  return {
    session_id: 'demo-session',
    status: 'ready',
    total_cost: 8.4321,
    total_carbon_kg: 2.1847,
    total_energy_kwh: (block1Slots + block2Slots) * (7.2 * 0.25),
    num_charging_blocks: block2Slots > 0 ? 2 : 1,
    num_transitions: block2Slots > 0 ? 3 : 1,
    slots,
  };
}


// ── App State Machine ──────────────────────────────────────────────────────

const VIEW = {
  FORM: 'FORM',
  LOADING: 'LOADING',
  RESULTS: 'RESULTS',
};

export default function App() {
  // Vehicle state
  const [vehicles, setVehicles] = useState([]);
  const [selectedVehicleId, setSelectedVehicleId] = useState(null);
  const [vehiclesLoading, setVehiclesLoading] = useState(true);
  const [isDemoMode, setIsDemoMode] = useState(false);

  // View state
  const [view, setView] = useState(VIEW.FORM);
  const [schedule, setSchedule] = useState(null);
  const [error, setError] = useState(null);

  // ── Fetch vehicles on mount ────────────────────────────────────────────

  useEffect(() => {
    async function loadVehicles() {
      try {
        const data = await fetchVehicles();
        if (data && data.vehicles) {
          setVehicles(data.vehicles);
          setIsDemoMode(data.demo_mode);
          if (data.vehicles.length > 0) {
            setSelectedVehicleId(data.vehicles[0].id);
          }
        } else if (Array.isArray(data)) {
          setVehicles(data);
          if (data.length > 0) {
            setSelectedVehicleId(data[0].id);
          }
        }
      } catch (err) {
        console.warn('Backend unavailable — using demo vehicles:', err.message);
        setVehicles(MOCK_VEHICLES);
        setSelectedVehicleId(MOCK_VEHICLES[0].id);
        setIsDemoMode(true);
      } finally {
        setVehiclesLoading(false);
      }
    }
    loadVehicles();
  }, []);

  // ── Schedule Generation Handler ────────────────────────────────────────

  const handleGenerateSchedule = useCallback(async ({ departureTime, targetLimit, chargerPower }) => {
    setError(null);
    setView(VIEW.LOADING);

    try {
      // 1. Submit to backend
      const { task_id, session_id } = await generateSchedule(
        selectedVehicleId,
        departureTime,
        targetLimit,
        chargerPower,
      );

      // 2. Poll for completion
      await pollTaskStatus(task_id);

      // 3. Fetch the full schedule
      const result = await fetchSchedule(session_id);
      setSchedule(result);
      setView(VIEW.RESULTS);

    } catch (err) {
      console.error('Schedule generation failed:', err.message);
      let errorMsg = err.message;
      if (err.message.includes('Infeasible')) {
        errorMsg = 'Not enough time to reach the target charge limit before departure.';
      }
      setError(errorMsg);
      setView(VIEW.FORM);
    }
  }, [selectedVehicleId]);

  // ── Reset Handler ──────────────────────────────────────────────────────

  const handleDismiss = () => {
    setSchedule(null);
    setView(VIEW.FORM);
  };

  // ── Render ─────────────────────────────────────────────────────────────

  const isConnected = vehicles.length > 0;

  return (
    <div className="min-h-screen flex flex-col">
      <Header isConnected={isConnected} isDemoMode={isDemoMode} />

      <main className="flex-1 max-w-xl mx-auto w-full px-4 py-8 space-y-6">
        {/* Error banner */}
        {error && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-sm text-red-700 animate-fade-in">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Vehicle selector — always visible */}
        {vehiclesLoading ? (
          <div className="metric-card space-y-3">
            <div className="shimmer h-4 w-24" />
            <div className="shimmer h-12 w-full" />
            <div className="shimmer h-20 w-full" />
          </div>
        ) : (
          <VehicleSelector
            vehicles={vehicles}
            selectedId={selectedVehicleId}
            onSelect={setSelectedVehicleId}
          />
        )}

        {/* Main content: form / loading / results */}
        {view === VIEW.FORM && (
          <ChargingForm
            onSubmit={handleGenerateSchedule}
            isLoading={false}
            disabled={!isConnected}
          />
        )}

        {view === VIEW.LOADING && <LoadingOverlay />}

        {view === VIEW.RESULTS && (
          <ScheduleResults
            schedule={schedule}
            onDismiss={handleDismiss}
          />
        )}

        {/* Footer */}
        <footer className="text-center pt-8 pb-4">
          <p className="text-xs text-slate-400">
            Carbon-Aware EV Charging · Research Prototype
          </p>
          <p className="text-xs text-slate-300 mt-1">
            Powered by WattTime MOER · Smartcar API · Dynamic Programming
          </p>
        </footer>
      </main>
    </div>
  );
}
