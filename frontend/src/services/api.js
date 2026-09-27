/**
 * API Service — Carbon-Aware EV Charging Frontend
 * 
 * Axios client configured for the FastAPI backend.
 * All API calls route through the Vite dev proxy (/api → localhost:8000).
 */

import axios from 'axios';

// ── Axios Instance ─────────────────────────────────────────────────────────

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Request Interceptor: attach auth token if present ──────────────────────

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response Interceptor: normalize errors ─────────────────────────────────

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail 
      || error.response?.data?.message 
      || error.message 
      || 'An unexpected error occurred';
    
    console.error('[API Error]', {
      url: error.config?.url,
      status: error.response?.status,
      message,
    });
    
    return Promise.reject(new Error(message));
  }
);


// ═══════════════════════════════════════════════════════════════════════════
//  Auth & Smartcar
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Redirect the user to Smartcar OAuth flow via the backend.
 * The backend handles the OAuth redirect URL construction.
 */
export function connectSmartcar() {
  window.location.href = `${api.defaults.baseURL}/api/v1/auth/smartcar`;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Vehicles
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Fetch the list of authenticated vehicles with live battery SoC.
 * 
 * @returns {Promise<{demo_mode: boolean, vehicles: Array}>}
 */
export async function fetchVehicles() {
  const { data } = await api.get('/api/v1/vehicles');
  return data;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Carbon Forecast
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Fetch 24-hour carbon intensity forecast.
 * 
 * @returns {Promise<{status: string, horizon_hours: number, forecast: Array, source: string}>}
 */
export async function fetchCarbonForecast() {
  const { data } = await api.get('/api/v1/carbon-forecast');
  return data;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Schedule Generation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Submit charging constraints and trigger the scheduling algorithm.
 * 
 * @param {string}  vehicleId      - Smartcar vehicle UUID
 * @param {string}  departureTime  - ISO 8601 departure timestamp
 * @param {number}  targetLimit    - Target charge % (50–100)
 * @param {number}  chargerPower   - Charger output in kW (3.6–22)
 * @param {Object}  [weights]      - Optional cost function weights
 * @param {number}  [weights.w_carbon=1.0]
 * @param {number}  [weights.w_price=0.5]  
 * @param {number}  [weights.w_degradation=2.0]
 * 
 * @returns {Promise<{ task_id: string, session_id: string, status: string }>}
 */
export async function generateSchedule(
  vehicleId, 
  departureTime, 
  targetLimit, 
  chargerPower,
  weights = {},
) {
  const { data } = await api.post('/schedule/generate', {
    vehicle_id: vehicleId,
    departure_time: departureTime,
    target_charge_percent: targetLimit,
    charger_power_kw: chargerPower,
    w_carbon: weights.w_carbon ?? 1.0,
    w_price: weights.w_price ?? 0.5,
    w_degradation: weights.w_degradation ?? 2.0,
  });
  return data;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Task Polling
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Poll a Celery task's status until completion.
 * 
 * @param {string}  taskId         - Celery task UUID
 * @param {number}  [interval=2000] - Polling interval in ms
 * @param {number}  [maxAttempts=60] - Max polling attempts
 * 
 * @returns {Promise<Object>} - Task result on success
 */
export async function pollTaskStatus(taskId, interval = 2000, maxAttempts = 60) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const { data } = await api.get(`/schedule/status/${taskId}`);
    
    if (data.status === 'SUCCESS') {
      return data.result;
    }
    
    if (data.status === 'FAILURE') {
      throw new Error(data.error || 'Schedule generation failed');
    }
    
    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
  
  throw new Error('Schedule generation timed out');
}


// ═══════════════════════════════════════════════════════════════════════════
//  Schedule Retrieval
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Retrieve the full generated schedule for a session.
 * 
 * @param {string} sessionId - Charging session UUID
 * @returns {Promise<Object>} - Complete schedule with slots and metrics
 */
export async function fetchSchedule(sessionId) {
  const { data } = await api.get(`/schedule/${sessionId}`);
  return data;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Health Check
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check backend health status.
 * 
 * @returns {Promise<Object>}
 */
export async function healthCheck() {
  const { data } = await api.get('/health');
  return data;
}


export default api;

