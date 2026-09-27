"""
Benchmark Engine for Carbon-Aware EV Charging.

Compares our multi-objective DP scheduler against two baselines using
real-world EV charging sessions from the Caltech ACN-Data open dataset
(https://ev.caltech.edu/dataset).

Data source:
    The Adaptive Charging Network (ACN) at Caltech records every EV
    charging session on campus, including exact plug-in/unplug times
    and energy delivered.  We fetch sessions via their public REST API.

Scenarios:
    1. Dumb Charging:   Charge immediately at max power until full.
    2. Smart Price:     Greedy — charge during cheapest price slots.
    3. Proposed Method: Multi-objective DP with battery degradation penalty.

Output:
    A Pandas DataFrame (and CSV) with per-session and aggregate metrics:
    Total CO₂ emissions, Total Energy Cost, Start/Stop Cycles.

Usage:
    python -m simulation.benchmark_engine --sessions 1000 --output results.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Import our scheduler ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler_algorithm import (
    ChargingSchedule,
    CostWeights,
    ScheduleRequest,
    SlotAction,
    get_charging_blocks,
    schedule_charging,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SLOT_MINUTES: int = 15
CHARGER_KW: float = 7.2                   # Level 2 charger
KWH_PER_SLOT: float = CHARGER_KW * (SLOT_MINUTES / 60.0)  # 1.8 kWh

# Caltech ACN-Data public API
ACN_API_BASE: str = "https://ev.caltech.edu/api/v1"
ACN_SITE: str = "caltech"                  # or "jpl" for JPL site


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EVSession:
    """A single EV charging session (from ACN-Data or synthetic)."""
    session_id: str
    arrival_time: datetime
    departure_time: datetime
    requested_kwh: float
    # Derived
    window_hours: float = 0.0
    n_slots: int = 0
    slots_needed: int = 0

    def __post_init__(self) -> None:
        self.window_hours = (
            (self.departure_time - self.arrival_time).total_seconds() / 3600
        )
        self.n_slots = int(self.window_hours * 60 // SLOT_MINUTES)
        self.slots_needed = math.ceil(self.requested_kwh / KWH_PER_SLOT)


@dataclass
class ScenarioResult:
    """Metrics for a single session under a single scenario."""
    session_id: str
    scenario: str
    total_co2_kg: float
    total_cost_usd: float
    total_energy_kwh: float
    start_stop_cycles: int
    num_charging_blocks: int
    schedule_cost: float  # optimizer's internal cost metric


@dataclass
class BenchmarkResults:
    """Aggregated results across all sessions and scenarios."""
    per_session: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════
#  ACN-Data Fetching
# ═══════════════════════════════════════════════════════════════════════════


def fetch_acn_sessions(
    n_sessions: int = 1000,
    site: str = ACN_SITE,
    start_date: str = "2019-01-01",
    end_date: str = "2021-12-31",
    random_seed: int = 42,
) -> list[EVSession]:
    """
    Fetch historical EV sessions from the Caltech ACN-Data REST API.

    The API returns real charging session records with connection/disconnect
    times and energy delivered.  We sample ``n_sessions`` randomly from
    the available data.

    Parameters
    ----------
    n_sessions : int
        Number of sessions to sample.
    site : str
        ACN site identifier ("caltech" or "jpl").
    start_date : str
        Start of the date range (YYYY-MM-DD).
    end_date : str
        End of the date range (YYYY-MM-DD).
    random_seed : int
        Seed for reproducible random sampling.

    Returns
    -------
    list[EVSession]
        Sampled sessions with valid time windows and energy targets.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available — falling back to synthetic data.")
        return generate_synthetic_sessions(n_sessions, random_seed)

    sessions: list[EVSession] = []
    page_size = 500
    url = f"{ACN_API_BASE}/sessions/{site}"

    logger.info(
        "Fetching ACN-Data sessions: site=%s, range=%s to %s",
        site, start_date, end_date,
    )

    try:
        with httpx.Client(timeout=60.0) as client:
            # Fetch in pages until we have enough valid sessions
            offset = 0
            max_pages = 20  # safety limit

            for page in range(max_pages):
                params = {
                    "where": f'connectionTime>="{start_date}" and connectionTime<="{end_date}"',
                    "page": str(page + 1),
                    "limit": str(page_size),
                }
                response = client.get(url, params=params)

                if response.status_code != 200:
                    logger.warning(
                        "ACN API returned %d: %s",
                        response.status_code, response.text[:200],
                    )
                    break

                data = response.json()
                records = data if isinstance(data, list) else data.get("_items", [])

                if not records:
                    break

                for record in records:
                    session = _parse_acn_record(record)
                    if session is not None:
                        sessions.append(session)

                logger.info(
                    "Page %d: fetched %d records, %d valid sessions total.",
                    page + 1, len(records), len(sessions),
                )

                if len(sessions) >= n_sessions * 2:
                    break  # we have enough to sample from

    except Exception as exc:
        logger.warning("ACN API fetch failed: %s — falling back to synthetic.", exc)

    if len(sessions) < n_sessions:
        logger.info(
            "Got %d sessions from ACN API (need %d). "
            "Supplementing with synthetic data.",
            len(sessions), n_sessions,
        )
        synthetic = generate_synthetic_sessions(
            n_sessions - len(sessions),
            random_seed + len(sessions),
        )
        sessions.extend(synthetic)

    # Sample exactly n_sessions
    rng = np.random.default_rng(random_seed)
    if len(sessions) > n_sessions:
        indices = rng.choice(len(sessions), size=n_sessions, replace=False)
        sessions = [sessions[i] for i in sorted(indices)]

    logger.info("Final dataset: %d sessions.", len(sessions))
    return sessions


def _parse_acn_record(record: dict[str, Any]) -> EVSession | None:
    """
    Parse a single ACN-Data API record into an EVSession.

    Filters out sessions that are:
    - Too short (< 1 hour)
    - Too long (> 24 hours)
    - Too little energy (< 2 kWh)
    - Infeasible (not enough time to deliver the energy)

    Returns None for invalid sessions.
    """
    try:
        # ACN-Data timestamps are ISO format
        connection = record.get("connectionTime") or record.get("connect_time")
        disconnect = record.get("disconnectTime") or record.get("disconnect_time")
        kwh = record.get("kWhDelivered") or record.get("energy_kwh", 0)

        if not connection or not disconnect or not kwh:
            return None

        arrival = datetime.fromisoformat(str(connection).replace("Z", "+00:00"))
        departure = datetime.fromisoformat(str(disconnect).replace("Z", "+00:00"))

        # Ensure timezone-aware
        if arrival.tzinfo is None:
            arrival = arrival.replace(tzinfo=timezone.utc)
        if departure.tzinfo is None:
            departure = departure.replace(tzinfo=timezone.utc)

        kwh_float = float(kwh)

        # Validity checks
        window_hours = (departure - arrival).total_seconds() / 3600
        if window_hours < 1.0 or window_hours > 24.0:
            return None
        if kwh_float < 2.0 or kwh_float > 80.0:
            return None

        # Feasibility: enough slots to deliver the energy?
        n_slots = int(window_hours * 60 // SLOT_MINUTES)
        slots_needed = math.ceil(kwh_float / KWH_PER_SLOT)
        if slots_needed > n_slots:
            return None

        session_id = str(
            record.get("sessionID")
            or record.get("_id")
            or f"acn_{hash(connection) % 100000}"
        )

        return EVSession(
            session_id=session_id,
            arrival_time=arrival,
            departure_time=departure,
            requested_kwh=kwh_float,
        )

    except (ValueError, TypeError, KeyError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic Data Generator (Offline Fallback)
# ═══════════════════════════════════════════════════════════════════════════


def generate_synthetic_sessions(
    n_sessions: int = 1000,
    random_seed: int = 42,
) -> list[EVSession]:
    """
    Generate synthetic EV sessions mimicking Caltech campus patterns.

    Distribution assumptions (from ACN-Data literature):
        - Arrival:  Gaussian centered at 9 AM (σ=3h) for morning,
                    and 6 PM (σ=2h) for evening sessions.
        - Dwell:    Log-normal, median ~8h, range 1–20h.
        - Energy:   Log-normal, median ~15 kWh, range 2–60 kWh.

    Parameters
    ----------
    n_sessions : int
        Number of synthetic sessions.
    random_seed : int
        For reproducibility.

    Returns
    -------
    list[EVSession]
    """
    rng = np.random.default_rng(random_seed)
    sessions: list[EVSession] = []

    # Base date range: 2020-01-01 to 2020-12-31
    base_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    date_range_days = 365

    for i in range(n_sessions):
        # Random day
        day_offset = rng.integers(0, date_range_days)
        day = base_date + timedelta(days=int(day_offset))

        # Arrival hour: bimodal (60% morning commuters, 40% evening)
        if rng.random() < 0.6:
            arrival_hour = rng.normal(9.0, 3.0)   # morning peak
        else:
            arrival_hour = rng.normal(17.0, 2.0)   # evening peak
        arrival_hour = np.clip(arrival_hour, 0, 23.5)

        arrival = day + timedelta(hours=float(arrival_hour))

        # Dwell time: log-normal (median ~8h)
        dwell_hours = float(rng.lognormal(mean=2.0, sigma=0.5))
        dwell_hours = np.clip(dwell_hours, 1.5, 20.0)
        departure = arrival + timedelta(hours=dwell_hours)

        # Requested energy: log-normal (median ~15 kWh)
        requested_kwh = float(rng.lognormal(mean=2.7, sigma=0.6))
        requested_kwh = np.clip(requested_kwh, 2.0, 60.0)

        # Feasibility check
        n_slots = int(dwell_hours * 60 // SLOT_MINUTES)
        slots_needed = math.ceil(requested_kwh / KWH_PER_SLOT)
        if slots_needed > n_slots or n_slots == 0:
            requested_kwh = max(2.0, n_slots * KWH_PER_SLOT * 0.5)

        sessions.append(EVSession(
            session_id=f"syn_{i:05d}",
            arrival_time=arrival,
            departure_time=departure,
            requested_kwh=round(requested_kwh, 2),
        ))

    logger.info("Generated %d synthetic EV sessions.", len(sessions))
    return sessions


# ═══════════════════════════════════════════════════════════════════════════
#  MOER & Price Profile Generators
# ═══════════════════════════════════════════════════════════════════════════


def generate_moer_profile(
    arrival: datetime,
    n_slots: int,
    rng: np.random.Generator,
) -> list[float]:
    """
    Generate a realistic MOER profile for a session's time window.

    Based on typical CAISO (California ISO) marginal emissions patterns:
        - Low overnight (wind + baseload nuclear):  200-400 lbs/MWh
        - Morning ramp (gas peakers come online):   500-700 lbs/MWh
        - Midday solar dip (solar PV displaces gas): 100-300 lbs/MWh
        - Evening peak (solar off, gas ramps up):    700-1000 lbs/MWh

    Parameters
    ----------
    arrival : datetime
        Session start time (used to determine time-of-day).
    n_slots : int
        Number of 15-minute slots.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list[float]
        MOER values in lbs CO₂/MWh for each slot.
    """
    moer: list[float] = []
    for i in range(n_slots):
        slot_time = arrival + timedelta(minutes=SLOT_MINUTES * i)
        hour = slot_time.hour + slot_time.minute / 60.0

        # Diurnal MOER curve based on CAISO patterns
        if 0 <= hour < 6:
            # Overnight: low emissions (wind + nuclear baseload)
            base = 300 + 50 * np.sin(2 * np.pi * hour / 24)
        elif 6 <= hour < 10:
            # Morning ramp: gas peakers starting up
            base = 300 + 300 * ((hour - 6) / 4)
        elif 10 <= hour < 15:
            # Midday solar: emissions drop significantly
            base = 200 + 100 * np.cos(np.pi * (hour - 12.5) / 5)
        elif 15 <= hour < 20:
            # Evening peak: solar dropping, gas ramping
            base = 400 + 400 * ((hour - 15) / 5)
        else:
            # Late evening: cooling off
            base = 800 - 200 * ((hour - 20) / 4)

        # Seasonal adjustment: summer has more solar, lower midday MOER
        month = slot_time.month
        seasonal = 1.0 - 0.15 * np.sin(2 * np.pi * (month - 1) / 12)

        # Random noise (±10%)
        noise = 1.0 + rng.normal(0, 0.10)

        moer.append(max(50.0, base * seasonal * noise))

    return moer


def generate_price_profile(
    arrival: datetime,
    n_slots: int,
) -> list[float]:
    """
    Generate a TOU electricity price profile for Southern California.

    Based on SCE TOU-D-PRIME rate schedule (typical Caltech area):
        - Off-peak:  $0.10/kWh  (9 PM – 8 AM)
        - Mid-peak:  $0.22/kWh  (8-4 PM, 9-11 PM)
        - On-peak:   $0.38/kWh  (4 PM – 9 PM)

    Parameters
    ----------
    arrival : datetime
        Session start time.
    n_slots : int
        Number of 15-minute slots.

    Returns
    -------
    list[float]
        Prices in $/kWh for each slot.
    """
    prices: list[float] = []
    for i in range(n_slots):
        slot_time = arrival + timedelta(minutes=SLOT_MINUTES * i)
        hour = slot_time.hour
        if 16 <= hour < 21:      # On-peak: 4 PM - 9 PM
            prices.append(0.38)
        elif 8 <= hour < 16:     # Mid-peak: 8 AM - 4 PM
            prices.append(0.22)
        elif 21 <= hour < 23:    # Mid-peak: 9 PM - 11 PM
            prices.append(0.22)
        else:                    # Off-peak: 11 PM - 8 AM
            prices.append(0.10)
    return prices


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline Strategies
# ═══════════════════════════════════════════════════════════════════════════


def run_dumb_charging(
    session: EVSession,
    moer_profile: list[float],
    price_profile: list[float],
) -> ScenarioResult:
    """
    Baseline 1: Charge immediately at max power until full.

    Simulates the behavior of a driver who plugs in and walks away.
    The vehicle charges continuously from the first slot until the
    energy target is met, regardless of carbon intensity or price.
    """
    n_slots = session.n_slots
    slots_needed = session.slots_needed
    actions = [1] * min(slots_needed, n_slots) + [0] * max(0, n_slots - slots_needed)

    return _compute_scenario_metrics(
        session=session,
        actions=actions,
        moer_profile=moer_profile,
        price_profile=price_profile,
        scenario_name="Dumb Charging",
    )


def run_smart_price_charging(
    session: EVSession,
    moer_profile: list[float],
    price_profile: list[float],
) -> ScenarioResult:
    """
    Baseline 2: Greedy price-optimal charging.

    Sorts all time slots by electricity price (ascending), then charges
    during the cheapest slots.  This is a common "smart charging"
    approach but ignores carbon emissions and causes maximum
    start-stop fragmentation (worst for battery degradation).
    """
    n_slots = session.n_slots
    slots_needed = session.slots_needed

    # Sort slot indices by price (cheapest first), break ties by time
    slot_indices = list(range(n_slots))
    slot_indices.sort(key=lambda i: (price_profile[i], i))

    # Assign charging to the cheapest slots
    actions = [0] * n_slots
    for idx in slot_indices[:slots_needed]:
        actions[idx] = 1

    return _compute_scenario_metrics(
        session=session,
        actions=actions,
        moer_profile=moer_profile,
        price_profile=price_profile,
        scenario_name="Smart Price",
    )


def run_proposed_method(
    session: EVSession,
    moer_profile: list[float],
    price_profile: list[float],
    weights: CostWeights | None = None,
) -> ScenarioResult:
    """
    Proposed Method: Multi-objective DP with battery degradation penalty.

    Uses our scheduler_algorithm.schedule_charging() to find the globally
    optimal charge/idle assignment that minimizes:
        Cost = w₁·Carbon + w₂·Price + w₃·DegradationPenalty
    """
    if weights is None:
        weights = CostWeights(w_carbon=1.0, w_price=0.5, w_degradation=2.0)

    request = ScheduleRequest(
        arrival_time=session.arrival_time,
        departure_time=session.departure_time,
        target_kwh=session.requested_kwh,
        charger_power_kw=CHARGER_KW,
        slot_duration_minutes=SLOT_MINUTES,
        moer_forecast=moer_profile,
        price_forecast=price_profile,
        weights=weights,
    )

    schedule: ChargingSchedule = schedule_charging(request)

    # Extract action list
    actions = [
        1 if slot.action == SlotAction.CHARGE else 0
        for slot in schedule.slots
    ]

    result = _compute_scenario_metrics(
        session=session,
        actions=actions,
        moer_profile=moer_profile,
        price_profile=price_profile,
        scenario_name="Proposed Method",
    )
    result.schedule_cost = schedule.total_cost
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════


def _compute_scenario_metrics(
    session: EVSession,
    actions: list[int],
    moer_profile: list[float],
    price_profile: list[float],
    scenario_name: str,
) -> ScenarioResult:
    """
    Compute CO₂, cost, and cycling metrics for a given action sequence.

    Parameters
    ----------
    actions : list[int]
        0/1 action for each slot (1=charge, 0=idle).
    moer_profile : list[float]
        MOER values (lbs CO₂/MWh) per slot.
    price_profile : list[float]
        Electricity prices ($/kWh) per slot.

    Returns
    -------
    ScenarioResult
    """
    total_co2_kg = 0.0
    total_cost_usd = 0.0
    total_energy_kwh = 0.0
    transitions = 0
    prev_action = 0

    for i, action in enumerate(actions):
        if action == 1:
            # CO₂: MOER (lbs/MWh) × kWh/1000 → lbs × 0.4536 → kg
            total_co2_kg += moer_profile[i] * (KWH_PER_SLOT / 1000.0) * 0.453592
            total_cost_usd += price_profile[i] * KWH_PER_SLOT
            total_energy_kwh += KWH_PER_SLOT
        if i > 0 and action != prev_action:
            transitions += 1
        prev_action = action

    # Count contiguous charging blocks
    num_blocks = 0
    in_block = False
    for a in actions:
        if a == 1 and not in_block:
            num_blocks += 1
            in_block = True
        elif a == 0:
            in_block = False

    # Start-stop cycles = number of times charging starts (= num_blocks)
    start_stop_cycles = num_blocks

    return ScenarioResult(
        session_id=session.session_id,
        scenario=scenario_name,
        total_co2_kg=round(total_co2_kg, 6),
        total_cost_usd=round(total_cost_usd, 4),
        total_energy_kwh=round(total_energy_kwh, 2),
        start_stop_cycles=start_stop_cycles,
        num_charging_blocks=num_blocks,
        schedule_cost=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark Loop
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmark(
    sessions: list[EVSession],
    weights: CostWeights | None = None,
    random_seed: int = 42,
) -> BenchmarkResults:
    """
    Run all three scenarios across all sessions.

    Parameters
    ----------
    sessions : list[EVSession]
        EV sessions to benchmark.
    weights : CostWeights | None
        Weights for the proposed method. Defaults to (1.0, 0.5, 2.0).
    random_seed : int
        Seed for MOER profile generation.

    Returns
    -------
    BenchmarkResults
        Per-session results DataFrame and aggregate summary DataFrame.
    """
    if weights is None:
        weights = CostWeights(w_carbon=1.0, w_price=0.5, w_degradation=2.0)

    rng = np.random.default_rng(random_seed)
    all_results: list[dict[str, Any]] = []
    n_total = len(sessions)
    skipped = 0

    logger.info("Starting benchmark: %d sessions × 3 scenarios", n_total)

    for idx, session in enumerate(sessions):
        if (idx + 1) % 100 == 0 or idx == 0:
            logger.info("  Progress: %d / %d sessions…", idx + 1, n_total)

        # Skip infeasible sessions
        if session.n_slots == 0 or session.slots_needed > session.n_slots:
            skipped += 1
            continue

        # Generate environment profiles for this session
        moer_profile = generate_moer_profile(session.arrival_time, session.n_slots, rng)
        price_profile = generate_price_profile(session.arrival_time, session.n_slots)

        # Run all three scenarios
        try:
            dumb = run_dumb_charging(session, moer_profile, price_profile)
            smart = run_smart_price_charging(session, moer_profile, price_profile)
            proposed = run_proposed_method(session, moer_profile, price_profile, weights)

            for result in [dumb, smart, proposed]:
                all_results.append({
                    "session_id": result.session_id,
                    "scenario": result.scenario,
                    "arrival_time": session.arrival_time.isoformat(),
                    "departure_time": session.departure_time.isoformat(),
                    "requested_kwh": session.requested_kwh,
                    "window_hours": round(session.window_hours, 2),
                    "co2_kg": result.total_co2_kg,
                    "cost_usd": result.total_cost_usd,
                    "energy_kwh": result.total_energy_kwh,
                    "start_stop_cycles": result.start_stop_cycles,
                    "charging_blocks": result.num_charging_blocks,
                    "schedule_cost": result.schedule_cost,
                })

        except Exception as exc:
            logger.warning(
                "Session %s failed: %s — skipping.", session.session_id, exc,
            )
            skipped += 1
            continue

    logger.info(
        "Benchmark complete: %d sessions processed, %d skipped.",
        n_total - skipped, skipped,
    )

    # ── Build DataFrames ────────────────────────────────────────────────────
    per_session_df = pd.DataFrame(all_results)

    # Aggregate summary by scenario
    if not per_session_df.empty:
        summary_df = (
            per_session_df
            .groupby("scenario")
            .agg(
                sessions=("session_id", "nunique"),
                total_co2_kg=("co2_kg", "sum"),
                mean_co2_kg=("co2_kg", "mean"),
                total_cost_usd=("cost_usd", "sum"),
                mean_cost_usd=("cost_usd", "mean"),
                total_energy_kwh=("energy_kwh", "sum"),
                total_start_stop_cycles=("start_stop_cycles", "sum"),
                mean_start_stop_cycles=("start_stop_cycles", "mean"),
                mean_charging_blocks=("charging_blocks", "mean"),
            )
            .round(4)
            .reset_index()
        )

        # Add percentage improvements vs dumb baseline
        dumb_row = summary_df[summary_df["scenario"] == "Dumb Charging"]
        if not dumb_row.empty:
            dumb_co2 = dumb_row["total_co2_kg"].values[0]
            dumb_cost = dumb_row["total_cost_usd"].values[0]
            dumb_cycles = dumb_row["total_start_stop_cycles"].values[0]

            summary_df["co2_reduction_pct"] = round(
                (1 - summary_df["total_co2_kg"] / dumb_co2) * 100, 2
            )
            summary_df["cost_reduction_pct"] = round(
                (1 - summary_df["total_cost_usd"] / dumb_cost) * 100, 2
            )
            summary_df["cycle_change_pct"] = round(
                (1 - summary_df["total_start_stop_cycles"] / dumb_cycles) * 100, 2
            )
    else:
        summary_df = pd.DataFrame()

    return BenchmarkResults(per_session=per_session_df, summary=summary_df)


# ═══════════════════════════════════════════════════════════════════════════
#  Output Formatting
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(results: BenchmarkResults) -> None:
    """Print a formatted summary table to stdout."""
    if results.summary.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("  BENCHMARK RESULTS — Carbon-Aware EV Charging")
    print("=" * 90)

    # Per-scenario summary
    for _, row in results.summary.iterrows():
        print(f"\n  ── {row['scenario']} {'─' * (60 - len(str(row['scenario'])))}──")
        print(f"    Sessions:              {int(row['sessions']):,}")
        print(f"    Total CO₂:             {row['total_co2_kg']:.2f} kg")
        print(f"    Mean CO₂/session:      {row['mean_co2_kg']:.4f} kg")
        print(f"    Total Cost:            ${row['total_cost_usd']:.2f}")
        print(f"    Mean Cost/session:     ${row['mean_cost_usd']:.4f}")
        print(f"    Total Start/Stop:      {int(row['total_start_stop_cycles']):,} cycles")
        print(f"    Mean Blocks/session:   {row['mean_charging_blocks']:.2f}")
        if "co2_reduction_pct" in row:
            print(f"    CO₂ vs Dumb:           {row['co2_reduction_pct']:+.2f}%")
            print(f"    Cost vs Dumb:          {row['cost_reduction_pct']:+.2f}%")
            print(f"    Cycles vs Dumb:        {row['cycle_change_pct']:+.2f}%")

    print("\n" + "=" * 90)


def save_results(
    results: BenchmarkResults,
    output_dir: str | Path = ".",
) -> tuple[Path, Path]:
    """
    Save results to CSV files.

    Returns
    -------
    tuple[Path, Path]
        Paths to (per_session.csv, summary.csv).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_session_path = out / "benchmark_per_session.csv"
    summary_path = out / "benchmark_summary.csv"

    results.per_session.to_csv(per_session_path, index=False)
    results.summary.to_csv(summary_path, index=False)

    logger.info("Results saved: %s, %s", per_session_path, summary_path)
    return per_session_path, summary_path


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entrypoint
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entrypoint for the benchmark engine."""
    parser = argparse.ArgumentParser(
        description="Benchmark Carbon-Aware EV Charging against baselines.",
    )
    parser.add_argument(
        "--sessions", type=int, default=1000,
        help="Number of EV sessions to simulate (default: 1000).",
    )
    parser.add_argument(
        "--output", type=str, default="simulation/results",
        help="Output directory for CSV files (default: simulation/results).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of ACN-Data API.",
    )
    parser.add_argument(
        "--w-carbon", type=float, default=1.0,
        help="Carbon weight for proposed method (default: 1.0).",
    )
    parser.add_argument(
        "--w-price", type=float, default=0.5,
        help="Price weight for proposed method (default: 0.5).",
    )
    parser.add_argument(
        "--w-degradation", type=float, default=2.0,
        help="Degradation penalty weight (default: 2.0).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Fetch sessions ──────────────────────────────────────────────────────
    if args.synthetic:
        sessions = generate_synthetic_sessions(args.sessions, args.seed)
    else:
        sessions = fetch_acn_sessions(args.sessions, random_seed=args.seed)

    # ── Run benchmark ───────────────────────────────────────────────────────
    weights = CostWeights(
        w_carbon=args.w_carbon,
        w_price=args.w_price,
        w_degradation=args.w_degradation,
    )
    results = run_benchmark(sessions, weights=weights, random_seed=args.seed)

    # ── Output ──────────────────────────────────────────────────────────────
    print_summary(results)
    per_session_path, summary_path = save_results(results, args.output)
    print(f"\n  CSV saved: {per_session_path}")
    print(f"  CSV saved: {summary_path}")


if __name__ == "__main__":
    main()
