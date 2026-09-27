"""
Carbon-Aware EV Charging Scheduler — Dynamic Programming Optimizer.

Schedules EV charging in 15-minute intervals between arrival and departure
to minimize a multi-objective cost:

    Cost = (w₁ × Carbon_Emissions) + (w₂ × Electricity_Price) + (w₃ × Battery_Degradation_Penalty)

The battery degradation penalty discourages frequent charge/idle transitions
("start-stop cycling") that accelerate lithium-ion calendar and cycle aging.

Algorithm:
    Dynamic Programming over state space (time_slot, slots_charged, was_charging).
    Guarantees global optimality given the discretized slot model.

    State space complexity: O(N × K × 2)
        N = number of time slots (e.g., 96 for a 24h window)
        K = number of slots required to meet target_kwh
        2 = previous charging state (charging / idle)

    For a typical 24h session with 30 kWh target at 7.2 kW:
        96 × 17 × 2 = 3,264 states — trivially fast.

Public API:
    schedule_charging(request) → ChargingSchedule
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Default slot duration in minutes
SLOT_DURATION_MINUTES: int = 15

# Default Level 2 charger power (kW)
DEFAULT_CHARGER_POWER_KW: float = 7.2


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════


class SlotAction(str, Enum):
    """Action for a single time slot."""
    CHARGE = "charge"
    IDLE = "idle"


@dataclass(frozen=True)
class CostWeights:
    """
    Weights for the multi-objective cost function.

    Tuning guidance:
        • w_carbon:       Primary objective. Normalized to $/lb CO₂.
        • w_price:        Electricity cost weight. Set to 1.0 for $/kWh parity.
        • w_degradation:  Switching penalty. Higher values produce fewer,
                          longer charging blocks. Recommended range: 0.5–5.0.
                          A value of 2.0 typically prevents fragmentation
                          while still allowing 2–3 charging blocks.
    """
    w_carbon: float = 1.0
    w_price: float = 0.5
    w_degradation: float = 2.0


@dataclass
class ScheduleRequest:
    """
    Input parameters for the charging scheduler.

    Attributes
    ----------
    arrival_time : datetime
        When the vehicle is plugged in (timezone-aware).
    departure_time : datetime
        When the vehicle must be unplugged (timezone-aware).
    target_kwh : float
        Total energy to deliver (kWh). Must be > 0.
    starting_kwh : float
        Energy already in the battery (kWh). Informational only.
    charger_power_kw : float
        Charger output power (kW). Default: 7.2 (Level 2).
    slot_duration_minutes : int
        Granularity of scheduling (minutes). Default: 15.
    moer_forecast : list[float]
        MOER values (lbs CO₂/MWh) for each time slot.
        Length must equal the number of slots in the window.
    price_forecast : list[float] | None
        Electricity prices ($/kWh) for each time slot.
        If None, uniform pricing is assumed (cost weight ignored).
    weights : CostWeights
        Objective function weights.
    """
    arrival_time: datetime
    departure_time: datetime
    target_kwh: float
    starting_kwh: float = 0.0
    charger_power_kw: float = DEFAULT_CHARGER_POWER_KW
    slot_duration_minutes: int = SLOT_DURATION_MINUTES
    moer_forecast: list[float] = field(default_factory=list)
    price_forecast: list[float] | None = None
    weights: CostWeights = field(default_factory=CostWeights)


@dataclass
class ScheduledSlot:
    """A single scheduled time slot with its action and cost breakdown."""
    start_time: datetime
    end_time: datetime
    action: SlotAction
    moer_value: float
    price_value: float
    slot_cost: float
    switching_penalty_applied: bool


@dataclass
class ChargingSchedule:
    """
    Output of the scheduling algorithm.

    Attributes
    ----------
    slots : list[ScheduledSlot]
        Ordered list of all slots with their assigned actions.
    charging_slots : list[ScheduledSlot]
        Only the slots where action == CHARGE.
    total_cost : float
        Minimized total cost from the DP solution.
    total_carbon_kg : float
        Total CO₂ emissions from charging (kg).
    total_energy_kwh : float
        Total energy scheduled (kWh).
    total_price_cost : float
        Total electricity cost ($).
    num_charging_blocks : int
        Number of contiguous charging blocks (fewer = less degradation).
    num_transitions : int
        Number of charge↔idle state transitions.
    slots_charged : int
        Number of slots where charging occurs.
    slots_idle : int
        Number of slots where the vehicle is idle.
    """
    slots: list[ScheduledSlot] = field(default_factory=list)
    charging_slots: list[ScheduledSlot] = field(default_factory=list)
    total_cost: float = 0.0
    total_carbon_kg: float = 0.0
    total_energy_kwh: float = 0.0
    total_price_cost: float = 0.0
    num_charging_blocks: int = 0
    num_transitions: int = 0
    slots_charged: int = 0
    slots_idle: int = 0

    def summary(self) -> str:
        """Human-readable summary of the schedule."""
        lines = [
            "═══ Charging Schedule ═══",
            f"  Energy delivered:    {self.total_energy_kwh:.2f} kWh",
            f"  Total cost:          {self.total_cost:.4f}",
            f"  Carbon emissions:    {self.total_carbon_kg:.4f} kg CO₂",
            f"  Electricity cost:    ${self.total_price_cost:.4f}",
            f"  Charging blocks:     {self.num_charging_blocks}",
            f"  State transitions:   {self.num_transitions}",
            f"  Slots (charge/idle): {self.slots_charged}/{self.slots_idle}",
        ]
        if self.charging_slots:
            lines.append(f"  First charge slot:   {self.charging_slots[0].start_time}")
            lines.append(f"  Last charge slot:    {self.charging_slots[-1].start_time}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_request(request: ScheduleRequest) -> tuple[int, int, float]:
    """
    Validate the schedule request and compute derived quantities.

    Returns
    -------
    tuple[int, int, float]
        (n_slots, slots_needed, kwh_per_slot)

    Raises
    ------
    ValueError
        If any constraint is infeasible.
    """
    if request.departure_time <= request.arrival_time:
        raise ValueError(
            f"departure_time ({request.departure_time}) must be after "
            f"arrival_time ({request.arrival_time})"
        )

    if request.target_kwh <= 0:
        raise ValueError(f"target_kwh must be > 0, got {request.target_kwh}")

    if request.charger_power_kw <= 0:
        raise ValueError(f"charger_power_kw must be > 0, got {request.charger_power_kw}")

    # Compute slot parameters
    window_minutes = (request.departure_time - request.arrival_time).total_seconds() / 60
    n_slots = int(window_minutes // request.slot_duration_minutes)

    if n_slots == 0:
        raise ValueError(
            f"Time window ({window_minutes:.0f} min) is shorter than "
            f"one slot ({request.slot_duration_minutes} min)"
        )

    # Energy per slot = power × (slot_duration / 60)
    kwh_per_slot = request.charger_power_kw * (request.slot_duration_minutes / 60.0)

    # Minimum slots needed (ceiling division)
    slots_needed = math.ceil(request.target_kwh / kwh_per_slot)

    if slots_needed > n_slots:
        raise ValueError(
            f"Infeasible: need {slots_needed} slots ({request.target_kwh:.1f} kWh "
            f"at {kwh_per_slot:.2f} kWh/slot) but only {n_slots} slots available "
            f"in the {window_minutes:.0f}-minute window"
        )

    # Validate forecast lengths
    if len(request.moer_forecast) != n_slots:
        raise ValueError(
            f"moer_forecast length ({len(request.moer_forecast)}) must match "
            f"number of slots ({n_slots})"
        )

    if request.price_forecast is not None and len(request.price_forecast) != n_slots:
        raise ValueError(
            f"price_forecast length ({len(request.price_forecast)}) must match "
            f"number of slots ({n_slots})"
        )

    return n_slots, slots_needed, kwh_per_slot


# ═══════════════════════════════════════════════════════════════════════════
#  Dynamic Programming Optimizer
# ═══════════════════════════════════════════════════════════════════════════


def _compute_slot_cost(
    action: int,
    prev_action: int,
    moer: float,
    price: float,
    kwh_per_slot: float,
    weights: CostWeights,
) -> tuple[float, bool]:
    """
    Compute the cost of a single slot decision.

    Parameters
    ----------
    action : int
        1 = charge, 0 = idle.
    prev_action : int
        Previous slot's action (1 or 0). -1 if this is the first slot.
    moer : float
        MOER value for this slot (lbs CO₂/MWh).
    price : float
        Electricity price for this slot ($/kWh).
    kwh_per_slot : float
        Energy delivered per charging slot (kWh).
    weights : CostWeights
        Objective function weights.

    Returns
    -------
    tuple[float, bool]
        (cost, switching_penalty_applied)

    Cost breakdown when charging (action=1):
        Carbon cost  = w₁ × MOER × (kwh / 1000)
            MOER is in lbs CO₂ per MWh, kwh/1000 converts kWh→MWh.
        Price cost   = w₂ × price × kwh
        Switch cost  = w₃  (only if action ≠ prev_action AND prev_action ≥ 0)

    When idle (action=0):
        Carbon cost  = 0
        Price cost   = 0
        Switch cost  = w₃  (only if transitioning from charging)
    """
    cost = 0.0
    switching = False

    if action == 1:
        # Carbon emissions cost: MOER (lbs/MWh) × energy (MWh)
        carbon_cost = weights.w_carbon * moer * (kwh_per_slot / 1000.0)
        # Electricity price cost
        price_cost = weights.w_price * price * kwh_per_slot
        cost += carbon_cost + price_cost

    # Battery degradation penalty for state transitions
    # Only apply if we have a prior state to compare against (prev_action ≥ 0)
    if prev_action >= 0 and action != prev_action:
        cost += weights.w_degradation
        switching = True

    return cost, switching


def _dp_optimize(
    n_slots: int,
    slots_needed: int,
    moer: np.ndarray,
    price: np.ndarray,
    kwh_per_slot: float,
    weights: CostWeights,
) -> list[int]:
    """
    Find the optimal charge/idle assignment via dynamic programming.

    State: (t, k, c)
        t = time slot index (0 … N-1)
        k = number of slots charged so far (0 … slots_needed)
        c = previous charging state (0=idle, 1=charging)

    Transition:
        For each state, try both actions (charge / idle).
        If charging: k → k+1 (capped at slots_needed)
        If idle:     k → k

    Terminal constraint:
        Only states with k == slots_needed at t == N are valid.

    We use backward induction (fill from t=N-1 to t=0) to reconstruct
    the optimal action sequence.
    """
    INF = float("inf")

    # DP table: cost_to_go[t][k][c] = minimum cost from slot t to end
    # Shape: (n_slots + 1) × (slots_needed + 1) × 2
    cost_to_go = np.full((n_slots + 1, slots_needed + 1, 2), INF, dtype=np.float64)

    # Action table: action_taken[t][k][c] = optimal action at state (t, k, c)
    action_taken = np.full((n_slots, slots_needed + 1, 2), -1, dtype=np.int8)

    # ── Base case: at t=N, only states with k=slots_needed are feasible ────
    cost_to_go[n_slots, slots_needed, 0] = 0.0
    cost_to_go[n_slots, slots_needed, 1] = 0.0

    # ── Backward induction ─────────────────────────────────────────────────
    for t in range(n_slots - 1, -1, -1):
        slots_remaining = n_slots - t
        for k in range(slots_needed + 1):
            for c in range(2):  # c=0 (was idle), c=1 (was charging)
                best_cost = INF
                best_action = -1

                # Determine the "previous action" for cost computation.
                # At t=0, there is no predecessor → use -1 (no switching penalty)
                prev_action = c if t > 0 else -1

                # ── Option 1: IDLE this slot ────────────────────────────
                # Feasibility: remaining slots after this one must be enough
                # to charge the remaining (slots_needed - k) slots
                remaining_to_charge = slots_needed - k
                if remaining_to_charge <= (slots_remaining - 1):
                    idle_cost, _ = _compute_slot_cost(
                        action=0, prev_action=prev_action,
                        moer=moer[t], price=price[t],
                        kwh_per_slot=kwh_per_slot, weights=weights,
                    )
                    future = cost_to_go[t + 1, k, 0]
                    total = idle_cost + future
                    if total < best_cost:
                        best_cost = total
                        best_action = 0

                # ── Option 2: CHARGE this slot ──────────────────────────
                if k < slots_needed:
                    charge_cost, _ = _compute_slot_cost(
                        action=1, prev_action=prev_action,
                        moer=moer[t], price=price[t],
                        kwh_per_slot=kwh_per_slot, weights=weights,
                    )
                    future = cost_to_go[t + 1, k + 1, 1]
                    total = charge_cost + future
                    if total < best_cost:
                        best_cost = total
                        best_action = 1

                cost_to_go[t, k, c] = best_cost
                action_taken[t, k, c] = best_action

    # ── Forward pass: reconstruct optimal action sequence ──────────────────
    actions: list[int] = []
    k = 0
    c = 0  # start from idle state

    for t in range(n_slots):
        # At t=0, use c=0 (no prior state)
        a = int(action_taken[t, k, c if t > 0 else 0])
        if a < 0:
            # Fallback: if DP couldn't find a valid path (shouldn't happen
            # after validation), force charge if behind schedule
            a = 1 if (slots_needed - k) >= (n_slots - t) else 0
            logger.warning("DP fallback at t=%d: forced action=%d", t, a)

        actions.append(a)
        if a == 1:
            k += 1
        c = a

    return actions


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════


def schedule_charging(request: ScheduleRequest) -> ChargingSchedule:
    """
    Schedule EV charging to minimize the multi-objective cost function.

    Uses dynamic programming to find the globally optimal charge/idle
    assignment across all time slots, subject to the energy delivery
    constraint and battery degradation penalty.

    Parameters
    ----------
    request : ScheduleRequest
        All input parameters: time window, energy target, forecasts, weights.

    Returns
    -------
    ChargingSchedule
        Complete schedule with per-slot actions, cost breakdown, and summary.

    Raises
    ------
    ValueError
        If the request is infeasible (e.g., not enough time to charge).

    Example
    -------
    >>> from datetime import datetime, timezone
    >>> request = ScheduleRequest(
    ...     arrival_time=datetime(2024, 6, 1, 18, 0, tzinfo=timezone.utc),
    ...     departure_time=datetime(2024, 6, 2, 8, 0, tzinfo=timezone.utc),
    ...     target_kwh=30.0,
    ...     moer_forecast=[...],  # 56 values (14h × 4 slots/h)
    ... )
    >>> schedule = schedule_charging(request)
    >>> print(schedule.summary())
    """
    # ── Validate ────────────────────────────────────────────────────────────
    n_slots, slots_needed, kwh_per_slot = _validate_request(request)

    logger.info(
        "Scheduling: %d slots, %d needed, %.2f kWh/slot, weights=(%s)",
        n_slots, slots_needed, kwh_per_slot,
        f"carbon={request.weights.w_carbon}, price={request.weights.w_price}, "
        f"degradation={request.weights.w_degradation}",
    )

    # ── Prepare forecast arrays ─────────────────────────────────────────────
    moer = np.array(request.moer_forecast, dtype=np.float64)
    price = (
        np.array(request.price_forecast, dtype=np.float64)
        if request.price_forecast is not None
        else np.zeros(n_slots, dtype=np.float64)
    )

    # ── Run DP optimizer ────────────────────────────────────────────────────
    actions = _dp_optimize(
        n_slots=n_slots,
        slots_needed=slots_needed,
        moer=moer,
        price=price,
        kwh_per_slot=kwh_per_slot,
        weights=request.weights,
    )

    # ── Build schedule output ───────────────────────────────────────────────
    slot_delta = timedelta(minutes=request.slot_duration_minutes)
    schedule = ChargingSchedule()
    total_cost = 0.0
    total_carbon = 0.0
    total_price_cost = 0.0
    transitions = 0
    prev_action = -1

    for t, action in enumerate(actions):
        slot_start = request.arrival_time + t * slot_delta
        slot_end = slot_start + slot_delta

        slot_cost, switching = _compute_slot_cost(
            action=action,
            prev_action=prev_action if t > 0 else -1,
            moer=float(moer[t]),
            price=float(price[t]),
            kwh_per_slot=kwh_per_slot,
            weights=request.weights,
        )

        total_cost += slot_cost
        if switching:
            transitions += 1

        if action == 1:
            # Carbon: MOER (lbs/MWh) × kWh/1000 → lbs, then × 0.4536 → kg
            total_carbon += float(moer[t]) * (kwh_per_slot / 1000.0) * 0.453592
            total_price_cost += float(price[t]) * kwh_per_slot

        slot = ScheduledSlot(
            start_time=slot_start,
            end_time=slot_end,
            action=SlotAction.CHARGE if action == 1 else SlotAction.IDLE,
            moer_value=float(moer[t]),
            price_value=float(price[t]),
            slot_cost=slot_cost,
            switching_penalty_applied=switching,
        )
        schedule.slots.append(slot)
        if action == 1:
            schedule.charging_slots.append(slot)

        prev_action = action

    # ── Compute charging blocks ─────────────────────────────────────────────
    num_blocks = 0
    in_block = False
    for a in actions:
        if a == 1 and not in_block:
            num_blocks += 1
            in_block = True
        elif a == 0:
            in_block = False

    # ── Populate summary fields ─────────────────────────────────────────────
    schedule.total_cost = total_cost
    schedule.total_carbon_kg = total_carbon
    schedule.total_energy_kwh = slots_needed * kwh_per_slot
    schedule.total_price_cost = total_price_cost
    schedule.num_charging_blocks = num_blocks
    schedule.num_transitions = transitions
    schedule.slots_charged = sum(1 for a in actions if a == 1)
    schedule.slots_idle = sum(1 for a in actions if a == 0)

    logger.info(
        "Schedule complete: %.2f kWh in %d blocks (%d transitions), cost=%.4f",
        schedule.total_energy_kwh, num_blocks, transitions, total_cost,
    )

    return schedule


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: Extract Charging Timestamps
# ═══════════════════════════════════════════════════════════════════════════


def get_charging_timestamps(schedule: ChargingSchedule) -> list[datetime]:
    """
    Extract the start times of all charging slots.

    Returns a sorted list of datetime objects representing when the
    charger should be active.  Useful for sending control commands
    to the Smartcar API or a smart-plug controller.

    Parameters
    ----------
    schedule : ChargingSchedule
        A completed charging schedule.

    Returns
    -------
    list[datetime]
        Sorted charging slot start times.
    """
    return [slot.start_time for slot in schedule.charging_slots]


def get_charging_blocks(
    schedule: ChargingSchedule,
) -> list[tuple[datetime, datetime]]:
    """
    Extract contiguous charging blocks as (start, end) tuples.

    Merges consecutive charging slots into blocks.  Useful for
    displaying to the user: "Charge from 11:00 PM to 2:30 AM"
    rather than listing 14 individual 15-minute slots.

    Parameters
    ----------
    schedule : ChargingSchedule
        A completed charging schedule.

    Returns
    -------
    list[tuple[datetime, datetime]]
        List of (block_start, block_end) tuples.
    """
    if not schedule.charging_slots:
        return []

    blocks: list[tuple[datetime, datetime]] = []
    block_start = schedule.charging_slots[0].start_time
    block_end = schedule.charging_slots[0].end_time

    for slot in schedule.charging_slots[1:]:
        if slot.start_time == block_end:
            # Extend current block
            block_end = slot.end_time
        else:
            # Finish current block, start new one
            blocks.append((block_start, block_end))
            block_start = slot.start_time
            block_end = slot.end_time

    blocks.append((block_start, block_end))
    return blocks


def visualize_schedule(schedule: ChargingSchedule) -> str:
    """
    Render a text-based timeline of the charging schedule.

    Uses █ for charging slots and ░ for idle slots.
    Useful for quick terminal debugging and logging.

    Parameters
    ----------
    schedule : ChargingSchedule
        A completed charging schedule.

    Returns
    -------
    str
        Multi-line string with the visual timeline and legend.
    """
    if not schedule.slots:
        return "(empty schedule)"

    # Build timeline bar
    bar_chars: list[str] = []
    for slot in schedule.slots:
        bar_chars.append("█" if slot.action == SlotAction.CHARGE else "░")

    timeline = "".join(bar_chars)

    # Annotate with times
    first = schedule.slots[0].start_time.strftime("%H:%M")
    last = schedule.slots[-1].end_time.strftime("%H:%M")
    mid_idx = len(schedule.slots) // 2
    mid = schedule.slots[mid_idx].start_time.strftime("%H:%M")

    lines = [
        f"  {first}{' ' * (len(timeline) - len(first) - len(last))}{last}",
        f"  {timeline}",
        f"  █ = charging   ░ = idle   ({len(schedule.slots)} slots × 15 min)",
        "",
        f"  Blocks: {schedule.num_charging_blocks}  |  "
        f"Transitions: {schedule.num_transitions}  |  "
        f"Cost: {schedule.total_cost:.4f}",
    ]
    return "\n".join(lines)
