"""
Celery task definitions for the Carbon-Aware EV Charging system.

Tasks:
    fetch_and_store_grid_data  — Periodic (15-min) ingestion of MOER + weather data.
    generate_charging_schedule — On-demand DP optimization for a charging session.
    execute_smartcar_command   — Timed physical charge start/stop via Smartcar API.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.celery_app import celery

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Async Bridge
# ═══════════════════════════════════════════════════════════════════════════


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from a synchronous Celery task context.
    
    Reuses the global event loop to ensure SQLAlchemy's asyncpg
    connection pool remains attached to a valid loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
#  Task 1: Periodic Grid Data Ingestion
# ═══════════════════════════════════════════════════════════════════════════


@celery.task(
    name="app.tasks.fetch_and_store_grid_data",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    queue="grid_data",
)
def fetch_and_store_grid_data(self: Any) -> dict[str, object]:
    """
    Periodically fetch MOER and weather data for all active charging sessions.

    Scheduled by Celery Beat every 15 minutes. For each unique (lat, lon)
    across active sessions, it:
    1. Fetches realtime MOER from WattTime.
    2. Fetches hourly weather from Open-Meteo.
    3. Persists combined records into the GridData table.

    Returns:
        Summary dict with record counts and timestamp.
    """
    logger.info("fetch_and_store_grid_data: Starting periodic ingestion…")

    try:
        result = _run_async(_fetch_and_store_grid_data_async())
        return result
    except Exception as exc:
        logger.exception("fetch_and_store_grid_data: Failed — retrying…")
        raise self.retry(exc=exc)


async def _fetch_and_store_grid_data_async() -> dict[str, object]:
    """Async implementation: fetch MOER + weather for active session locations."""
    from sqlalchemy import select

    from app.api_clients import OpenMeteoClient, WattTimeClient
    from app.database import AsyncSessionLocal
    from app.models import ChargingSession, GridData, GridDataSource, SessionStatus

    records_created: int = 0

    # 1. Find unique locations of active/optimizing sessions
    async with AsyncSessionLocal() as db:
        stmt = (
            select(
                ChargingSession.latitude,
                ChargingSession.longitude,
            )
            .where(ChargingSession.status.in_([
                SessionStatus.ACTIVE,
                SessionStatus.OPTIMIZING,
            ]))
            .distinct()
        )
        result = await db.execute(stmt)
        locations: list[tuple[float, float]] = [
            (row[0], row[1]) for row in result.all()
        ]

    if not locations:
        logger.info("fetch_and_store_grid_data: No active sessions — skipping.")
        return {"records_created": 0, "timestamp": datetime.now(timezone.utc).isoformat()}

    logger.info(
        "fetch_and_store_grid_data: Fetching data for %d unique locations.",
        len(locations),
    )

    # 2. Fetch MOER + weather for each location
    async with WattTimeClient() as wt, OpenMeteoClient() as om:
        async with AsyncSessionLocal() as db:
            for lat, lon in locations:
                now = datetime.now(timezone.utc)

                # MOER from WattTime
                moer_value: float | None = None
                try:
                    signal = await wt.get_realtime_signal(latitude=lat, longitude=lon)
                    if signal.data:
                        moer_value = signal.data[0].value
                except Exception:
                    logger.warning(
                        "WattTime fetch failed for (%.4f, %.4f)", lat, lon,
                        exc_info=True,
                    )

                # Weather from Open-Meteo
                solar: float | None = None
                temp: float | None = None
                cloud: float | None = None
                wind: float | None = None
                try:
                    weather = await om.get_weather_forecast(
                        latitude=lat, longitude=lon, forecast_days=1,
                    )
                    if weather.hourly.time:
                        idx = 0
                        solar = (
                            weather.hourly.shortwave_radiation[idx]
                            if weather.hourly.shortwave_radiation else None
                        )
                        temp = (
                            weather.hourly.temperature_2m[idx]
                            if weather.hourly.temperature_2m else None
                        )
                        cloud = (
                            weather.hourly.cloud_cover[idx]
                            if weather.hourly.cloud_cover else None
                        )
                        wind = (
                            weather.hourly.wind_speed_10m[idx]
                            if weather.hourly.wind_speed_10m else None
                        )
                except Exception:
                    logger.warning(
                        "Open-Meteo fetch failed for (%.4f, %.4f)", lat, lon,
                        exc_info=True,
                    )

                # Persist combined record
                grid_record = GridData(
                    timestamp=now,
                    latitude=lat,
                    longitude=lon,
                    moer_value=moer_value,
                    moer_units="lbs_co2_per_mwh",
                    solar_irradiance_w_m2=solar,
                    temperature_celsius=temp,
                    cloud_cover_percent=cloud,
                    wind_speed_m_s=wind,
                    source=GridDataSource.COMBINED,
                )
                db.add(grid_record)
                records_created += 1

            await db.commit()

    summary: dict[str, object] = {
        "records_created": records_created,
        "locations_processed": len(locations),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("fetch_and_store_grid_data: Complete — %s", summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Task 2: Generate Charging Schedule
# ═══════════════════════════════════════════════════════════════════════════


@celery.task(
    name="app.tasks.generate_charging_schedule",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    queue="scheduling",
)
def generate_charging_schedule(
    self: Any,
    session_id: str,
    w_carbon: float = 1.0,
    w_price: float = 0.5,
    w_degradation: float = 2.0,
    charger_power_kw: float = 7.2,
) -> dict[str, object]:
    """
    Generate an optimized charging schedule for a given session.

    Workflow:
    1. Load the ChargingSession from the database.
    2. Fetch MOER forecast from GridData (or WattTime if insufficient history).
    3. Run the DP scheduler from ``scheduler_algorithm.py``.
    4. Persist the resulting ScheduledBlock rows.
    5. Enqueue ``execute_smartcar_command`` tasks for each state transition.

    Parameters
    ----------
    session_id : str
        UUID of the ChargingSession to optimize.
    w_carbon : float
        Carbon emissions weight.
    w_price : float
        Electricity price weight.
    w_degradation : float
        Battery degradation (switching) penalty weight.

    Returns
    -------
    dict
        Summary with total_cost, num_blocks, and scheduled slot count.
    """
    logger.info(
        "generate_charging_schedule: session=%s weights=(%.2f, %.2f, %.2f)",
        session_id, w_carbon, w_price, w_degradation,
    )

    try:
        result = _run_async(
            _generate_schedule_async(session_id, w_carbon, w_price, w_degradation, charger_power_kw)
        )
        return result
    except ValueError as ve:
        logger.error("generate_charging_schedule: Infeasible request for session %s: %s", session_id, ve)
        _run_async(_mark_session_failed(session_id))
        raise ve
    except Exception as exc:
        logger.exception("generate_charging_schedule: Failed for session %s", session_id)
        # Mark session as failed before retrying
        _run_async(_mark_session_failed(session_id))
        raise self.retry(exc=exc)


async def _generate_schedule_async(
    session_id: str,
    w_carbon: float,
    w_price: float,
    w_degradation: float,
    charger_power_kw: float = 7.2,
) -> dict[str, object]:
    """Async implementation of schedule generation."""
    from sqlalchemy import delete, select

    from app.database import AsyncSessionLocal
    from app.models import (
        ChargingSession,
        GridData,
        ScheduledBlock,
        ScheduleStatus,
        SessionStatus,
    )
    import sys
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from scheduler_algorithm import (
        ChargingSchedule,
        CostWeights,
        ScheduleRequest,
        SlotAction,
        get_charging_blocks,
        schedule_charging,
    )

    async with AsyncSessionLocal() as db:
        # ── 1. Load session ─────────────────────────────────────────────────
        stmt = select(ChargingSession).where(ChargingSession.id == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"ChargingSession {session_id} not found")

        # Update status to OPTIMIZING
        session.status = SessionStatus.OPTIMIZING
        await db.commit()

        # ── 2. Build MOER forecast ──────────────────────────────────────────
        # Calculate number of 15-min slots in the session window
        window_seconds = (session.departure_time - session.arrival_time).total_seconds()
        n_slots = int(window_seconds // 900)  # 900s = 15 min

        if n_slots == 0:
            raise ValueError("Session window too short for any 15-minute slot")

        # Try to get MOER data from our GridData table
        moer_stmt = (
            select(GridData.timestamp, GridData.moer_value)
            .where(
                GridData.latitude.between(session.latitude - 0.5, session.latitude + 0.5),
                GridData.longitude.between(session.longitude - 0.5, session.longitude + 0.5),
                GridData.timestamp >= session.arrival_time,
                GridData.timestamp <= session.departure_time,
                GridData.moer_value.isnot(None),
            )
            .order_by(GridData.timestamp)
        )
        grid_result = await db.execute(moer_stmt)
        grid_rows = grid_result.all()

        if len(grid_rows) >= n_slots:
            # Use stored MOER data
            moer_forecast = [float(row.moer_value) for row in grid_rows[:n_slots]]
        else:
            # Fallback: fetch forecast from WattTime directly
            logger.info(
                "Insufficient GridData (%d/%d), fetching WattTime forecast…",
                len(grid_rows), n_slots,
            )
            moer_forecast = await _fetch_moer_forecast(
                session.latitude, session.longitude, n_slots,
            )

        # ── 3. Build price forecast (TOU placeholder) ───────────────────────
        # In production, integrate with utility rate schedules
        price_forecast: list[float] = []
        for i in range(n_slots):
            slot_time = session.arrival_time + timedelta(minutes=15 * i)
            hour = slot_time.hour
            if 16 <= hour < 21:     # Peak: 4-9 PM
                price_forecast.append(0.35)
            elif 21 <= hour or hour < 7:  # Off-peak: 9 PM - 7 AM
                price_forecast.append(0.10)
            else:                    # Shoulder
                price_forecast.append(0.22)

        # ── 4. Run DP optimizer ─────────────────────────────────────────────
        request = ScheduleRequest(
            arrival_time=session.arrival_time,
            departure_time=session.departure_time,
            target_kwh=session.target_kwh,
            starting_kwh=session.starting_kwh,
            charger_power_kw=charger_power_kw,
            moer_forecast=moer_forecast,
            price_forecast=price_forecast,
            weights=CostWeights(
                w_carbon=w_carbon,
                w_price=w_price,
                w_degradation=w_degradation,
            ),
        )

        schedule: ChargingSchedule = schedule_charging(request)

        # ── 5. Delete any existing schedule for this session ────────────────
        await db.execute(
            delete(ScheduledBlock).where(ScheduledBlock.session_id == session_id)
        )

        # ── 6. Persist ScheduledBlock rows ──────────────────────────────────
        for slot in schedule.slots:
            block = ScheduledBlock(
                session_id=session_id,
                slot_start=slot.start_time,
                slot_end=slot.end_time,
                is_charging=(slot.action == SlotAction.CHARGE),
                moer_value=slot.moer_value,
                price_value=slot.price_value,
                slot_cost=slot.slot_cost,
                switching_penalty_applied=slot.switching_penalty_applied,
                schedule_status=ScheduleStatus.READY,
            )
            db.add(block)

        # ── 7. Update session status ────────────────────────────────────────
        session.status = SessionStatus.ACTIVE
        await db.commit()

        # ── 8. Enqueue Smartcar commands for state transitions ──────────────
        blocks = get_charging_blocks(schedule)
        commands_enqueued = 0

        for block_start, block_end in blocks:
            # Schedule START_CHARGE at block start
            execute_smartcar_command.apply_async(
                kwargs={
                    "session_id": session_id,
                    "command": "start_charge",
                    "scheduled_time": block_start.isoformat(),
                },
                eta=block_start,  # execute at the exact scheduled time
                queue="commands",
            )
            # Schedule STOP_CHARGE at block end
            execute_smartcar_command.apply_async(
                kwargs={
                    "session_id": session_id,
                    "command": "stop_charge",
                    "scheduled_time": block_end.isoformat(),
                },
                eta=block_end,
                queue="commands",
            )
            commands_enqueued += 2

        logger.info(
            "generate_charging_schedule: Complete — %d slots, %d blocks, "
            "%d commands enqueued, cost=%.4f",
            len(schedule.slots), schedule.num_charging_blocks,
            commands_enqueued, schedule.total_cost,
        )

    return {
        "session_id": session_id,
        "status": "ready",
        "total_cost": round(schedule.total_cost, 4),
        "total_carbon_kg": round(schedule.total_carbon_kg, 4),
        "total_energy_kwh": round(schedule.total_energy_kwh, 2),
        "num_charging_blocks": schedule.num_charging_blocks,
        "num_transitions": schedule.num_transitions,
        "slots_total": len(schedule.slots),
        "slots_charging": schedule.slots_charged,
        "commands_enqueued": commands_enqueued,
    }


async def _fetch_moer_forecast(
    latitude: float,
    longitude: float,
    n_slots: int,
) -> list[float]:
    """Fetch MOER forecast using the ML pipeline, with fallback to WattTime."""
    import joblib
    import pandas as pd
    import math
    from app.api_clients import OpenMeteoClient, WattTimeClient
    from ml_pipeline.feature_engineering import engineer_features, get_feature_columns

    now = datetime.now(timezone.utc)
    start_time = now.replace(minute=0, second=0, microsecond=0)
    
    try:
        model = joblib.load("model_artifact.joblib")
        
        start_date_str = (start_time - timedelta(hours=24)).strftime("%Y-%m-%d")
        end_date_str = (start_time + timedelta(hours=24)).strftime("%Y-%m-%d")
        
        async with OpenMeteoClient() as om:
            weather_resp = await om.get_historical_weather(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date_str,
                end_date=end_date_str
            )
            weather_df = pd.DataFrame({
                "timestamp": weather_resp.hourly.time,
                "temperature_celsius": weather_resp.hourly.temperature_2m,
                "solar_irradiance_w_m2": weather_resp.hourly.shortwave_radiation,
                "cloud_cover_percent": weather_resp.hourly.cloud_cover,
                "wind_speed_m_s": weather_resp.hourly.wind_speed_10m,
            })

        moer_data = []
        async with WattTimeClient() as wt:
            try:
                moer_resp = await wt.get_historical(
                    latitude=latitude,
                    longitude=longitude,
                    start_time=start_time - timedelta(hours=24),
                    end_time=start_time,
                    region="CAISO_NORTH",
                )
                moer_data = [{"timestamp": d.point_time, "moer_value": d.value} for d in moer_resp.data]
            except Exception as e:
                logger.warning(f"Failed to fetch WattTime history for inference: {e}")
                
        if not moer_data:
            for t in weather_resp.hourly.time:
                t_utc = t.replace(tzinfo=timezone.utc)
                if t_utc > start_time:
                    break
                hour = t.hour
                curve = math.cos((hour - 3) * math.pi / 12)
                moer = 600 + (300 * curve) + ((hour % 3) * 15 - 15)
                moer_data.append({"timestamp": t_utc, "moer_value": moer})
                
        moer_df = pd.DataFrame(moer_data)

        df = engineer_features(weather_df, moer_df)
        feature_cols = get_feature_columns(df)
        
        current_state = df[df["timestamp"] == start_time].iloc[0:1]
        preds = model.predict(current_state)
        
        # Predictions are hourly, we need n_slots (15-min intervals)
        slots = []
        preds_array = preds.to_numpy() if hasattr(preds, "to_numpy") else preds
        for i in range(n_slots):
            hour_offset = i // 4
            # If request exceeds 24 hours, just cap it at the 24th hour prediction
            idx = min(hour_offset, 23)
            slots.append(float(preds_array[0, idx]))
        
        return slots
        
    except Exception as exc:
        logger.warning(f"ML Pipeline forecast failed, falling back to WattTime: {exc}", exc_info=True)
        try:
            async with WattTimeClient() as wt:
                forecast = await wt.get_forecast(
                    latitude=latitude,
                    longitude=longitude,
                    horizon_hours=max(24, (n_slots * 15) // 60 + 1),
                    region="CAISO_NORTH",
                )
                if forecast.data and len(forecast.data) >= n_slots:
                    return [dp.value for dp in forecast.data[:n_slots]]
        except Exception:
            logger.warning("WattTime forecast fallback failed", exc_info=True)

    # Ultimate fallback: use a moderate default MOER
    logger.warning("Using default MOER=600 for all %d slots", n_slots)
    return [600.0] * n_slots


async def _mark_session_failed(session_id: str) -> None:
    """Mark a session as FAILED after an unrecoverable error."""
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.models import ChargingSession, SessionStatus

    try:
        async with AsyncSessionLocal() as db:
            stmt = select(ChargingSession).where(ChargingSession.id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
            if session:
                session.status = SessionStatus.FAILED
                await db.commit()
    except Exception:
        logger.exception("Failed to mark session %s as FAILED", session_id)


# ═══════════════════════════════════════════════════════════════════════════
#  Task 3: Execute Smartcar Command
# ═══════════════════════════════════════════════════════════════════════════


@celery.task(
    name="app.tasks.execute_smartcar_command",
    bind=True,
    max_retries=5,
    default_retry_delay=30,
    acks_late=True,
    queue="commands",
)
def execute_smartcar_command(
    self: Any,
    session_id: str,
    command: str,
    scheduled_time: str,
) -> dict[str, object]:
    """
    Execute a physical charge start/stop command via the Smartcar API.

    This task is enqueued with an `eta` (estimated time of arrival) equal
    to the scheduled timestamp, so Celery delivers it to the worker at
    precisely the right moment.

    Parameters
    ----------
    session_id : str
        UUID of the ChargingSession.
    command : str
        Either "start_charge" or "stop_charge".
    scheduled_time : str
        ISO-format timestamp of when this command was supposed to fire.

    Returns
    -------
    dict
        Command execution result with status and timestamp.
    """
    logger.info(
        "execute_smartcar_command: session=%s command=%s scheduled=%s",
        session_id, command, scheduled_time,
    )

    try:
        result = _run_async(
            _execute_smartcar_command_async(
                session_id, command, scheduled_time, self.request.id,
            )
        )
        return result
    except Exception as exc:
        logger.exception(
            "execute_smartcar_command: Failed for session %s, command %s",
            session_id, command,
        )
        raise self.retry(exc=exc)


async def _execute_smartcar_command_async(
    session_id: str,
    command: str,
    scheduled_time: str,
    celery_task_id: str | None,
) -> dict[str, object]:
    """Async implementation of Smartcar command execution."""
    import httpx
    from cryptography.fernet import Fernet
    from sqlalchemy import select, update

    from app.config import get_settings
    from app.database import AsyncSessionLocal
    from app.models import (
        ChargingSession,
        ScheduledBlock,
        User,
    )

    settings = get_settings()
    now = datetime.now(timezone.utc)
    scheduled_dt = datetime.fromisoformat(scheduled_time)

    async with AsyncSessionLocal() as db:
        # ── 1. Load session + user (for Smartcar tokens) ────────────────────
        session_stmt = (
            select(ChargingSession)
            .where(ChargingSession.id == session_id)
        )
        result = await db.execute(session_stmt)
        session = result.scalar_one_or_none()

        if session is None:
            raise ValueError(f"ChargingSession {session_id} not found")

        user_stmt = select(User).where(User.id == session.user_id)
        user_result = await db.execute(user_stmt)
        user = user_result.scalar_one_or_none()

        if user is None:
            raise ValueError(f"User {session.user_id} not found")

        # ── 2. Build Smartcar API call ──────────────────────────────────────
        vehicle_id = session.vehicle_id or user.smartcar_vehicle_id
        encrypted_access_token = user.smartcar_access_token

        if not vehicle_id or not encrypted_access_token:
            logger.warning(
                "Smartcar credentials missing for session %s — "
                "recording command without sending.",
                session_id,
            )
            api_response: dict[str, object] = {
                "status": "skipped",
                "reason": "missing_credentials",
            }
        else:
            # ── 3. Decrypt token and execute the Smartcar API call ──────────
            try:
                f = Fernet(settings.fernet_key.encode())
                access_token = f.decrypt(encrypted_access_token.encode()).decode()
            except Exception as decrypt_err:
                logger.error("Failed to decrypt Smartcar token: %s", decrypt_err)
                api_response = {
                    "status": "error",
                    "error": "token_decryption_failed",
                }
                # Still mark the block as attempted
                access_token = None

            if access_token:
                smartcar_action = "START" if command == "start_charge" else "STOP"
                smartcar_url = f"https://api.smartcar.com/v2.0/vehicles/{vehicle_id}/charge"

                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            smartcar_url,
                            json={"action": smartcar_action},
                            headers={
                                "Authorization": f"Bearer {access_token}",
                                "Content-Type": "application/json",
                            },
                        )
                        api_response = {
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                        }

                        if response.status_code != 200:
                            logger.warning(
                                "Smartcar API returned %d for %s",
                                response.status_code, command,
                            )

                except httpx.HTTPError as http_err:
                    api_response = {
                        "status": "error",
                        "error": str(http_err),
                    }
                    logger.error("Smartcar HTTP error: %s", http_err)

        # ── 4. Mark the ScheduledBlock as command_sent ──────────────────────
        block_update_stmt = (
            update(ScheduledBlock)
            .where(
                ScheduledBlock.session_id == session_id,
                ScheduledBlock.slot_start == scheduled_dt,
            )
            .values(
                command_sent=True,
                command_sent_at=now,
                command_celery_task_id=celery_task_id,
            )
        )
        await db.execute(block_update_stmt)
        await db.commit()

    summary: dict[str, object] = {
        "session_id": session_id,
        "command": command,
        "scheduled_time": scheduled_time,
        "executed_at": now.isoformat(),
        "delay_seconds": round((now - scheduled_dt).total_seconds(), 2),
        "api_response": api_response,
    }

    logger.info(
        "execute_smartcar_command: Complete — %s %s (delay: %.2fs)",
        command, session_id, summary["delay_seconds"],
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Synchronous Execution (for deployment without Redis/Celery)
# ═══════════════════════════════════════════════════════════════════════════


async def _run_schedule_sync(
    session_id: str,
    w_carbon: float = 1.0,
    w_price: float = 0.5,
    w_degradation: float = 2.0,
    charger_power_kw: float = 7.2,
) -> dict:
    """
    Run schedule generation synchronously (called from FastAPI when Redis is unavailable).
    This is the same logic as the Celery task but runs inline.
    """
    logger.info("_run_schedule_sync: Running inline for session=%s", session_id)
    try:
        result = await _generate_schedule_async(
            session_id, w_carbon, w_price, w_degradation, charger_power_kw
        )
        return result
    except Exception as e:
        logger.exception("_run_schedule_sync: Failed for session %s", session_id)
        await _mark_session_failed(session_id)
        raise
