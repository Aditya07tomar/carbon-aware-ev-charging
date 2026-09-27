"""
FastAPI application factory and entrypoint.

Configures:
    • Lifespan context manager (DB init / cleanup)
    • CORS middleware
    • Health-check endpoint
    • /api/v1/vehicles — list vehicles (real Smartcar or demo fallback)
    • /api/v1/carbon-forecast — ML-based MOER forecast
    • /api/v1/auth/smartcar — Smartcar OAuth redirect
    • /auth/smartcar/callback — Smartcar OAuth callback
    • /schedule/generate — enqueues DP charging schedule optimization
    • /schedule/status/{task_id} — polls Celery task status
    • /schedule/{session_id} — retrieves generated schedule from DB
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, DbSession, dispose_engine, get_db, init_db
from app.models import (
    ChargingSession,
    ScheduledBlock,
    ScheduleStatus,
    SessionStatus,
    User,
)
from app.schemas import (
    HealthCheckResponse,
    ScheduleGenerateRequest,
    ScheduleGenerateResponse,
    ScheduledBlockResponse,
    ScheduleResultResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup:
        • Initialize database tables (dev only — use Alembic in prod)
        • Log configuration summary

    Shutdown:
        • Dispose SQLAlchemy engine (close all pooled connections)
    """
    # ── Startup ──
    logger.info("Starting Carbon-Aware EV Charging API [%s]", settings.app_env)

    if settings.app_env == "development":
        await init_db()
        logger.info("Database tables created (development mode)")

    yield

    # ── Shutdown ──
    await dispose_engine()
    logger.info("Database connections closed. Goodbye.")


# ── Application Factory ────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="Carbon-Aware EV Charging API",
        description=(
            "Optimizes electric vehicle charging schedules to minimize "
            "marginal carbon emissions using real-time grid data."
        ),
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ═══════════════════════════════════════════════════════════════════════
    #  System Endpoints
    # ═══════════════════════════════════════════════════════════════════════

    @application.get(
        "/health",
        response_model=HealthCheckResponse,
        tags=["system"],
        summary="Service health check",
    )
    async def health_check(db: DbSession) -> dict[str, Any]:
        """Verify that the API, database, and Redis are reachable."""
        db_status = "connected"
        try:
            await db.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        except Exception:
            db_status = "unavailable"

        redis_status = "connected"
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
            r.ping()
        except Exception:
            redis_status = "unavailable"

        return {
            "status": "ok",
            "environment": settings.app_env,
            "database": db_status,
            "redis": redis_status,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Vehicle Endpoints
    # ═══════════════════════════════════════════════════════════════════════

    @application.get("/api/v1/vehicles", tags=["vehicles"])
    async def get_vehicles(db: AsyncSession = Depends(get_db)):
        """
        Fetch vehicles. If a user has Smartcar tokens, use the real API.
        Otherwise return demo vehicles so the UI is always usable.
        """
        import httpx
        from cryptography.fernet import Fernet

        # Look for any user with Smartcar tokens
        stmt = select(User).where(User.smartcar_access_token.isnot(None))
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user and user.smartcar_access_token:
            try:
                f = Fernet(settings.fernet_key.encode())
                access_token = f.decrypt(user.smartcar_access_token.encode()).decode()

                async with httpx.AsyncClient(timeout=15.0) as client:
                    # Get vehicle IDs
                    vehicles_resp = await client.get(
                        "https://api.smartcar.com/v2.0/vehicles",
                        headers={"Authorization": f"Bearer {access_token}"},
                    )

                    if vehicles_resp.status_code == 200:
                        vehicle_ids = vehicles_resp.json().get("vehicles", [])
                        real_vehicles = []

                        for vid in vehicle_ids:
                            # Get vehicle info
                            info_resp = await client.get(
                                f"https://api.smartcar.com/v2.0/vehicles/{vid}",
                                headers={"Authorization": f"Bearer {access_token}"},
                            )
                            # Get battery
                            batt_resp = await client.get(
                                f"https://api.smartcar.com/v2.0/vehicles/{vid}/battery",
                                headers={"Authorization": f"Bearer {access_token}"},
                            )

                            info = info_resp.json() if info_resp.status_code == 200 else {}
                            batt = batt_resp.json() if batt_resp.status_code == 200 else {}

                            # percentRemaining is 0-1 float
                            pct = batt.get("percentRemaining", 0.5)
                            battery_level = int(pct * 100) if pct <= 1.0 else int(pct)

                            real_vehicles.append({
                                "id": vid,
                                "make": info.get("make", "Unknown"),
                                "model": info.get("model", "Vehicle"),
                                "year": info.get("year", 2024),
                                "battery_level": battery_level,
                                "battery_capacity_kwh": batt.get("capacity", 60.0),
                                "is_charging": False,
                            })

                        if real_vehicles:
                            return {
                                "demo_mode": False,
                                "vehicles": real_vehicles,
                            }
                    elif vehicles_resp.status_code == 401:
                        # Token expired, try refresh
                        logger.warning("Smartcar access token expired, attempting refresh...")
                        refreshed = await _refresh_smartcar_token(user, db)
                        if not refreshed:
                            logger.warning("Token refresh failed, falling back to demo.")

            except Exception as e:
                logger.error(f"Error fetching real vehicles: {e}")

        # Fallback to demo data
        return {
            "demo_mode": True,
            "vehicles": [
                {
                    "id": "demo-vehicle-001",
                    "make": "Tesla",
                    "model": "Model 3",
                    "year": 2024,
                    "battery_level": 42,
                    "battery_capacity_kwh": 60,
                    "is_charging": False,
                },
                {
                    "id": "demo-vehicle-002",
                    "make": "Chevrolet",
                    "model": "Bolt EUV",
                    "year": 2023,
                    "battery_level": 67,
                    "battery_capacity_kwh": 65,
                    "is_charging": True,
                },
                {
                    "id": "demo-vehicle-003",
                    "make": "Hyundai",
                    "model": "Ioniq 5",
                    "year": 2023,
                    "battery_level": 25,
                    "battery_capacity_kwh": 77.4,
                    "is_charging": False,
                },
                {
                    "id": "demo-vehicle-004",
                    "make": "Ford",
                    "model": "Mustang Mach-E",
                    "year": 2022,
                    "battery_level": 85,
                    "battery_capacity_kwh": 91,
                    "is_charging": False,
                },
                {
                    "id": "demo-vehicle-005",
                    "make": "Rivian",
                    "model": "R1T",
                    "year": 2024,
                    "battery_level": 15,
                    "battery_capacity_kwh": 135,
                    "is_charging": False,
                },
            ]
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  Carbon Forecast Endpoint
    # ═══════════════════════════════════════════════════════════════════════

    @application.get("/api/v1/carbon-forecast", tags=["ml"])
    async def get_carbon_forecast():
        """
        Generate a 24-hour MOER forecast using the ML pipeline.

        Strategy:
            1. Load the CarbonIntensityForecaster from model_artifact.joblib
            2. Fetch weather data from Open-Meteo (past 48h for lag features)
            3. Get current MOER from WattTime (or synthesize history)
            4. Engineer features matching the model's expected 24-column input
            5. Call predict_single() to get 24h forecast
            6. Falls back to WattTime forecast, then synthetic curve
        """
        import math
        import joblib
        import numpy as np
        import pandas as pd
        from ml_pipeline.feature_engineering import engineer_features
        from app.api_clients import OpenMeteoClient, WattTimeClient

        now = datetime.now(timezone.utc)
        start_time = now.replace(minute=0, second=0, microsecond=0)

        try:
            # 1. Load the trained CarbonIntensityForecaster
            forecaster = joblib.load("model_artifact.joblib")
            if not hasattr(forecaster, 'predict_single') or not forecaster.is_fitted:
                raise RuntimeError("Model artifact is not a fitted CarbonIntensityForecaster")

            # 2. Fetch weather: need 48h history for rolling/lag features
            history_start = start_time - timedelta(hours=48)
            start_date_str = history_start.strftime("%Y-%m-%d")
            end_date_str = (start_time + timedelta(hours=1)).strftime("%Y-%m-%d")

            async with OpenMeteoClient() as om:
                weather_resp = await om.get_historical_weather(
                    latitude=37.7749,
                    longitude=-122.4194,
                    start_date=start_date_str,
                    end_date=end_date_str,
                )
                weather_df = pd.DataFrame({
                    "timestamp": weather_resp.hourly.time,
                    "temperature_celsius": weather_resp.hourly.temperature_2m,
                    "solar_irradiance_w_m2": weather_resp.hourly.shortwave_radiation,
                    "cloud_cover_percent": weather_resp.hourly.cloud_cover,
                    "wind_speed_m_s": weather_resp.hourly.wind_speed_10m,
                })

            logger.info("Weather data fetched: %d rows", len(weather_df))

            # 3. Build MOER history — need 25+ rows for lag_24h feature
            moer_data = []

            # Try WattTime realtime signal
            try:
                async with WattTimeClient() as wt:
                    signal = await wt.get_realtime_signal(
                        latitude=37.7749,
                        longitude=-122.4194,
                    )
                    if signal.data:
                        current_moer = signal.data[0].value
                        logger.info("Live WattTime MOER: %.1f", current_moer)
                    else:
                        current_moer = None
            except Exception as e:
                logger.warning("WattTime realtime failed: %s", e)
                current_moer = None

            # Build synthetic MOER history to match weather timestamps
            # This gives the model enough context for rolling/lag features
            for ts in weather_resp.hourly.time:
                hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
                # Physics-based synthetic MOER curve
                diurnal = -150 * math.sin(2 * math.pi * (hour - 3) / 24)
                base = 650 + diurnal
                noise = (hash(str(ts)) % 50) - 25  # deterministic "noise"
                moer_val = max(200, base + noise)

                moer_data.append({
                    "timestamp": ts,
                    "moer_value": round(moer_val, 1),
                })

            # If we have a live MOER reading, use it for the latest timestamp
            if current_moer is not None and moer_data:
                moer_data[-1]["moer_value"] = current_moer

            moer_df = pd.DataFrame(moer_data)

            # 4. Engineer features (produces the 24 columns the model expects)
            df = engineer_features(weather_df, moer_df, drop_raw_columns=True)

            # 5. Select the latest row (current state) for prediction
            current_state = df.iloc[-1:]
            if current_state.empty:
                raise RuntimeError("Feature engineering produced empty DataFrame")

            # Ensure all expected feature columns exist
            missing_cols = set(forecaster.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning("Missing feature columns: %s — filling with 0", missing_cols)
                for col in missing_cols:
                    df[col] = 0.0
                current_state = df.iloc[-1:]

            # 6. Predict next 24 hours
            predictions = forecaster.predict_single(current_state)

            forecast = []
            for i in range(24):
                t = start_time + timedelta(hours=i + 1)
                moer_val = float(predictions[i])
                # Clamp to physically reasonable range
                moer_val = max(50.0, min(1500.0, moer_val))
                forecast.append({
                    "timestamp": t.isoformat(),
                    "moer_value": round(moer_val, 1),
                })

            source = "ml_pipeline"
            if current_moer is not None:
                source = "ml_pipeline+watttime_live"

            logger.info(
                "ML forecast generated: range [%.1f, %.1f], source=%s",
                min(f["moer_value"] for f in forecast),
                max(f["moer_value"] for f in forecast),
                source,
            )

            return {
                "status": "ok",
                "horizon_hours": 24,
                "forecast": forecast,
                "source": source,
            }

        except Exception as e:
            logger.error("ML inference failed: %s", e, exc_info=True)

            # Try WattTime forecast API as fallback
            try:
                async with WattTimeClient() as wt:
                    wt_forecast = await wt.get_forecast(
                        latitude=37.7749,
                        longitude=-122.4194,
                        horizon_hours=24,
                    )
                    if wt_forecast.data:
                        forecast = []
                        for dp in wt_forecast.data[:24]:
                            forecast.append({
                                "timestamp": dp.point_time.isoformat(),
                                "moer_value": dp.value,
                            })
                        return {
                            "status": "ok",
                            "horizon_hours": 24,
                            "forecast": forecast,
                            "source": "watttime_forecast",
                        }
            except Exception as wt_err:
                logger.warning("WattTime forecast also failed: %s", wt_err)

            # Ultimate fallback: physics-based synthetic curve
            import math
            forecast = []
            for i in range(24):
                t = start_time + timedelta(hours=i)
                hour = t.hour
                diurnal = -150 * math.sin(2 * math.pi * (hour - 3) / 24)
                moer = 650 + diurnal + ((hour % 3) * 15 - 15)
                forecast.append({
                    "timestamp": t.isoformat(),
                    "moer_value": round(moer, 1),
                })

            return {
                "status": "ok",
                "horizon_hours": 24,
                "forecast": forecast,
                "source": "synthetic_fallback",
            }

    # ═══════════════════════════════════════════════════════════════════════
    #  Smartcar Auth Endpoints
    # ═══════════════════════════════════════════════════════════════════════

    @application.get("/api/v1/auth/smartcar", tags=["auth"])
    async def auth_smartcar():
        """Redirect user to Smartcar OAuth consent screen."""
        client_id = settings.smartcar_client_id
        if not client_id or client_id == "your_smartcar_client_id":
            raise HTTPException(
                status_code=503,
                detail="Smartcar API not configured. Set SMARTCAR_CLIENT_ID in .env."
            )

        redirect_uri = settings.smartcar_redirect_uri
        scope = "read_battery read_charge read_vehicle_info control_charge"
        url = (
            f"https://connect.smartcar.com/oauth/authorize"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scope}"
            f"&mode=test"
        )
        return RedirectResponse(url=url)

    @application.get("/auth/smartcar/callback", tags=["auth"])
    async def auth_smartcar_callback(
        code: str = None,
        error: str = None,
        db: AsyncSession = Depends(get_db),
    ):
        """Handle Smartcar OAuth callback, exchange code for tokens."""
        if error:
            logger.warning("Smartcar OAuth error: %s", error)
            return RedirectResponse(url="http://localhost:3000/?smartcar_error=" + error)

        if not code:
            raise HTTPException(status_code=400, detail="Missing authorization code")

        client_id = settings.smartcar_client_id
        client_secret = settings.smartcar_client_secret
        redirect_uri = settings.smartcar_redirect_uri

        import httpx
        from cryptography.fernet import Fernet
        import base64

        auth_string = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://auth.smartcar.com/oauth/token",
                headers={
                    "Authorization": f"Basic {auth_string}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            if resp.status_code != 200:
                logger.error(
                    "Smartcar token exchange failed: %d %s",
                    resp.status_code,
                    resp.text,
                )
                raise HTTPException(
                    status_code=400,
                    detail="Smartcar authentication failed. Check redirect URI and credentials.",
                )

            token_data = resp.json()
            access_token = token_data["access_token"]
            refresh_token = token_data["refresh_token"]
            expires_in = token_data.get("expires_in", 7200)

            # Encrypt tokens with Fernet
            f = Fernet(settings.fernet_key.encode())
            enc_access = f.encrypt(access_token.encode()).decode()
            enc_refresh = f.encrypt(refresh_token.encode()).decode()
            token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            # Get vehicle ID from Smartcar
            vehicle_id = None
            try:
                vehicles_resp = await client.get(
                    "https://api.smartcar.com/v2.0/vehicles",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if vehicles_resp.status_code == 200:
                    vehicle_ids = vehicles_resp.json().get("vehicles", [])
                    if vehicle_ids:
                        vehicle_id = vehicle_ids[0]
            except Exception as e:
                logger.warning("Failed to fetch vehicle ID during callback: %s", e)

            # Upsert user — use a deterministic user ID for demo
            demo_user_id = "00000000-0000-0000-0000-000000000001"
            stmt = select(User).where(User.id == demo_user_id)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                user = User(
                    id=demo_user_id,
                    email="demo@carbonev.app",
                    name="Demo User",
                )
                db.add(user)

            user.smartcar_access_token = enc_access
            user.smartcar_refresh_token = enc_refresh
            user.smartcar_token_expiry = token_expiry
            if vehicle_id:
                user.smartcar_vehicle_id = vehicle_id
            await db.commit()

            logger.info("Smartcar tokens stored for user %s", demo_user_id)

        return RedirectResponse(url="http://localhost:3000/")

    # ═══════════════════════════════════════════════════════════════════════
    #  Schedule Endpoints
    # ═══════════════════════════════════════════════════════════════════════

    @application.post(
        "/schedule/generate",
        response_model=ScheduleGenerateResponse,
        tags=["scheduling"],
        summary="Enqueue charging schedule generation",
        status_code=202,
    )
    async def generate_schedule(
        request: ScheduleGenerateRequest,
        db: DbSession,
    ) -> ScheduleGenerateResponse:
        """
        Enqueue the DP charging schedule optimizer as a Celery task.

        Validates constraints, creates a ChargingSession, then dispatches
        the task to the `scheduling` queue. Returns the Celery task ID
        for status polling.

        **Returns 202 Accepted** — the schedule is generated asynchronously.
        Poll `/schedule/status/{task_id}` for completion.
        """
        from app.tasks import generate_charging_schedule
        from dateutil.parser import parse

        # 1. Ensure a demo user exists
        demo_user_id = "00000000-0000-0000-0000-000000000001"
        stmt = select(User).where(User.id == demo_user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=demo_user_id, email="demo@carbonev.app", name="Demo User")
            db.add(user)
            await db.commit()

        # 2. Parse constraints
        departure_dt = parse(request.departure_time)
        if departure_dt.tzinfo is None:
            departure_dt = departure_dt.replace(tzinfo=timezone.utc)

        # Determine capacity based on vehicle data or defaults
        vehicle_id = request.vehicle_id
        capacity_kwh = 60.0  # default

        # Try to get real battery data if user has Smartcar tokens
        if user.smartcar_access_token and vehicle_id and not vehicle_id.startswith("demo"):
            try:
                from cryptography.fernet import Fernet
                import httpx

                f_cipher = Fernet(settings.fernet_key.encode())
                token = f_cipher.decrypt(user.smartcar_access_token.encode()).decode()
                async with httpx.AsyncClient(timeout=10.0) as client:
                    batt_resp = await client.get(
                        f"https://api.smartcar.com/v2.0/vehicles/{vehicle_id}/battery",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if batt_resp.status_code == 200:
                        batt_data = batt_resp.json()
                        capacity_kwh = batt_data.get("capacity", 60.0) or 60.0
                        pct = batt_data.get("percentRemaining", 0.42)
                        starting_kwh = capacity_kwh * (pct if pct <= 1.0 else pct / 100.0)
                    else:
                        starting_kwh = capacity_kwh * 0.42
            except Exception as e:
                logger.warning("Failed to get real battery data: %s", e)
                starting_kwh = capacity_kwh * 0.42
        else:
            # Demo vehicles: use approximate values
            demo_capacities = {
                "demo-vehicle-001": (60.0, 0.42),
                "demo-vehicle-002": (65.0, 0.67),
                "demo-vehicle-003": (77.4, 0.25),
                "demo-vehicle-004": (91.0, 0.85),
                "demo-vehicle-005": (135.0, 0.15),
            }
            cap, soc = demo_capacities.get(vehicle_id, (60.0, 0.42))
            capacity_kwh = cap
            starting_kwh = capacity_kwh * soc

        target_kwh_absolute = capacity_kwh * (request.target_charge_percent / 100.0)
        # Energy to DELIVER = absolute target − current level
        energy_needed_kwh = max(target_kwh_absolute - starting_kwh, 0.0)

        # 3. Create the session
        session = ChargingSession(
            user_id=demo_user_id,
            vehicle_id=vehicle_id,
            arrival_time=datetime.now(timezone.utc),
            departure_time=departure_dt,
            starting_kwh=starting_kwh,
            target_kwh=energy_needed_kwh,
            latitude=37.7749,
            longitude=-122.4194,
            status=SessionStatus.PENDING,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

        # Enqueue the Celery task
        task = generate_charging_schedule.apply_async(
            kwargs={
                "session_id": session.id,
                "w_carbon": request.w_carbon,
                "w_price": request.w_price,
                "w_degradation": request.w_degradation,
                "charger_power_kw": request.charger_power_kw,
            },
            queue="scheduling",
        )

        logger.info(
            "Schedule generation enqueued: session=%s task=%s",
            session.id, task.id,
        )

        return ScheduleGenerateResponse(
            task_id=task.id,
            session_id=session.id,
            status="queued",
            message="Charging schedule generation has been enqueued.",
        )

    @application.get(
        "/schedule/status/{task_id}",
        response_model=TaskStatusResponse,
        tags=["scheduling"],
        summary="Poll Celery task status",
    )
    async def get_task_status(task_id: str) -> TaskStatusResponse:
        """
        Check the status of a schedule generation task.

        Celery states: PENDING → STARTED → SUCCESS / FAILURE / RETRY.
        """
        from app.celery_app import celery as celery_app

        result = celery_app.AsyncResult(task_id)

        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status,
        )

        if result.successful():
            response.result = result.result
        elif result.failed():
            response.error = str(result.result)

        return response

    @application.get(
        "/schedule/{session_id}",
        response_model=ScheduleResultResponse,
        tags=["scheduling"],
        summary="Retrieve generated schedule for a session",
    )
    async def get_schedule(
        session_id: str,
        db: DbSession,
    ) -> ScheduleResultResponse:
        """
        Retrieve the generated charging schedule for a session.

        Returns all ScheduledBlock rows with cost breakdown, or 404
        if no schedule has been generated yet.
        """
        # Validate UUID format
        import uuid
        try:
            uuid.UUID(session_id)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Verify session exists
        session_stmt = select(ChargingSession).where(
            ChargingSession.id == session_id
        )
        session_result = await db.execute(session_stmt)
        session = session_result.scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"ChargingSession {session_id} not found",
            )

        # Fetch scheduled blocks
        blocks_stmt = (
            select(ScheduledBlock)
            .where(ScheduledBlock.session_id == session_id)
            .order_by(ScheduledBlock.slot_start)
        )
        blocks_result = await db.execute(blocks_stmt)
        blocks = blocks_result.scalars().all()

        if not blocks:
            raise HTTPException(
                status_code=404,
                detail=f"No schedule found for session {session_id}. "
                       "Generate one via POST /schedule/generate.",
            )

        # Compute summary metrics
        charging_blocks = [b for b in blocks if b.is_charging]
        total_cost = sum(b.slot_cost for b in blocks)
        total_carbon = sum(
            (b.moer_value or 0) * (7.2 * 0.25 / 1000.0) * 0.453592
            for b in charging_blocks
        )
        total_energy = len(charging_blocks) * 7.2 * 0.25  # kWh per 15-min slot

        # Count contiguous charging blocks
        num_blocks = 0
        in_block = False
        transitions = 0
        prev_charging = None
        for b in blocks:
            if b.is_charging and not in_block:
                num_blocks += 1
                in_block = True
            elif not b.is_charging:
                in_block = False
            if prev_charging is not None and b.is_charging != prev_charging:
                transitions += 1
            prev_charging = b.is_charging

        return ScheduleResultResponse(
            session_id=session_id,
            status=blocks[0].schedule_status.value if blocks else "unknown",
            total_cost=round(total_cost, 4),
            total_carbon_kg=round(total_carbon, 4),
            total_energy_kwh=round(total_energy, 2),
            num_charging_blocks=num_blocks,
            num_transitions=transitions,
            slots=[ScheduledBlockResponse.model_validate(b) for b in blocks],
        )

    return application


# ── Helper: Refresh Smartcar Token ──────────────────────────────────────────


async def _refresh_smartcar_token(user: User, db: AsyncSession) -> bool:
    """Attempt to refresh a user's Smartcar tokens. Returns True on success."""
    import httpx
    import base64
    from cryptography.fernet import Fernet

    if not user.smartcar_refresh_token:
        return False

    try:
        f = Fernet(settings.fernet_key.encode())
        refresh_token = f.decrypt(user.smartcar_refresh_token.encode()).decode()

        client_id = settings.smartcar_client_id
        client_secret = settings.smartcar_client_secret
        auth_string = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://auth.smartcar.com/oauth/token",
                headers={
                    "Authorization": f"Basic {auth_string}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            if resp.status_code == 200:
                token_data = resp.json()
                enc_access = f.encrypt(token_data["access_token"].encode()).decode()
                enc_refresh = f.encrypt(token_data["refresh_token"].encode()).decode()
                expires_in = token_data.get("expires_in", 7200)

                user.smartcar_access_token = enc_access
                user.smartcar_refresh_token = enc_refresh
                user.smartcar_token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                await db.commit()
                logger.info("Smartcar token refreshed successfully")
                return True
            else:
                logger.warning("Smartcar token refresh failed: %d", resp.status_code)
                return False
    except Exception as e:
        logger.error("Token refresh error: %s", e)
        return False


# ── Module-level app instance (used by Uvicorn) ────────────────────────────
app: FastAPI = create_app()
