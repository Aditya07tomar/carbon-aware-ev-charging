"""
Pydantic v2 schemas for request/response validation and serialization.

Organized by domain: User, ChargingSession, GridData, and external API responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ═══════════════════════════════════════════════════════════════════════════
#  User Schemas
# ═══════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """Payload for creating a new user."""
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)


class UserUpdate(BaseModel):
    """Payload for partially updating a user."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    """Public user representation (tokens never exposed)."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    name: str
    smartcar_vehicle_id: Optional[str] = None
    has_smartcar_token: bool = False
    created_at: datetime
    updated_at: datetime


# ═══════════════════════════════════════════════════════════════════════════
#  Charging Session Schemas
# ═══════════════════════════════════════════════════════════════════════════

class ChargingSessionCreate(BaseModel):
    """Payload for creating a new charging session."""
    vehicle_id: Optional[str] = None
    arrival_time: datetime
    departure_time: datetime
    target_kwh: float = Field(..., gt=0, description="Energy goal in kWh")
    starting_kwh: float = Field(default=0.0, ge=0, description="Current charge in kWh")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class ChargingSessionUpdate(BaseModel):
    """Payload for updating session constraints."""
    departure_time: Optional[datetime] = None
    target_kwh: Optional[float] = Field(default=None, gt=0)
    status: Optional[str] = None


class ChargingSessionResponse(BaseModel):
    """Full charging session representation."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    vehicle_id: Optional[str] = None
    arrival_time: datetime
    departure_time: datetime
    target_kwh: float
    starting_kwh: float
    latitude: float
    longitude: float
    status: str
    created_at: datetime
    updated_at: datetime


# ═══════════════════════════════════════════════════════════════════════════
#  Grid Data Schemas
# ═══════════════════════════════════════════════════════════════════════════

class GridDataCreate(BaseModel):
    """Payload for inserting a grid data record."""
    timestamp: datetime
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    moer_value: Optional[float] = None
    moer_units: Optional[str] = "lbs_co2_per_mwh"
    forecast_horizon_minutes: Optional[int] = None
    solar_irradiance_w_m2: Optional[float] = None
    temperature_celsius: Optional[float] = None
    cloud_cover_percent: Optional[float] = None
    wind_speed_m_s: Optional[float] = None
    source: str = "combined"


class GridDataResponse(BaseModel):
    """Grid data record representation."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    latitude: float
    longitude: float
    moer_value: Optional[float] = None
    moer_units: Optional[str] = None
    forecast_horizon_minutes: Optional[int] = None
    solar_irradiance_w_m2: Optional[float] = None
    temperature_celsius: Optional[float] = None
    cloud_cover_percent: Optional[float] = None
    wind_speed_m_s: Optional[float] = None
    source: str
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════
#  WattTime API Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class WattTimeSignalDatum(BaseModel):
    """A single data point from the WattTime signal-index endpoint."""
    point_time: datetime
    value: float
    frequency: Optional[int] = None
    market: Optional[str] = None


class WattTimeRealtimeResponse(BaseModel):
    """Parsed response from WattTime /v3/signal-index."""
    data: list[WattTimeSignalDatum]
    meta: dict[str, object] = Field(default_factory=dict)


class WattTimeForecastDatum(BaseModel):
    """A single forecast data point from WattTime."""
    point_time: datetime
    value: float


class WattTimeForecastResponse(BaseModel):
    """Parsed response from WattTime /v3/forecast."""
    data: list[WattTimeForecastDatum]
    generated_at: Optional[datetime] = None
    meta: dict[str, object] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Open-Meteo API Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class OpenMeteoHourlyData(BaseModel):
    """Hourly weather data from Open-Meteo."""
    time: list[datetime]
    temperature_2m: list[Optional[float]] = Field(default_factory=list)
    shortwave_radiation: list[Optional[float]] = Field(default_factory=list)
    cloud_cover: list[Optional[float]] = Field(default_factory=list)
    wind_speed_10m: list[Optional[float]] = Field(default_factory=list)


class OpenMeteoWeatherResponse(BaseModel):
    """Parsed response from Open-Meteo /v1/forecast."""
    latitude: float
    longitude: float
    timezone: str = "UTC"
    hourly: OpenMeteoHourlyData


# ═══════════════════════════════════════════════════════════════════════════
#  Generic / Utility Schemas
# ═══════════════════════════════════════════════════════════════════════════

class HealthCheckResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str = "ok"
    environment: str
    database: str = "connected"
    redis: str = "connected"


# ═══════════════════════════════════════════════════════════════════════════
#  Schedule Generation Schemas
# ═══════════════════════════════════════════════════════════════════════════

class ScheduleGenerateRequest(BaseModel):
    """Payload for the /schedule/generate endpoint."""
    vehicle_id: str = Field(..., description="UUID of the vehicle to optimize")
    departure_time: str = Field(..., description="ISO 8601 departure timestamp")
    target_charge_percent: float = Field(..., description="Target charge % (50-100)")
    charger_power_kw: float = Field(..., description="Charger output in kW")
    w_carbon: float = Field(default=1.0, ge=0.0, description="Carbon emissions weight")
    w_price: float = Field(default=0.5, ge=0.0, description="Electricity price weight")
    w_degradation: float = Field(default=2.0, ge=0.0, description="Battery degradation penalty weight")


class ScheduleGenerateResponse(BaseModel):
    """Response from the /schedule/generate endpoint — returns the Celery task ID."""
    task_id: str
    session_id: str
    status: str = "queued"
    message: str = "Charging schedule generation has been enqueued."


class TaskStatusResponse(BaseModel):
    """Response for polling a Celery task's status."""
    task_id: str
    status: str
    result: Optional[dict[str, object]] = None
    error: Optional[str] = None


class ScheduledBlockResponse(BaseModel):
    """Public representation of a single scheduled slot."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    slot_start: datetime
    slot_end: datetime
    is_charging: bool
    moer_value: Optional[float] = None
    price_value: Optional[float] = None
    slot_cost: float
    switching_penalty_applied: bool
    command_sent: bool


class ScheduleResultResponse(BaseModel):
    """Full schedule result for a session."""
    session_id: str
    status: str
    total_cost: Optional[float] = None
    total_carbon_kg: Optional[float] = None
    total_energy_kwh: Optional[float] = None
    num_charging_blocks: Optional[int] = None
    num_transitions: Optional[int] = None
    slots: list[ScheduledBlockResponse] = Field(default_factory=list)
