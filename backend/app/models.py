"""
SQLAlchemy 2.0 ORM models for the Carbon-Aware EV Charging system.

Models:
    - User: profile + encrypted Smartcar OAuth tokens
    - ChargingSession: per-session constraints and location
    - GridData: time-series MOER + weather observations
    - ScheduledBlock: optimizer output — charge/idle slots with cost breakdown
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ── Base ────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Declarative base shared by all models."""
    pass


# ── Enums ───────────────────────────────────────────────────────────────────
class SessionStatus(str, enum.Enum):
    """Lifecycle states for a charging session."""
    PENDING = "pending"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GridDataSource(str, enum.Enum):
    """Origin of a grid data record."""
    WATTTIME = "watttime"
    OPEN_METEO = "open_meteo"
    COMBINED = "combined"


class ScheduleStatus(str, enum.Enum):
    """Lifecycle states for a generated charging schedule."""
    GENERATING = "generating"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class SmartcarCommandType(str, enum.Enum):
    """Types of Smartcar control commands."""
    START_CHARGE = "start_charge"
    STOP_CHARGE = "stop_charge"


# ── User ────────────────────────────────────────────────────────────────────
class User(Base):
    """
    Stores user profile information and Smartcar OAuth credentials.

    Tokens are stored as Fernet-encrypted ciphertext so they are never
    persisted in plaintext.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Smartcar OAuth tokens (Fernet-encrypted)
    smartcar_access_token: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    smartcar_refresh_token: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    smartcar_token_expiry: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Vehicle metadata (populated after first Smartcar sync)
    smartcar_vehicle_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    charging_sessions: Mapped[list[ChargingSession]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r}>"


# ── ChargingSession ────────────────────────────────────────────────────────
class ChargingSession(Base):
    """
    Records the constraints and state for a single EV charging session.

    arrival_time / departure_time define the flexibility window.
    target_kwh / starting_kwh define the energy goal.
    latitude / longitude allow grid-region lookups against WattTime.
    """

    __tablename__ = "charging_sessions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Vehicle reference
    vehicle_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    # Session constraints
    arrival_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    departure_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    target_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    starting_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Location (for grid-region / weather lookups)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    # Status tracking
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status", create_constraint=True),
        nullable=False,
        default=SessionStatus.PENDING,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship(back_populates="charging_sessions")

    def __repr__(self) -> str:
        return (
            f"<ChargingSession id={self.id!r} status={self.status.value!r} "
            f"target_kwh={self.target_kwh}>"
        )


# ── GridData ────────────────────────────────────────────────────────────────
class GridData(Base):
    """
    Time-series table storing marginal carbon emissions (MOER from WattTime)
    and localized weather observations (from Open-Meteo).

    Designed for fast range queries on (timestamp, latitude, longitude).
    """

    __tablename__ = "grid_data"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Temporal key
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )

    # Spatial key
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    # WattTime MOER data
    moer_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moer_units: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default="lbs_co2_per_mwh",
    )
    forecast_horizon_minutes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )

    # Open-Meteo weather data
    solar_irradiance_w_m2: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    temperature_celsius: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    cloud_cover_percent: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    wind_speed_m_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )

    # Metadata
    source: Mapped[GridDataSource] = mapped_column(
        Enum(GridDataSource, name="grid_data_source", create_constraint=True),
        nullable=False,
        default=GridDataSource.COMBINED,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # ── Composite Indexes ───────────────────────────────────────────────────
    __table_args__ = (
        Index(
            "ix_grid_data_timestamp_location",
            "timestamp", "latitude", "longitude",
        ),
        Index(
            "ix_grid_data_source_timestamp",
            "source", "timestamp",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<GridData ts={self.timestamp!r} "
            f"moer={self.moer_value} source={self.source.value!r}>"
        )


# ── ScheduledBlock ──────────────────────────────────────────────────────────
class ScheduledBlock(Base):
    """
    Stores the optimizer's output: individual charge/idle slots
    for a given ChargingSession.

    Created by the `generate_charging_schedule` Celery task after
    running the DP optimizer.  Read by `execute_smartcar_command`
    to trigger physical charge start/stop at the correct times.
    """

    __tablename__ = "scheduled_blocks"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("charging_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Slot timing
    slot_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    slot_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    # Action: True = charge, False = idle
    is_charging: Mapped[bool] = mapped_column(
        Boolean, nullable=False,
    )

    # Cost breakdown for this slot
    moer_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slot_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    switching_penalty_applied: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )

    # Execution tracking
    command_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    command_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    command_celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    # Schedule-level metadata (denormalized for query convenience)
    schedule_status: Mapped[ScheduleStatus] = mapped_column(
        Enum(ScheduleStatus, name="schedule_status", create_constraint=True),
        nullable=False,
        default=ScheduleStatus.READY,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # ── Indexes ─────────────────────────────────────────────────────────────
    __table_args__ = (
        Index("ix_scheduled_blocks_session_start", "session_id", "slot_start"),
        Index("ix_scheduled_blocks_pending_commands", "is_charging", "command_sent", "slot_start"),
    )

    def __repr__(self) -> str:
        action = "CHARGE" if self.is_charging else "IDLE"
        return f"<ScheduledBlock {action} {self.slot_start} → {self.slot_end}>"
