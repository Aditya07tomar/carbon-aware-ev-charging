"""
Application configuration loaded from environment variables.

Uses Pydantic BaseSettings for strict typing, validation, and .env file support.
"""

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = Field(default="dev-secret-key-change-in-prod")

    # ── Direct database URL (takes priority if set) ─────────
    database_url_raw: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # ── PostgreSQL (fallback when DATABASE_URL is not set) ──
    postgres_user: str = "carbon_ev_user"
    postgres_password: str = Field(default="", min_length=0)
    postgres_db: str = "carbon_ev_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Frontend URL (for CORS in production) ───────────────
    frontend_url: str = ""

    # ── WattTime API ────────────────────────────────────────
    watttime_username: str = ""
    watttime_password: str = ""

    # ── Smartcar API ────────────────────────────────────────
    smartcar_client_id: str = ""
    smartcar_client_secret: str = ""
    smartcar_redirect_uri: str = "http://localhost:8000/auth/smartcar/callback"

    # ── Encryption ──────────────────────────────────────────
    fernet_key: str = Field(default="", description="Fernet symmetric encryption key for token storage")

    # ── Database Connection Pool ────────────────────────────
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 300

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Construct the async PostgreSQL DSN."""
        if self.database_url_raw:
            url = self.database_url_raw
            # Supabase gives postgres:// but asyncpg needs postgresql+asyncpg://
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif not url.startswith("postgresql+asyncpg://"):
                url = "postgresql+asyncpg://" + url
            return url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        """Construct the sync PostgreSQL DSN (used by Alembic migrations)."""
        if self.database_url_raw:
            url = self.database_url_raw
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]

