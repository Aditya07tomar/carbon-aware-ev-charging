"""
Application configuration loaded from environment variables.

Uses Pydantic BaseSettings for strict typing, validation, and .env file support.
"""

from functools import lru_cache
from typing import Literal

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
    secret_key: str = Field(..., min_length=16)

    # ── PostgreSQL ──────────────────────────────────────────
    postgres_user: str = "carbon_ev_user"
    postgres_password: str = Field(default="", min_length=0)
    postgres_db: str = "carbon_ev_db"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

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
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800  # 30 minutes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Construct the async PostgreSQL DSN."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        """Construct the sync PostgreSQL DSN (used by Alembic migrations)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]
