"""
Async SQLAlchemy engine, session factory, and FastAPI dependency.

Provides:
- `async_engine`: connection-pooled async engine (asyncpg)
- `AsyncSessionLocal`: session factory bound to the engine
- `get_db()`: async generator for FastAPI Depends()
- `init_db()` / `dispose_engine()`: lifecycle helpers
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

settings = get_settings()

# ── Async Engine ────────────────────────────────────────────────────────────
async_engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
    pool_pre_ping=True,  # verify connections before checkout
)

# ── Session Factory ─────────────────────────────────────────────────────────
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ── FastAPI Dependency ──────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for the lifetime of a single request.

    Usage:
        @router.get("/example")
        async def example(db: DbSession):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Type alias for cleaner route signatures
DbSession = Annotated[AsyncSession, Depends(get_db)]


# ── Lifecycle Helpers ───────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all tables defined in the ORM metadata.

    Call during application startup for development convenience.
    In production, use Alembic migrations instead.
    """
    from app.models import Base  # deferred to avoid circular imports

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Gracefully close all pooled connections."""
    await async_engine.dispose()
