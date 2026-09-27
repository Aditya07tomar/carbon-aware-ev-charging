"""
Celery worker entrypoint.

Usage:
    celery -A celery_worker:celery worker --loglevel=info --concurrency=4
    celery -A celery_worker:celery beat --loglevel=info

This module configures the Celery application with the Redis broker,
task routing, and concurrency settings suitable for handling
concurrent EV charging workloads.
"""

from app.celery_app import celery  # noqa: F401 — re-export for CLI discovery

__all__ = ["celery"]
