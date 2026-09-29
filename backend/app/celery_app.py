"""
Celery application factory for the Carbon-Aware EV Charging system.

Configured with:
    • Redis as broker and result backend
    • JSON serialization with UTC timezone
    • Task routing: grid data → default queue, scheduling → schedule queue,
      Smartcar commands → commands queue
    • Celery Beat schedule: grid data fetch every 15 minutes
    • Late acks and task rejection on worker loss for reliability
"""

import ssl

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery = Celery(
    "carbon_ev_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

# ── Celery Configuration ───────────────────────────────────────────────────
ssl_conf = {"ssl_cert_reqs": ssl.CERT_REQUIRED} if settings.redis_url.startswith("rediss://") else None

celery.conf.update(
    broker_use_ssl=ssl_conf,
    redis_backend_use_ssl=ssl_conf,

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Result expiry (24 hours)
    result_expires=86400,

    # Worker settings
    worker_prefetch_multiplier=1,       # fair scheduling across concurrent EVs
    worker_max_tasks_per_child=200,     # prevent memory leaks from long-running workers
    worker_max_memory_per_child=512000, # 512 MB memory cap per worker child

    # Task settings
    task_acks_late=True,                # re-deliver on worker crash
    task_reject_on_worker_lost=True,    # requeue if worker dies mid-task
    task_track_started=True,            # track STARTED state for status polling

    # Task routing — separate queues for isolation and priority
    task_routes={
        "app.tasks.fetch_and_store_grid_data": {"queue": "grid_data"},
        "app.tasks.generate_charging_schedule": {"queue": "scheduling"},
        "app.tasks.execute_smartcar_command":   {"queue": "commands"},
    },

    # Rate limiting for external API calls
    task_annotations={
        "app.tasks.fetch_and_store_grid_data": {"rate_limit": "4/m"},
        "app.tasks.execute_smartcar_command":   {"rate_limit": "10/m"},
    },

    # Periodic tasks (Celery Beat schedule)
    beat_schedule={
        "fetch-grid-data-every-15-minutes": {
            "task": "app.tasks.fetch_and_store_grid_data",
            "schedule": 900.0,  # 15 minutes = 900 seconds
            "kwargs": {},
            "options": {"queue": "grid_data"},
        },
    },
)
