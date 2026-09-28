#!/bin/bash
# start.sh
# Starts both the Celery worker and the FastAPI web server in the same Render instance.

echo "Starting Celery worker in the background..."
celery -A app.celery_app worker --loglevel=info &

echo "Starting FastAPI web server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT
