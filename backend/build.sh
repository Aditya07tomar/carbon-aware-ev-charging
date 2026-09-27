#!/bin/bash
# Build script for Render deployment
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Training ML model (this takes ~30 seconds)..."
cd /opt/render/project/src/backend
python train_model.py

echo "==> Build complete!"
