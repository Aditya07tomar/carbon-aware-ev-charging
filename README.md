# ⚡ Carbon-Aware EV Charging Optimizer

An intelligent EV charging orchestration platform that shifts charging loads to times when the electric grid is cleanest and cheapest — powered by real-time carbon emissions data, weather-driven ML forecasting, and dynamic programming optimization.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (React/Vite)                 │
│  Vehicle selector · Schedule config · Timeline viewer    │
└────────────────────────┬─────────────────────────────────┘
                         │ REST API
┌────────────────────────▼─────────────────────────────────┐
│                 Backend (FastAPI + Celery)                │
│                                                          │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ WattTime  │  │  Open-Meteo  │  │    Smartcar      │  │
│  │ MOER API  │  │  Weather API │  │   Vehicle API    │  │
│  └─────┬─────┘  └──────┬───────┘  └────────┬─────────┘  │
│        │               │                    │            │
│  ┌─────▼───────────────▼────────────────────▼─────────┐  │
│  │              ML Pipeline                           │  │
│  │  HistGradientBoosting · 24 direct-horizon models   │  │
│  │  Feature Engineering: temporal + weather + MOER    │  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────▼─────────────────────────────┐  │
│  │         Dynamic Programming Optimizer              │  │
│  │   min: w₁·Carbon + w₂·Price + w₃·Degradation     │  │
│  │   State: (slot, charged_count, was_charging)       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  PostgreSQL ◄──── SQLAlchemy (asyncpg) ────► Redis       │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── backend/                  # Python backend
│   ├── app/
│   │   ├── main.py           # FastAPI application & endpoints
│   │   ├── tasks.py          # Celery tasks (schedule generation)
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── schemas.py        # Pydantic request/response schemas
│   │   ├── api_clients.py    # WattTime & Open-Meteo API clients
│   │   ├── config.py         # Settings management
│   │   ├── database.py       # Async DB engine & session factory
│   │   └── celery_app.py     # Celery configuration
│   ├── ml_pipeline/
│   │   ├── forecaster.py     # CarbonIntensityForecaster model
│   │   ├── feature_engineering.py  # 24-feature pipeline
│   │   └── evaluation.py     # Model evaluation metrics
│   ├── scheduler_algorithm.py # DP optimizer
│   ├── train_model.py        # ML training script
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx           # Main application
│   │   └── components/       # UI components
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
└── README.md
```

## Features

- **Dynamic Schedule Generation** — DP optimizer finds globally-optimal charging windows
- **Real-time Carbon Data** — WattTime MOER API for live grid emissions
- **ML Forecasting** — 24-horizon HistGradientBoosting model predicts carbon intensity
- **Weather Integration** — Open-Meteo API feeds temperature, solar, cloud, wind data
- **Vehicle Telematics** — Smartcar API for real battery SoC (with demo fallback)
- **Async Processing** — Celery + Redis for non-blocking schedule computation
- **Battery Health** — Switching penalties protect EV contactor hardware

## ML Model

The `CarbonIntensityForecaster` uses a **direct multi-step forecasting** strategy:

- **24 separate models**, one per forecast horizon (h+1 to h+24)
- **Algorithm**: `HistGradientBoostingRegressor` (scikit-learn, LightGBM-inspired)
- **24 engineered features**: temporal encodings, weather rolling/lag/delta, MOER autoregressive lags, solar×cloud interactions
- **Training**: 90-day dataset with realistic physical relationships

## Quick Start

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
python train_model.py  # train the ML model
uvicorn app.main:app --reload --port 8000
# In another terminal:
celery -A app.celery_app worker --pool=solo --loglevel=info
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Keys Required

| Service | Purpose | Get it at |
|---|---|---|
| WattTime | Carbon emissions data | [watttime.org](https://watttime.org) |
| Smartcar | Vehicle telematics (optional) | [smartcar.com](https://smartcar.com) |
| Open-Meteo | Weather forecasts | Free, no key needed |

## License

MIT
