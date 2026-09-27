"""
ML Pipeline Training Script — Trains the CarbonIntensityForecaster.

Fetches weather + MOER data, engineers features, trains the 24-horizon
direct-forecasting model, evaluates it, and saves the artifact.

Usage:
    python train_model.py
"""

import asyncio
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Location: Pasadena, CA (near Caltech, CAISO grid)
LATITUDE = 37.7749
LONGITUDE = -122.4194


def generate_synthetic_dataset(n_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic synthetic dataset for training when real historical
    data is unavailable (WattTime free tier has limited historical access).

    The synthetic data models real physical relationships:
    - MOER follows a diurnal curve (high at night when gas plants dominate,
      low midday when solar floods the grid)
    - Temperature affects HVAC load -> grid demand -> MOER
    - Solar irradiance inversely correlates with MOER (displaces fossil gen)
    - Cloud cover reduces effective solar -> raises MOER
    - Wind contributes to renewable generation
    - Weekend vs weekday demand patterns differ
    """
    rng = np.random.default_rng(42)
    hours = n_days * 24
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=h) for h in range(hours)]

    weather_rows = []
    moer_rows = []

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        is_weekend = ts.weekday() >= 5

        # ── Temperature (C): seasonal + diurnal ──
        seasonal_temp = 15 + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        diurnal_temp = 5 * math.sin(2 * math.pi * (hour - 6) / 24)
        temp = seasonal_temp + diurnal_temp + rng.normal(0, 2)

        # ── Solar Irradiance (W/m2) ──
        solar_angle = max(0, math.sin(2 * math.pi * (hour - 6) / 24))
        seasonal_solar = 0.7 + 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        solar = max(0, 1000 * solar_angle * seasonal_solar + rng.normal(0, 30))

        # ── Cloud Cover (%) ──
        cloud = max(0, min(100, 30 + 20 * math.sin(2 * math.pi * day_of_year / 365) + rng.normal(0, 15)))

        # ── Wind Speed (m/s) ──
        wind = max(0, 5 + 3 * math.sin(2 * math.pi * hour / 24) + rng.normal(0, 1.5))

        # ── MOER (lbs CO2/MWh) — the target ──
        base_moer = 650

        # Diurnal pattern: lowest 10AM-4PM (solar peak), highest 6-9PM (evening peak)
        diurnal = -150 * math.sin(2 * math.pi * (hour - 3) / 24)

        # Solar displacement: more solar -> lower MOER
        effective_solar = solar * (1 - cloud / 100)
        solar_effect = -effective_solar * 0.15

        # Temperature effect: extreme temps -> more HVAC -> more marginal gen
        temp_effect = max(0, abs(temp - 20) - 5) * 8

        # Weekend effect: lower demand -> lower MOER
        weekend_effect = -40 if is_weekend else 0

        # Wind effect: more wind -> lower MOER
        wind_effect = -wind * 3

        # Seasonal
        seasonal = 30 * math.sin(2 * math.pi * (day_of_year - 170) / 365)

        moer = base_moer + diurnal + solar_effect + temp_effect + weekend_effect + wind_effect + seasonal
        moer += rng.normal(0, 25)
        moer = max(100, moer)

        weather_rows.append({
            "timestamp": ts,
            "temperature_celsius": round(temp, 1),
            "solar_irradiance_w_m2": round(solar, 1),
            "cloud_cover_percent": round(cloud, 1),
            "wind_speed_m_s": round(wind, 1),
        })

        moer_rows.append({
            "timestamp": ts,
            "moer_value": round(moer, 1),
        })

    weather_df = pd.DataFrame(weather_rows)
    moer_df = pd.DataFrame(moer_rows)

    logger.info(
        "Generated synthetic dataset: %d hours (%d days), "
        "MOER range: [%.0f, %.0f] lbs CO2/MWh",
        hours, n_days,
        moer_df["moer_value"].min(),
        moer_df["moer_value"].max(),
    )

    return weather_df, moer_df


async def try_fetch_real_weather() -> pd.DataFrame | None:
    """Attempt to fetch real weather data from Open-Meteo."""
    try:
        from app.api_clients import OpenMeteoClient

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=14)
        start_date_str = start_time.strftime("%Y-%m-%d")
        end_date_str = end_time.strftime("%Y-%m-%d")

        async with OpenMeteoClient() as om:
            logger.info("Fetching Open-Meteo weather %s to %s", start_date_str, end_date_str)
            weather_resp = await om.get_historical_weather(
                latitude=LATITUDE,
                longitude=LONGITUDE,
                start_date=start_date_str,
                end_date=end_date_str,
            )
            weather_df = pd.DataFrame({
                "timestamp": weather_resp.hourly.time,
                "temperature_celsius": weather_resp.hourly.temperature_2m,
                "solar_irradiance_w_m2": weather_resp.hourly.shortwave_radiation,
                "cloud_cover_percent": weather_resp.hourly.cloud_cover,
                "wind_speed_m_s": weather_resp.hourly.wind_speed_10m,
            })
            logger.info("Real weather data: %d rows", len(weather_df))
            return weather_df

    except Exception as e:
        logger.warning("Failed to fetch real weather: %s", e)
        return None


async def main():
    from ml_pipeline.feature_engineering import engineer_features, get_feature_columns
    from ml_pipeline.forecaster import CarbonIntensityForecaster, ForecastConfig
    from ml_pipeline.evaluation import evaluate_forecast

    print("=" * 70)
    print("  Carbon-Aware EV Charging — ML Model Training")
    print("=" * 70)

    # ── 1. Get Data ─────────────────────────────────────────────────────────
    print("\n── 1. Preparing Training Data ──")

    # Always use synthetic for training (need 90+ days for good model)
    # Real data from free APIs is limited to a few days
    logger.info("Generating 90-day synthetic training dataset")
    weather_df, moer_df = generate_synthetic_dataset(n_days=90)
    data_source = "synthetic_90d"

    print(f"   Data source: {data_source}")
    print(f"   Weather rows: {len(weather_df)}")
    print(f"   MOER rows: {len(moer_df)}")

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    print("\n── 2. Engineering Features ──")

    df = engineer_features(weather_df, moer_df, drop_raw_columns=True)
    feature_cols = get_feature_columns(df)

    print(f"   Total rows: {len(df)}")
    print(f"   Features: {len(feature_cols)}")
    for i, col in enumerate(feature_cols):
        print(f"     {i+1:2d}. {col}")

    # ── 3. Train/Test Split ─────────────────────────────────────────────────
    print("\n── 3. Splitting Data ──")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"   Train: {len(train_df)} rows")
    print(f"   Test:  {len(test_df)} rows")

    # ── 4. Train Forecaster ─────────────────────────────────────────────────
    print("\n── 4. Training CarbonIntensityForecaster ──")

    config = ForecastConfig(
        horizon_hours=24,
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        rf_n_estimators=200,
        rf_max_depth=12,
        random_state=42,
    )

    forecaster = CarbonIntensityForecaster(config=config)
    forecaster.fit(
        train_df,
        feature_columns=feature_cols,
        target_column="moer_value",
        fit_baseline=True,
    )

    print(f"   Primary models: {len(forecaster.models)}")
    print(f"   Baseline models: {len(forecaster.baseline_models)}")

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    print("\n── 5. Evaluating Model ──")

    y_pred_df = forecaster.predict(test_df)

    y_true_data = {}
    for h in range(1, 25):
        shifted = test_df["moer_value"].shift(-h)
        y_true_data[f"h+{h}"] = shifted.values
    y_true_df = pd.DataFrame(y_true_data, index=test_df.index)

    report = evaluate_forecast(
        y_true_df, y_pred_df,
        model_name="HistGradientBoosting (Direct)",
        notes=f"Data: {data_source}, Train: {len(train_df)}, Test: {len(test_df)}"
    )
    print(report.summary())

    # Baseline
    y_pred_baseline = forecaster.predict(test_df, use_baseline=True)
    baseline_report = evaluate_forecast(
        y_true_df, y_pred_baseline,
        model_name="RandomForest Baseline",
        notes=f"Data: {data_source}"
    )
    print()
    print(baseline_report.summary())

    # ── 6. Feature Importance ───────────────────────────────────────────────
    print("\n── 6. Feature Importance (h=1) ──")

    importance = forecaster.get_feature_importance(horizon=1)
    for _, row in importance.head(10).iterrows():
        bar_len = int(row["importance"] * 100)
        bar = "#" * bar_len
        print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")

    # ── 7. Save Artifact ────────────────────────────────────────────────────
    print("\n── 7. Saving Model Artifact ──")

    artifact_path = Path("model_artifact.joblib")
    joblib.dump(forecaster, artifact_path)

    dir_path = Path("model_artifact_dir")
    forecaster.save(dir_path)

    print(f"   Saved monolithic: {artifact_path} ({artifact_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   Saved directory:  {dir_path}/")

    # ── 8. Verify Inference ─────────────────────────────────────────────────
    print("\n── 8. Verifying Inference ──")

    loaded = joblib.load(artifact_path)
    single_row = test_df.iloc[0:1]
    preds = loaded.predict_single(single_row)

    print(f"   Input features shape: {single_row[feature_cols].shape}")
    print(f"   Output predictions:   {preds.shape}")
    print(f"   Prediction range:     [{preds.min():.1f}, {preds.max():.1f}] lbs CO2/MWh")
    print(f"   h+1={preds[0]:.1f}, h+6={preds[5]:.1f}, h+12={preds[11]:.1f}, h+24={preds[23]:.1f}")

    print("\n" + "=" * 70)
    print("  Training complete! Model artifact saved.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
