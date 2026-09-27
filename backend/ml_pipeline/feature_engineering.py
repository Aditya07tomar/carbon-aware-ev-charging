"""
Feature engineering for carbon intensity (MOER) forecasting.

Transforms raw Open-Meteo weather data and WattTime MOER data into a
machine-learning-ready feature matrix.  Every feature is chosen for a
specific physical or statistical reason documented inline.

Public API:
    engineer_features(weather_df, moer_df) → pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Hours per day / week — used for cyclical encoding
_HOURS_PER_DAY: int = 24
_DAYS_PER_WEEK: int = 7


# ═══════════════════════════════════════════════════════════════════════════
#  Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def engineer_features(
    weather_df: pd.DataFrame,
    moer_df: pd.DataFrame,
    *,
    drop_raw_columns: bool = True,
    fillna_strategy: str = "ffill",
) -> pd.DataFrame:
    """
    Merge raw weather and MOER data, then generate time-series features.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Open-Meteo hourly data.  Expected columns:
            timestamp, temperature_celsius, solar_irradiance_w_m2,
            cloud_cover_percent, wind_speed_m_s
    moer_df : pd.DataFrame
        WattTime MOER data.  Expected columns:
            timestamp, moer_value
    drop_raw_columns : bool
        If True, remove the raw weather/MOER columns after feature
        generation to avoid leaking unprocessed data into the model.
    fillna_strategy : str
        Strategy for filling NaN values created by rolling/lag operations.
        One of "ffill" (forward-fill) or "zero" (fill with 0).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by timestamp, with the target column
        ``moer_value`` preserved.  All feature columns are numeric and
        ready for model ingestion.
    """
    # ── 1. Validate & Prepare ───────────────────────────────────────────────
    df = _merge_and_prepare(weather_df, moer_df)

    # ── 2. Temporal Features ────────────────────────────────────────────────
    #   Motivation: Grid carbon intensity follows strong diurnal and weekly
    #   patterns driven by demand cycles and generation dispatch schedules.
    df = _add_temporal_features(df)

    # ── 3. Rolling / Window Features ────────────────────────────────────────
    #   Motivation: Short-term trends capture momentum in solar generation
    #   and emission trends that simple point-in-time values miss.
    df = _add_rolling_features(df)

    # ── 4. Lag Features ─────────────────────────────────────────────────────
    #   Motivation: Auto-regressive lags give the model direct access to
    #   recent historical MOER values — the single strongest predictor
    #   family for short-horizon time-series forecasting.
    df = _add_lag_features(df)

    # ── 5. Delta / Rate-of-Change Features ──────────────────────────────────
    #   Motivation: Temperature deltas capture thermal ramp events that
    #   trigger peaker plants (high-carbon gas turbines).  MOER rate of
    #   change captures whether emissions are trending up or down.
    df = _add_delta_features(df)

    # ── 6. Interaction Features ─────────────────────────────────────────────
    #   Motivation: Solar irradiance alone is insufficient — high cloud
    #   cover dramatically reduces actual PV output.  The interaction
    #   captures *effective* solar generation.
    df = _add_interaction_features(df)

    # ── 7. Handle Missing Values ────────────────────────────────────────────
    df = _handle_missing_values(df, strategy=fillna_strategy)

    # ── 8. Optionally Drop Raw Columns ──────────────────────────────────────
    if drop_raw_columns:
        df = _drop_raw_columns(df)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _merge_and_prepare(
    weather_df: pd.DataFrame,
    moer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather and MOER DataFrames on timestamp, sort, and set index."""
    # Normalise column names
    weather = weather_df.copy()
    moer = moer_df.copy()

    # Ensure timestamps are datetime
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    moer["timestamp"] = pd.to_datetime(moer["timestamp"], utc=True)

    # Merge on nearest timestamp (weather is hourly, MOER may be 5-min)
    df = pd.merge_asof(
        moer.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar / cyclical time features.

    Features created:
        hour_sin, hour_cos       — Cyclical hour-of-day (captures midnight wrap)
        dow_sin, dow_cos         — Cyclical day-of-week (captures Sun→Mon wrap)
        month_sin, month_cos     — Cyclical month (captures seasonal patterns)
        is_weekend               — Binary flag (Sat/Sun → 1)

    Why cyclical encoding?
        A linear "hour" column implies hour 23 is far from hour 0, but in
        reality they are one hour apart.  Sin/cos projections preserve the
        true circular distance between time points.
    """
    ts = pd.to_datetime(df["timestamp"])

    # Hour of day — cyclical
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / _HOURS_PER_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * hour / _HOURS_PER_DAY)

    # Day of week — cyclical (Monday=0 … Sunday=6)
    dow = ts.dt.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / _DAYS_PER_WEEK)
    df["dow_cos"] = np.cos(2 * np.pi * dow / _DAYS_PER_WEEK)

    # Month — cyclical (captures seasonal generation mix shifts)
    month = ts.dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Weekend flag — weekend demand profiles differ materially from weekdays
    df["is_weekend"] = dow.isin([5, 6]).astype(np.int8)

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling-window aggregations.

    Features created:
        solar_rolling_3h_mean    — Mean solar irradiance over past 3 hours
        solar_rolling_6h_mean    — Mean solar irradiance over past 6 hours
        moer_rolling_3h_mean     — Mean MOER over past 3 hours
        moer_rolling_6h_mean     — Mean MOER over past 6 hours
        moer_rolling_12h_std     — MOER volatility over past 12 hours
        temp_rolling_6h_mean     — Mean temperature over past 6 hours

    Why rolling windows?
        Point-in-time solar irradiance can spike/dip due to transient
        clouds.  A 3-hour rolling mean smooths these and better represents
        the sustained solar generation trend that actually affects the
        grid's carbon intensity.

        MOER rolling std captures volatility — high volatility periods
        indicate grid stress and frequent dispatch changes, which is
        useful for the model to distinguish stable vs. volatile regimes.
    """
    # Solar irradiance windows
    if "solar_irradiance_w_m2" in df.columns:
        df["solar_rolling_3h_mean"] = (
            df["solar_irradiance_w_m2"]
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df["solar_rolling_6h_mean"] = (
            df["solar_irradiance_w_m2"]
            .rolling(window=6, min_periods=1)
            .mean()
        )

    # MOER windows
    if "moer_value" in df.columns:
        df["moer_rolling_3h_mean"] = (
            df["moer_value"]
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df["moer_rolling_6h_mean"] = (
            df["moer_value"]
            .rolling(window=6, min_periods=1)
            .mean()
        )
        df["moer_rolling_12h_std"] = (
            df["moer_value"]
            .rolling(window=12, min_periods=1)
            .std()
        )

    # Temperature windows
    if "temperature_celsius" in df.columns:
        df["temp_rolling_6h_mean"] = (
            df["temperature_celsius"]
            .rolling(window=6, min_periods=1)
            .mean()
        )

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add auto-regressive lag features for MOER.

    Features created:
        moer_lag_1h   — MOER value 1 hour ago
        moer_lag_3h   — MOER value 3 hours ago
        moer_lag_6h   — MOER value 6 hours ago
        moer_lag_12h  — MOER value 12 hours ago
        moer_lag_24h  — MOER value 24 hours ago (same hour yesterday)

    Why these lags?
        • 1h lag: strongest single predictor — MOER changes slowly
        • 3h / 6h: capture intra-day dispatch transitions
        • 12h: captures AM↔PM shift (solar on → solar off)
        • 24h: captures same-hour-yesterday seasonality (daily pattern)

    Research note:
        The 24h lag is critical for capturing recurring daily demand
        patterns.  Without it, the model struggles to learn that e.g.
        6 PM peak demand recurs daily.
    """
    if "moer_value" not in df.columns:
        return df

    for lag_hours in [1, 3, 6, 12, 24]:
        df[f"moer_lag_{lag_hours}h"] = df["moer_value"].shift(lag_hours)

    return df


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate-of-change and delta features.

    Features created:
        temp_delta_24h            — Temperature difference from 24 hours ago
        moer_rate_of_change_1h    — MOER change over the past hour
        moer_rate_of_change_3h    — MOER change over the past 3 hours
        solar_rate_of_change_1h   — Solar irradiance change over the past hour

    Why temperature delta?
        A sharp temperature increase (e.g. +10°C in 24h) triggers HVAC
        load spikes that force dispatch of marginal (typically gas-fired)
        generators.  The delta captures this ramp signal that the
        absolute temperature alone cannot.

    Why MOER rate of change?
        Knowing whether emissions are rising or falling provides the
        model with momentum information — a rapidly rising MOER is
        likely to continue rising in the near term due to generator
        ramp constraints.
    """
    if "temperature_celsius" in df.columns:
        df["temp_delta_24h"] = (
            df["temperature_celsius"] - df["temperature_celsius"].shift(24)
        )

    if "moer_value" in df.columns:
        df["moer_rate_of_change_1h"] = (
            df["moer_value"] - df["moer_value"].shift(1)
        )
        df["moer_rate_of_change_3h"] = (
            df["moer_value"] - df["moer_value"].shift(3)
        )

    if "solar_irradiance_w_m2" in df.columns:
        df["solar_rate_of_change_1h"] = (
            df["solar_irradiance_w_m2"] - df["solar_irradiance_w_m2"].shift(1)
        )

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-variable interaction features.

    Features created:
        effective_solar     — solar_irradiance × (1 - cloud_cover / 100)
        temp_x_wind_chill   — temperature × wind_speed  (simplified wind chill proxy)

    Why effective solar?
        Raw solar irradiance (GHI measured or modeled at surface) does
        not account for cloud occlusion.  Multiplying by the clear-sky
        fraction gives a proxy for actual PV panel output, which
        directly displaces fossil generation and lowers MOER.

    Why temp × wind?
        Wind chill accelerates EV battery thermal drain and affects
        building HVAC load — both influence grid demand.
    """
    has_solar = "solar_irradiance_w_m2" in df.columns
    has_cloud = "cloud_cover_percent" in df.columns
    has_temp = "temperature_celsius" in df.columns
    has_wind = "wind_speed_m_s" in df.columns

    if has_solar and has_cloud:
        clear_sky_fraction = 1.0 - (df["cloud_cover_percent"].fillna(0) / 100.0)
        df["effective_solar"] = df["solar_irradiance_w_m2"].fillna(0) * clear_sky_fraction

    if has_temp and has_wind:
        df["temp_x_wind_chill"] = (
            df["temperature_celsius"].fillna(0) * df["wind_speed_m_s"].fillna(0)
        )

    return df


def _handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "ffill",
) -> pd.DataFrame:
    """
    Fill NaN values introduced by rolling/lag operations.

    The first N rows (where N = max lag window) will have NaNs.
    Forward-fill propagates the earliest available value backward,
    which is safer than zero-fill for MOER (where 0 is physically
    impossible and would confuse the model).
    """
    if strategy == "ffill":
        df = df.ffill().bfill()  # ffill first, bfill to catch leading NaNs
    elif strategy == "zero":
        df = df.fillna(0)
    else:
        raise ValueError(f"Unknown fillna strategy: {strategy!r}. Use 'ffill' or 'zero'.")

    return df


def _drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove raw input columns that have been superseded by engineered features.

    We keep ``timestamp`` (for index/join purposes) and ``moer_value``
    (as the prediction target).  Everything else raw is dropped.
    """
    raw_columns_to_drop = [
        "solar_irradiance_w_m2",
        "cloud_cover_percent",
        "wind_speed_m_s",
        "temperature_celsius",
    ]
    existing = [c for c in raw_columns_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Feature List for Model Consumption
# ═══════════════════════════════════════════════════════════════════════════


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excluding target and metadata).

    Useful for passing to the forecaster's ``fit()`` and ``predict()`` methods
    to ensure consistent feature ordering.
    """
    exclude = {"timestamp", "moer_value"}
    return [c for c in df.columns if c not in exclude]
