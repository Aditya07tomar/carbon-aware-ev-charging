"""
MOER Forecaster — predicts the next 24 hours of grid carbon intensity.

Architecture decision:
    We use scikit-learn's HistGradientBoostingRegressor as the primary model
    and RandomForestRegressor as a baseline.  For tabular time-series with
    engineered features, gradient-boosted trees consistently outperform
    neural networks (see Borisov et al., 2022 "Deep Neural Networks and
    Tabular Data: A Survey") while being faster to train, easier to
    interpret, and more robust to hyperparameter choices.

    HistGradientBoostingRegressor specifically is:
    • Inspired by LightGBM — histogram-based splits are O(n) not O(n log n)
    • Natively handles missing values (no imputation required)
    • Supports early stopping to prevent overfitting

Forecasting strategy:
    "Direct" multi-step — we train one model per forecast horizon step
    (h=1, h=2, … h=24).  Each model learns to predict MOER at t+h given
    features at time t.  This avoids error accumulation from recursive
    (autoregressive) forecasting, at the cost of 24× model storage.

Public API:
    CarbonIntensityForecaster  — Train, predict, save, load, inspect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ForecastConfig:
    """Hyperparameters and training configuration."""

    # Forecast horizon
    horizon_hours: int = 24

    # HistGradientBoosting hyperparameters
    max_iter: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1
    max_bins: int = 255
    early_stopping: bool = True
    n_iter_no_change: int = 15
    validation_fraction: float = 0.1

    # RandomForest baseline hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 10

    # Cross-validation
    cv_n_splits: int = 5

    # Random state for reproducibility
    random_state: int = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Forecaster
# ═══════════════════════════════════════════════════════════════════════════


class CarbonIntensityForecaster:
    """
    Multi-step MOER forecaster using direct forecasting strategy.

    For each forecast horizon h ∈ {1, 2, …, 24}, a separate gradient-
    boosted tree model is trained to predict MOER(t + h) given features
    at time t.

    Attributes
    ----------
    config : ForecastConfig
        Training hyperparameters.
    models : dict[int, BaseEstimator]
        Trained models keyed by horizon hour (1–24).
    baseline_models : dict[int, BaseEstimator]
        Trained RandomForest baselines keyed by horizon hour.
    feature_columns : list[str]
        Ordered list of feature names used during training.
    is_fitted : bool
        Whether the forecaster has been trained.

    Example
    -------
    >>> forecaster = CarbonIntensityForecaster()
    >>> forecaster.fit(feature_df, feature_columns=["hour_sin", ...])
    >>> predictions = forecaster.predict(latest_features_df)
    >>> predictions.shape  # (n_samples, 24)
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config: ForecastConfig = config or ForecastConfig()
        self.models: dict[int, BaseEstimator] = {}
        self.baseline_models: dict[int, BaseEstimator] = {}
        self.feature_columns: list[str] = []
        self.is_fitted: bool = False
        self._training_metrics: dict[int, dict[str, float]] = {}

    # ── Training ────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str = "moer_value",
        fit_baseline: bool = True,
    ) -> CarbonIntensityForecaster:
        """
        Train one model per horizon step using direct forecasting.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with the target column.
            Must be sorted chronologically.
        feature_columns : list[str] | None
            Columns to use as input features.  If None, all columns
            except ``target_column`` and ``timestamp`` are used.
        target_column : str
            Name of the MOER target column.
        fit_baseline : bool
            If True, also train RandomForest baselines for comparison.

        Returns
        -------
        self
            The fitted forecaster (for method chaining).
        """
        if feature_columns is None:
            exclude = {target_column, "timestamp"}
            feature_columns = [c for c in df.columns if c not in exclude]

        self.feature_columns = feature_columns

        X = df[feature_columns].values
        y = df[target_column].values

        logger.info(
            "Training %d horizon models on %d samples × %d features",
            self.config.horizon_hours, len(X), len(feature_columns),
        )

        for h in range(1, self.config.horizon_hours + 1):
            # Shift target by h steps into the future
            y_shifted = pd.Series(y).shift(-h).values

            # Drop the last h rows where the target is NaN
            valid_mask = ~np.isnan(y_shifted)
            X_h = X[valid_mask]
            y_h = y_shifted[valid_mask]

            if len(X_h) < 50:
                logger.warning(
                    "Horizon h=%d: only %d valid samples — skipping.", h, len(X_h),
                )
                continue

            # ── Primary Model: HistGradientBoosting ─────────────────────
            model = HistGradientBoostingRegressor(
                max_iter=self.config.max_iter,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                max_bins=self.config.max_bins,
                early_stopping=self.config.early_stopping,
                n_iter_no_change=self.config.n_iter_no_change,
                validation_fraction=self.config.validation_fraction,
                random_state=self.config.random_state,
            )
            model.fit(X_h, y_h)
            self.models[h] = model

            # ── Baseline Model: RandomForest ────────────────────────────
            if fit_baseline:
                rf = RandomForestRegressor(
                    n_estimators=self.config.rf_n_estimators,
                    max_depth=self.config.rf_max_depth,
                    min_samples_leaf=self.config.rf_min_samples_leaf,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_h, y_h)
                self.baseline_models[h] = rf

            logger.debug("Horizon h=%d trained (%d samples).", h, len(X_h))

        self.is_fitted = True
        logger.info(
            "Training complete: %d primary + %d baseline models.",
            len(self.models), len(self.baseline_models),
        )
        return self

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        *,
        use_baseline: bool = False,
    ) -> pd.DataFrame:
        """
        Predict the next 24 hours of MOER for each row in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with the same columns used during training.
            Each row represents a "now" timestamp; the model predicts
            MOER at t+1, t+2, …, t+24 for each row.
        use_baseline : bool
            If True, use RandomForest baselines instead of the primary
            HistGradientBoosting models.

        Returns
        -------
        pd.DataFrame
            Shape (n_samples, 24).  Column names are "h+1", "h+2", …, "h+24".
            Values are predicted MOER (lbs CO₂ / MWh).
        """
        self._assert_fitted()

        models = self.baseline_models if use_baseline else self.models
        X = df[self.feature_columns].values
        predictions: dict[str, np.ndarray] = {}

        for h in range(1, self.config.horizon_hours + 1):
            if h in models:
                predictions[f"h+{h}"] = models[h].predict(X)
            else:
                # Fill missing horizons with NaN
                predictions[f"h+{h}"] = np.full(len(X), np.nan)

        return pd.DataFrame(predictions, index=df.index)

    def predict_single(
        self,
        features: pd.DataFrame,
        *,
        use_baseline: bool = False,
    ) -> np.ndarray:
        """
        Predict the next 24h MOER for a single observation (1-row DF).

        Returns
        -------
        np.ndarray
            Shape (24,) — predicted MOER for each of the next 24 hours.
        """
        preds = self.predict(features.head(1), use_baseline=use_baseline)
        return preds.values.flatten()

    # ── Cross-Validation ────────────────────────────────────────────────────

    def cross_validate(
        self,
        df: pd.DataFrame,
        *,
        target_column: str = "moer_value",
        horizon: int = 1,
    ) -> dict[str, list[float]]:
        """
        Run time-series cross-validation for a single horizon.

        Uses ``TimeSeriesSplit`` to respect temporal ordering (no future
        leakage).

        Parameters
        ----------
        df : pd.DataFrame
            Full feature-engineered DataFrame.
        target_column : str
            Name of the target column.
        horizon : int
            Which forecast horizon to evaluate (1–24).

        Returns
        -------
        dict
            Keys: "mae", "rmse" — each a list of per-fold scores.
        """
        from ml_pipeline.evaluation import mean_absolute_error, root_mean_square_error

        feature_cols = self.feature_columns or [
            c for c in df.columns if c not in {target_column, "timestamp"}
        ]

        X = df[feature_cols].values
        y_shifted = pd.Series(df[target_column].values).shift(-horizon).values

        valid = ~np.isnan(y_shifted)
        X, y = X[valid], y_shifted[valid]

        tscv = TimeSeriesSplit(n_splits=self.config.cv_n_splits)
        mae_scores: list[float] = []
        rmse_scores: list[float] = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = HistGradientBoostingRegressor(
                max_iter=self.config.max_iter,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                random_state=self.config.random_state,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(root_mean_square_error(y_test, y_pred))

        return {"mae": mae_scores, "rmse": rmse_scores}

    # ── Feature Importance ──────────────────────────────────────────────────

    def get_feature_importance(
        self,
        horizon: int = 1,
        *,
        use_baseline: bool = False,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Return feature importances for a given horizon model.

        Attempts to use the model's built-in ``feature_importances_``
        attribute (MDI-based).  If unavailable (e.g. newer scikit-learn
        HistGradientBoosting), falls back to permutation importance
        when validation data is provided, or to a uniform placeholder.

        Parameters
        ----------
        horizon : int
            Forecast horizon step (1–24).
        use_baseline : bool
            If True, inspect the RandomForest baseline.
        X_val : array-like | None
            Validation features for permutation importance fallback.
        y_val : array-like | None
            Validation targets for permutation importance fallback.

        Returns
        -------
        pd.DataFrame
            Columns: "feature", "importance".  Sorted descending.
        """
        self._assert_fitted()

        models = self.baseline_models if use_baseline else self.models
        if horizon not in models:
            raise ValueError(f"No model trained for horizon h={horizon}")

        model = models[horizon]

        # Try built-in feature importances first
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif X_val is not None and y_val is not None:
            # Permutation importance: model-agnostic, more robust
            from sklearn.inspection import permutation_importance
            X_val_arr = np.asarray(X_val)
            y_val_arr = np.asarray(y_val)
            result = permutation_importance(
                model, X_val_arr, y_val_arr,
                n_repeats=10,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            importances = result.importances_mean
        else:
            # Last resort: uniform placeholder
            logger.warning(
                "Model for h=%d has no feature_importances_ and no validation "
                "data was provided for permutation importance. Returning uniform.",
                horizon,
            )
            importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Save all models, config, and feature metadata to a directory.

        File layout:
            directory/
            ├── config.joblib
            ├── feature_columns.joblib
            ├── primary/
            │   ├── model_h1.joblib
            │   ├── model_h2.joblib
            │   └── …
            └── baseline/
                ├── model_h1.joblib
                └── …
        """
        self._assert_fitted()
        base = Path(directory)

        # Save config + features
        base.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.config, base / "config.joblib")
        joblib.dump(self.feature_columns, base / "feature_columns.joblib")

        # Save primary models
        primary_dir = base / "primary"
        primary_dir.mkdir(exist_ok=True)
        for h, model in self.models.items():
            joblib.dump(model, primary_dir / f"model_h{h}.joblib")

        # Save baseline models
        if self.baseline_models:
            baseline_dir = base / "baseline"
            baseline_dir.mkdir(exist_ok=True)
            for h, model in self.baseline_models.items():
                joblib.dump(model, baseline_dir / f"model_h{h}.joblib")

        logger.info("Forecaster saved to %s", base)

    @classmethod
    def load(cls, directory: str | Path) -> CarbonIntensityForecaster:
        """Load a saved forecaster from a directory."""
        base = Path(directory)

        config: ForecastConfig = joblib.load(base / "config.joblib")
        feature_columns: list[str] = joblib.load(base / "feature_columns.joblib")

        forecaster = cls(config=config)
        forecaster.feature_columns = feature_columns

        # Load primary models
        primary_dir = base / "primary"
        if primary_dir.exists():
            for path in sorted(primary_dir.glob("model_h*.joblib")):
                h = int(path.stem.replace("model_h", ""))
                forecaster.models[h] = joblib.load(path)

        # Load baseline models
        baseline_dir = base / "baseline"
        if baseline_dir.exists():
            for path in sorted(baseline_dir.glob("model_h*.joblib")):
                h = int(path.stem.replace("model_h", ""))
                forecaster.baseline_models[h] = joblib.load(path)

        forecaster.is_fitted = True
        logger.info(
            "Forecaster loaded from %s (%d primary, %d baseline models)",
            base, len(forecaster.models), len(forecaster.baseline_models),
        )
        return forecaster

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        """Raise if the forecaster has not been trained yet."""
        if not self.is_fitted or not self.models:
            raise RuntimeError(
                "Forecaster has not been fitted yet. Call .fit() first."
            )
