"""
Model evaluation utilities for carbon intensity (MOER) forecasting.

Provides:
    • Core metrics: MAE, RMSE (required for the research paper methodology).
    • Extended metrics: MAPE, R², per-horizon error breakdown.
    • Visualization: actual-vs-predicted plots and per-horizon error bars.
    • Full evaluation report suitable for inclusion in a LaTeX paper.

Public API:
    mean_absolute_error(y_true, y_pred) → float
    root_mean_square_error(y_true, y_pred) → float
    evaluate_forecast(y_true, y_pred, ...) → EvaluationReport
    plot_actual_vs_predicted(y_true, y_pred, ...) → Figure
    plot_horizon_errors(report, ...) → Figure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Core Metrics
# ═══════════════════════════════════════════════════════════════════════════


def mean_absolute_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    Compute Mean Absolute Error (MAE).

        MAE = (1/n) × Σ|yᵢ − ŷᵢ|

    MAE is interpretable in the same units as the target (lbs CO₂/MWh)
    and is robust to outliers, making it the preferred primary metric
    for our research paper.

    Parameters
    ----------
    y_true : array-like
        Ground-truth MOER values.
    y_pred : array-like
        Predicted MOER values.

    Returns
    -------
    float
        The MAE score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Remove pairs where either value is NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No valid (non-NaN) samples to evaluate.")

    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_square_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    Compute Root Mean Square Error (RMSE).

        RMSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]

    RMSE penalizes large errors more heavily than MAE, which is
    important for identifying catastrophic forecast failures where
    charging schedules could be seriously sub-optimal.

    Parameters
    ----------
    y_true : array-like
        Ground-truth MOER values.
    y_pred : array-like
        Predicted MOER values.

    Returns
    -------
    float
        The RMSE score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No valid (non-NaN) samples to evaluate.")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
#  Extended Metrics
# ═══════════════════════════════════════════════════════════════════════════


def mean_absolute_percentage_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

        MAPE = (100/n) × Σ|yᵢ − ŷᵢ| / |yᵢ|

    Excluded from the primary metrics because MOER can approach zero
    during high-renewables periods, making MAPE unstable.  Useful as
    a supplementary metric when MOER > 0.

    Returns
    -------
    float
        MAPE as a percentage (0–100+).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (np.abs(y_true) > 1e-8)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def r_squared(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    Compute the coefficient of determination (R²).

        R² = 1 − SS_res / SS_tot

    R² indicates the fraction of variance in MOER explained by the
    model.  A value of 1.0 is a perfect fit; ≤0 means the model is
    worse than predicting the mean.

    Returns
    -------
    float
        R² score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation Report
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HorizonMetrics:
    """Metrics for a single forecast horizon step."""
    horizon_hour: int
    mae: float
    rmse: float
    mape: float
    r2: float
    n_samples: int


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for multi-horizon MOER forecasting.

    Suitable for direct inclusion in the research paper's results section.
    """
    # Aggregate metrics (averaged across all horizons)
    overall_mae: float
    overall_rmse: float
    overall_mape: float
    overall_r2: float
    total_samples: int

    # Per-horizon breakdown
    horizon_metrics: list[HorizonMetrics] = field(default_factory=list)

    # Model metadata
    model_name: str = ""
    notes: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-horizon metrics to a DataFrame for easy display."""
        records = [
            {
                "horizon_h": hm.horizon_hour,
                "MAE": round(hm.mae, 4),
                "RMSE": round(hm.rmse, 4),
                "MAPE_%": round(hm.mape, 2),
                "R²": round(hm.r2, 4),
                "n_samples": hm.n_samples,
            }
            for hm in self.horizon_metrics
        ]
        return pd.DataFrame(records)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"═══ Evaluation Report: {self.model_name} ═══",
            f"  Samples:      {self.total_samples:,}",
            f"  Overall MAE:  {self.overall_mae:.4f} lbs CO₂/MWh",
            f"  Overall RMSE: {self.overall_rmse:.4f} lbs CO₂/MWh",
            f"  Overall MAPE: {self.overall_mape:.2f}%",
            f"  Overall R²:   {self.overall_r2:.4f}",
            "",
        ]
        if self.horizon_metrics:
            lines.append("  Per-Horizon Breakdown:")
            lines.append(f"  {'h':>4s}  {'MAE':>10s}  {'RMSE':>10s}  {'R²':>8s}")
            lines.append(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")
            for hm in self.horizon_metrics:
                lines.append(
                    f"  {hm.horizon_hour:>4d}  "
                    f"{hm.mae:>10.4f}  "
                    f"{hm.rmse:>10.4f}  "
                    f"{hm.r2:>8.4f}"
                )
        if self.notes:
            lines.extend(["", f"  Notes: {self.notes}"])
        return "\n".join(lines)


def evaluate_forecast(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    *,
    model_name: str = "HistGradientBoosting",
    notes: str = "",
) -> EvaluationReport:
    """
    Evaluate a multi-horizon forecast against ground truth.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        Ground-truth MOER values.  Shape (n_samples, 24).
        Column names should be "h+1", "h+2", …, "h+24".
    y_pred_df : pd.DataFrame
        Predicted MOER values.  Same shape and columns as y_true_df.
    model_name : str
        Human-readable model name for the report header.
    notes : str
        Any additional notes (e.g. feature set version, data window).

    Returns
    -------
    EvaluationReport
        Complete report with aggregate and per-horizon metrics.
    """
    horizon_results: list[HorizonMetrics] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    for col in y_true_df.columns:
        yt = y_true_df[col].values
        yp = y_pred_df[col].values

        # Filter NaNs
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt_clean, yp_clean = yt[mask], yp[mask]

        if len(yt_clean) == 0:
            continue

        h = int(col.replace("h+", ""))
        horizon_results.append(HorizonMetrics(
            horizon_hour=h,
            mae=mean_absolute_error(yt_clean, yp_clean),
            rmse=root_mean_square_error(yt_clean, yp_clean),
            mape=mean_absolute_percentage_error(yt_clean, yp_clean),
            r2=r_squared(yt_clean, yp_clean),
            n_samples=len(yt_clean),
        ))

        all_true.extend(yt_clean.tolist())
        all_pred.extend(yp_clean.tolist())

    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)

    report = EvaluationReport(
        overall_mae=mean_absolute_error(all_true_arr, all_pred_arr),
        overall_rmse=root_mean_square_error(all_true_arr, all_pred_arr),
        overall_mape=mean_absolute_percentage_error(all_true_arr, all_pred_arr),
        overall_r2=r_squared(all_true_arr, all_pred_arr),
        total_samples=len(all_true_arr),
        horizon_metrics=sorted(horizon_results, key=lambda x: x.horizon_hour),
        model_name=model_name,
        notes=notes,
    )

    logger.info("Evaluation complete: MAE=%.4f, RMSE=%.4f", report.overall_mae, report.overall_rmse)
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    title: str = "Actual vs Predicted MOER",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted MOER values with identity line.

    Useful for visual inspection of prediction quality across the full
    range of MOER values.  Points near the 45° line indicate good
    predictions; systematic deviations indicate bias.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    save_path : Path | None
        If provided, save the figure to this path.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="#4A90D9", label="Predictions")

    # Identity line (perfect prediction)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "--", color="#E74C3C", linewidth=1.5, label="Perfect Prediction")

    # Metrics annotation
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_square_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    ax.text(
        0.05, 0.92,
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Actual MOER (lbs CO₂/MWh)", fontsize=12)
    ax.set_ylabel("Predicted MOER (lbs CO₂/MWh)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)

    return fig


def plot_horizon_errors(
    report: EvaluationReport,
    *,
    metric: str = "mae",
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Bar chart showing per-horizon forecast errors.

    Reveals how prediction accuracy degrades with forecast distance —
    critical for determining the optimal scheduling lookahead window
    in the EV charging optimizer.

    Parameters
    ----------
    report : EvaluationReport
        A completed evaluation report with per-horizon metrics.
    metric : str
        Which metric to plot: "mae", "rmse", "mape", or "r2".
    title : str | None
        Custom title.  If None, auto-generated.
    save_path : Path | None
        If provided, save the figure.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if not report.horizon_metrics:
        raise ValueError("Report has no per-horizon metrics to plot.")

    horizons = [hm.horizon_hour for hm in report.horizon_metrics]
    values = [getattr(hm, metric) for hm in report.horizon_metrics]

    metric_labels: dict[str, str] = {
        "mae": "MAE (lbs CO₂/MWh)",
        "rmse": "RMSE (lbs CO₂/MWh)",
        "mape": "MAPE (%)",
        "r2": "R²",
    }

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(horizons)))  # type: ignore[attr-defined]
    bars = ax.bar(horizons, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Forecast Horizon (hours ahead)", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
    ax.set_title(
        title or f"{report.model_name}: {metric.upper()} by Forecast Horizon",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(horizons)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)

    return fig


def compare_models(
    reports: list[EvaluationReport],
    *,
    metric: str = "mae",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Overlay per-horizon errors from multiple models for comparison.

    Essential for the research paper's results section to visually
    compare HistGradientBoosting vs RandomForest baselines.

    Parameters
    ----------
    reports : list[EvaluationReport]
        List of evaluation reports from different models.
    metric : str
        Which metric to compare.
    save_path : Path | None
        If provided, save the figure.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    palette = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]

    for idx, report in enumerate(reports):
        if not report.horizon_metrics:
            continue
        horizons = [hm.horizon_hour for hm in report.horizon_metrics]
        values = [getattr(hm, metric) for hm in report.horizon_metrics]
        color = palette[idx % len(palette)]
        ax.plot(
            horizons, values,
            marker="o", markersize=5, linewidth=2,
            color=color, label=report.model_name,
        )

    metric_labels: dict[str, str] = {
        "mae": "MAE (lbs CO₂/MWh)",
        "rmse": "RMSE (lbs CO₂/MWh)",
        "mape": "MAPE (%)",
        "r2": "R²",
    }

    ax.set_xlabel("Forecast Horizon (hours ahead)", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=12)
    ax.set_title(
        f"Model Comparison: {metric.upper()} by Forecast Horizon",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)

    return fig
