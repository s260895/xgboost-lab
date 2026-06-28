"""Evaluation metrics and diagnostic plots.

Stage 1 covers binary-classification metrics; Stage 2 adds regression metrics plus
residual and learning-curve plots. Stages 4+ add calibration.

Why several metrics and not just accuracy: on imbalanced data accuracy is
misleading (Stage 4), and a model can rank well (high ROC-AUC) yet be poorly
calibrated (high Brier). For regression, RMSE and MAE answer different questions
(RMSE punishes large errors more). Logging the whole panel keeps us honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def binary_classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, float]:
    """Standard panel for binary classification from predicted probabilities.

    Args:
        y_true: ground-truth 0/1 labels.
        proba: predicted probability of the positive class.
        threshold: decision threshold for the hard-label metrics.
        prefix: prepended to each key (e.g. ``"val_"`` / ``"test_"``).

    Returns a dict of: logloss, roc_auc, pr_auc (average precision), accuracy,
    and brier score.
    """
    preds = (proba >= threshold).astype(int)
    metrics = {
        "logloss": float(log_loss(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Standard panel for regression.

    - **rmse** — root mean squared error; in target units, punishes large errors.
    - **mae** — mean absolute error; in target units, robust to outliers.
    - **r2** — fraction of variance explained (1.0 perfect, 0.0 = predicting the
      mean, can go negative for a model worse than the mean).
    """
    metrics = {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Residuals (y_true - y_pred) vs. predicted value.

    A healthy plot is a structureless cloud centered on zero. Patterns — a funnel
    (heteroscedasticity), curvature (missed nonlinearity), or a ceiling/floor
    (clipped target) — are diagnostic of what the model is missing.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    ax.scatter(y_pred, residuals, s=8, alpha=0.3, edgecolors="none")
    ax.axhline(0.0, color="red", lw=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (true - pred)")
    ax.set_title("Residual plot")
    return ax


def plot_learning_curve(
    evals_result: dict[str, dict[str, list[float]]],
    metric: str,
    ax: Axes | None = None,
) -> Axes:
    """Plot one metric's train vs. validation curve from XGBoost's evals_result.

    The gap between the curves is a read on variance; the floor they reach is a
    read on bias (see knowledge/05).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for split in ("train", "val"):
        if split in evals_result and metric in evals_result[split]:
            history = evals_result[split][metric]
            ax.plot(range(len(history)), history, label=split)
    ax.set_xlabel("boosting round")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning curve ({metric})")
    ax.legend()
    return ax
