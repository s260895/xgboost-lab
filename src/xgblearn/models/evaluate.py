"""Evaluation metrics.

Stage 1 covers binary-classification metrics. Stage 2 extends this module with
regression metrics, residual diagnostics, learning curves, and calibration.

Why several metrics and not just accuracy: on imbalanced data accuracy is
misleading (Stage 4), and a model can rank well (high ROC-AUC) yet be poorly
calibrated (high Brier). Logging the whole panel keeps us honest.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


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
