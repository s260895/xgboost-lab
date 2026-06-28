"""Stage 2 regression tests: both APIs, determinism, early stopping, monotonicity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xgblearn.config import ModelConfig
from xgblearn.data.loaders import load_dataset
from xgblearn.data.splits import Splits, train_val_test_split
from xgblearn.models.evaluate import regression_metrics
from xgblearn.models.train import predict_regression, train_native, train_sklearn


def _splits(tmp_path: Path) -> Splits:
    ds = load_dataset("california_housing", raw_dir=tmp_path)
    return train_val_test_split(
        ds.X, ds.y, test_size=0.2, val_size=0.2, stratify=False, seed=42
    )


def _cfg(
    extra: dict | None = None, rounds: int = 80, early_stop: int | None = None
) -> ModelConfig:
    return ModelConfig(
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        device="cpu",
        tree_method="hist",
        num_boost_round=rounds,
        early_stopping_rounds=early_stop,
        seed=42,
        params={"learning_rate": 0.1, "max_depth": 5, **(extra or {})},
    )


def test_regression_metrics_panel() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # Perfect predictions -> rmse/mae 0, r2 1.
    m = regression_metrics(y, y.copy())
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_regression_scores_reasonably(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    res = train_native(s, _cfg())
    preds = predict_regression(res, s.X_val)
    m = regression_metrics(s.y_val.to_numpy(), preds)
    # California housing is learnable; a basic model clears R2 > 0.7.
    assert m["r2"] > 0.7


def test_regression_both_apis_agree(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    cfg = _cfg()
    p_sk = predict_regression(train_sklearn(s, cfg), s.X_val)
    p_nat = predict_regression(train_native(s, cfg), s.X_val)
    assert np.allclose(p_sk, p_nat, atol=1e-4)


def test_regression_is_deterministic(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    cfg = _cfg()
    p1 = predict_regression(train_native(s, cfg), s.X_val)
    p2 = predict_regression(train_native(s, cfg), s.X_val)
    assert np.array_equal(p1, p2)


def test_early_stopping_regression(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    res = train_native(s, _cfg(rounds=1000, early_stop=50))
    assert res.best_iteration is not None
    assert res.best_iteration < 999  # stopped before the cap


def test_monotone_constraint_is_enforced(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    res = train_sklearn(s, _cfg(extra={"monotone_constraints": {"MedInc": 1}}))

    # 1-D partial dependence: vary MedInc, hold other features at their medians.
    base = s.X_train.median(numeric_only=True)
    grid = np.linspace(
        s.X_train["MedInc"].quantile(0.01), s.X_train["MedInc"].quantile(0.99), 50
    )
    rows = pd.DataFrame([{**base.to_dict(), "MedInc": v} for v in grid])[
        s.X_train.columns
    ]
    preds = predict_regression(res, rows)

    # Predictions must be non-decreasing in MedInc (allow tiny float noise).
    assert np.all(np.diff(preds) >= -1e-6)
