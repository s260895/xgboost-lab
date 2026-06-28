"""Stage 1 model tests: both APIs agree, training is deterministic, GPU works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xgblearn.config import ModelConfig
from xgblearn.data.loaders import load_dataset
from xgblearn.data.splits import Splits, train_val_test_split
from xgblearn.models.evaluate import binary_classification_metrics
from xgblearn.models.train import predict_proba_binary, train_native, train_sklearn


def _splits(tmp_path: Path) -> Splits:
    ds = load_dataset("breast_cancer", raw_dir=tmp_path)
    return train_val_test_split(
        ds.X, ds.y, test_size=0.2, val_size=0.2, stratify=True, seed=42
    )


def _cfg(device: str = "cpu") -> ModelConfig:
    # Small, fast config for tests. CPU is bit-for-bit reproducible (GPU
    # histogram building can vary slightly run-to-run), so default to CPU here.
    return ModelConfig(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        device=device,
        tree_method="hist",
        num_boost_round=50,
        early_stopping_rounds=10,
        seed=42,
        params={"learning_rate": 0.1, "max_depth": 3},
    )


def test_sklearn_trains_and_scores_well(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    result = train_sklearn(s, _cfg())
    proba = predict_proba_binary(result, s.X_val)
    metrics = binary_classification_metrics(s.y_val.to_numpy(), proba)
    # Breast cancer is easy; a sane baseline clears 0.95 AUC comfortably.
    assert metrics["roc_auc"] > 0.95


def test_native_trains_and_scores_well(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    result = train_native(s, _cfg())
    proba = predict_proba_binary(result, s.X_val)
    metrics = binary_classification_metrics(s.y_val.to_numpy(), proba)
    assert metrics["roc_auc"] > 0.95


def test_both_apis_agree(tmp_path: Path) -> None:
    # Same engine, same params, same seed -> near-identical predictions.
    s = _splits(tmp_path)
    cfg = _cfg()
    p_sk = predict_proba_binary(train_sklearn(s, cfg), s.X_val)
    p_nat = predict_proba_binary(train_native(s, cfg), s.X_val)
    assert np.allclose(p_sk, p_nat, atol=1e-5)


def test_training_is_deterministic(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    cfg = _cfg()
    p1 = predict_proba_binary(train_sklearn(s, cfg), s.X_val)
    p2 = predict_proba_binary(train_sklearn(s, cfg), s.X_val)
    assert np.array_equal(p1, p2)


def test_early_stopping_truncates_native(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    result = train_native(s, _cfg())
    # With early stopping the model should stop before the 50-round cap.
    assert result.best_iteration is not None
    assert result.best_iteration < 49


@pytest.mark.gpu
def test_gpu_training_matches_cpu_closely(tmp_path: Path) -> None:
    s = _splits(tmp_path)
    p_cpu = predict_proba_binary(train_native(s, _cfg("cpu")), s.X_val)
    p_gpu = predict_proba_binary(train_native(s, _cfg("cuda")), s.X_val)
    # GPU != CPU bit-for-bit, but rankings/probabilities should be very close.
    assert np.corrcoef(p_cpu, p_gpu)[0, 1] > 0.99
