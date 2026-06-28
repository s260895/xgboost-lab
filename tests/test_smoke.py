"""Stage 0 smoke tests: XGBoost trains on CPU; GPU path when available."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

N = 500


def _toy_binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((N, 8), dtype=np.float32)
    y = (x[:, 0] + x[:, 1] > 1.0).astype(np.int32)
    return x, y


def test_cpu_training_smoke() -> None:
    x, y = _toy_binary()
    dtrain = xgb.QuantileDMatrix(x, label=y)
    bst = xgb.train(
        {"device": "cpu", "tree_method": "hist", "objective": "binary:logistic"},
        dtrain,
        num_boost_round=10,
    )
    preds = bst.predict(dtrain)
    assert preds.shape == (N,)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()


@pytest.mark.gpu
def test_gpu_training_smoke() -> None:
    x, y = _toy_binary()
    dtrain = xgb.QuantileDMatrix(x, label=y)
    bst = xgb.train(
        {"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"},
        dtrain,
        num_boost_round=10,
    )
    assert bst.predict(dtrain).shape == (N,)
