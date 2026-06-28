"""Shared pytest fixtures and GPU auto-skip.

Tests marked ``@pytest.mark.gpu`` are skipped automatically when no working
CUDA GPU is detected, so the suite is green on CPU-only machines and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb


def _gpu_available() -> bool:
    """Return True only if a CUDA wheel is installed AND a GPU train succeeds."""
    try:
        if not xgb.build_info().get("USE_CUDA", False):
            return False
        rng = np.random.default_rng(0)
        x = rng.random((64, 3), dtype=np.float32)
        d = xgb.QuantileDMatrix(x, label=(x[:, 0] > 0.5).astype(np.int32))
        xgb.train(
            {"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"},
            d,
            num_boost_round=1,
        )
        return True
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="no working CUDA GPU detected")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
