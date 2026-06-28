"""Stage 0 tests: seed reproducibility and config loading/validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from xgblearn.config import (
    DataConfig,
    ModelConfig,
    TuningConfig,
    load_config,
    set_global_seed,
)


def test_set_global_seed_is_reproducible() -> None:
    # Uses the legacy global RNG on purpose — that's exactly what
    # set_global_seed seeds, so it's what we assert is reproducible.
    set_global_seed(123)
    a = np.random.rand(5)  # noqa: NPY002
    set_global_seed(123)
    b = np.random.rand(5)  # noqa: NPY002
    assert np.allclose(a, b)


def test_load_data_config() -> None:
    cfg = load_config("configs/data.yaml", DataConfig)
    assert cfg.dataset == "breast_cancer"
    assert 0.0 < cfg.test_size < 1.0


def test_model_config_to_xgb_params() -> None:
    cfg = load_config("configs/model_baseline.yaml", ModelConfig)
    params = cfg.to_xgb_params()
    assert params["tree_method"] == "hist"
    assert params["device"] in {"cuda", "cpu"}
    assert "learning_rate" in params  # came from the nested `params` block


def test_tuning_config_loads() -> None:
    cfg = load_config("configs/tuning.yaml", TuningConfig)
    assert cfg.n_trials > 0
    assert "learning_rate" in cfg.search_space


def test_unknown_config_key_is_rejected() -> None:
    # extra="forbid" should turn a typo'd key into a loud error.
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"dataset": "x", "max_dpeth": 5})
