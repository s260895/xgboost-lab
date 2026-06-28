"""Configuration and reproducibility utilities.

This module is the backbone of every run in the repo. It does two jobs:

1. **Seed management** — one place to make a run reproducible.
2. **Config loading** — typed, validated YAML configs via pydantic, so a typo
   in a config file fails loudly at load time instead of silently training the
   wrong model.

Design choice: configs live in ``configs/*.yaml`` and are loaded into pydantic
models. ``extra="forbid"`` means an unknown key (a typo like ``max_dpeth``)
raises immediately rather than being ignored.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

# Repo root, resolved from this file's location: src/xgblearn/config.py -> repo/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_global_seed(seed: int = 42) -> None:
    """Seed Python's ``random``, NumPy, and ``PYTHONHASHSEED``.

    Why this matters: reproducibility is a first-class feature of this repo. We
    seed the standard sources here and pass a model-level ``seed`` to XGBoost
    separately (see :meth:`ModelConfig.to_xgb_params`).

    Caveat worth teaching: GPU histogram construction can introduce tiny,
    run-to-run floating-point differences even with a fixed seed. CPU training
    with a fixed seed is bit-for-bit reproducible; GPU is "close but not always
    identical". We always set + log the seed regardless so a run is as
    reproducible as the hardware allows.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seed the *legacy* global NumPy RNG on purpose: lots of library code
    # (sklearn internals, older snippets) still calls np.random.* directly, and
    # this makes those reproducible. New code should prefer np.random.default_rng.
    np.random.seed(seed)  # noqa: NPY002


class _Base(BaseModel):
    """Base for all configs: forbid unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    """How to load and split a dataset.

    The split discipline encoded here is deliberate (see README "Design
    decisions"): we carve out a **test set that is touched exactly once**, and a
    separate **validation set** that drives early stopping and tuning.
    """

    dataset: str = Field(description="Loader key, e.g. 'breast_cancer'.")
    seed: int = 42
    test_size: float = Field(0.2, ge=0.0, lt=1.0, description="Held-out test fraction.")
    val_size: float = Field(
        0.2, ge=0.0, lt=1.0, description="Validation fraction (of the non-test data)."
    )
    stratify: bool = True
    target_name: str | None = None
    raw_dir: Path = Path("data/raw")


class ModelConfig(_Base):
    """XGBoost training configuration (API-agnostic).

    ``params`` holds the tunable hyperparameters (learning_rate, max_depth, ...)
    while the structural fields (objective, device, ...) are kept explicit
    because they rarely change and shouldn't be buried in a generic dict.
    """

    objective: str = "binary:logistic"
    eval_metric: list[str] = Field(default_factory=lambda: ["logloss"])
    device: str = "cuda"  # modern GPU API; "cpu" to force CPU.
    tree_method: str = "hist"  # required for native categorical + constraints.
    num_boost_round: int = 500
    early_stopping_rounds: int | None = 50
    enable_categorical: bool = False
    seed: int = 42
    # Tunable hyperparameters: learning_rate, max_depth, min_child_weight,
    # subsample, colsample_bytree, reg_alpha, reg_lambda, gamma, ...
    params: dict[str, Any] = Field(default_factory=dict)

    def to_xgb_params(self) -> dict[str, Any]:
        """Flatten into the ``params`` dict that ``xgb.train`` / estimators expect.

        Note the alias mapping XGBoost accepts both ways:
        ``learning_rate``=``eta``, ``reg_alpha``=``alpha``,
        ``reg_lambda``=``lambda``, ``min_split_loss``=``gamma``. We use the
        sklearn-style names in configs for consistency.
        """
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "device": self.device,
            "tree_method": self.tree_method,
            "seed": self.seed,
            **self.params,
        }


class TuningConfig(_Base):
    """Optuna study configuration (exercised in Stage 5).

    The recommended tuning *order* (fix LR + early stopping; then tree
    structure; then sampling; then regularization) is documented in the spec
    and the tuning module — this config just holds the search ranges and study
    knobs.
    """

    n_trials: int = 50
    timeout: int | None = None  # seconds; None = no time limit.
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "validation-logloss"  # Optuna observation key.
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: Literal["median", "successive_halving", "none"] = "median"
    seed: int = 42
    search_space: dict[str, Any] = Field(default_factory=dict)


def load_config[ConfigT: BaseModel](path: str | Path, schema: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against a pydantic ``schema``.

    Example:
        >>> from xgblearn.config import DataConfig, load_config
        >>> cfg = load_config("configs/data.yaml", DataConfig)
        >>> cfg.dataset
        'breast_cancer'
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return schema.model_validate(raw)
