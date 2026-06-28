"""xgblearn — a learning-oriented, production-aware XGBoost toolkit.

This package is the reusable core behind the ``xgboost-lab`` notebooks and
scripts. Each stage of the learning path (breast cancer -> housing -> adult ->
fraud) builds on the same modules so the lessons compound:

    xgblearn.config      seed management + pydantic-loaded YAML configs
    xgblearn.data        dataset loaders + leakage-safe splits
    xgblearn.features    categorical handling, transforms, selection
    xgblearn.models      training (both APIs), Optuna tuning, evaluation
    xgblearn.interpret   SHAP + importance pitfalls
    xgblearn.tracking    MLflow experiment tracking utilities
"""

from __future__ import annotations

__version__ = "0.1.0"

# Convenience re-exports for the most-used helpers.
from xgblearn.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    TuningConfig,
    load_config,
    set_global_seed,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TuningConfig",
    "load_config",
    "set_global_seed",
    "__version__",
]
