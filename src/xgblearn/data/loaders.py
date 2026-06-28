"""Dataset loaders with on-disk caching.

Each dataset in the learning path is exposed through a single entry point,
:func:`load_dataset`, which returns a typed :class:`Dataset` bundle (features,
target, and the metadata downstream code needs: task type, target name, and
which columns are categorical).

Caching: the assembled frames are written to ``data/raw/<name>_{X,y}.parquet``
on first load and read back afterwards. For sklearn's bundled datasets this is
mostly to establish the pattern; for the OpenML datasets in later stages
(Adult, Credit Card Fraud) it avoids re-downloading on every run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer

from xgblearn.config import PROJECT_ROOT

Task = Literal["binary", "multiclass", "regression"]


@dataclass
class Dataset:
    """A loaded dataset plus the metadata modeling code needs."""

    X: pd.DataFrame
    y: pd.Series
    name: str
    task: Task
    target_name: str
    categorical_features: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")


@dataclass(frozen=True)
class _Spec:
    """Static metadata + a builder for one dataset."""

    task: Task
    target_name: str
    categorical_features: list[str]
    builder: Callable[[], tuple[pd.DataFrame, pd.Series]]


def _build_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """Breast Cancer Wisconsin — 569 samples, 30 numeric features, binary."""
    bunch = load_breast_cancer(as_frame=True)
    x: pd.DataFrame = bunch.data
    # sklearn labels: 0 = malignant, 1 = benign. Keep as int.
    y: pd.Series = bunch.target.astype("int64")
    y.name = "target"
    return x, y


def _build_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """California Housing — ~20,640 rows, 8 numeric features, regression.

    Target is median house value in units of $100,000. Downloaded by sklearn on
    first use (then cached to data/raw/ as parquet).
    """
    bunch = fetch_california_housing(as_frame=True)
    x: pd.DataFrame = bunch.data
    y: pd.Series = bunch.target.astype("float64")
    y.name = "MedHouseVal"
    return x, y


# Registry of datasets the repo knows how to load. Later stages append to this.
_REGISTRY: dict[str, _Spec] = {
    "breast_cancer": _Spec(
        task="binary",
        target_name="target",
        categorical_features=[],
        builder=_build_breast_cancer,
    ),
    "california_housing": _Spec(
        task="regression",
        target_name="MedHouseVal",
        categorical_features=[],
        builder=_build_california_housing,
    ),
}


def available_datasets() -> list[str]:
    """Return the dataset keys known to :func:`load_dataset`."""
    return sorted(_REGISTRY)


def _resolve_raw_dir(raw_dir: str | Path) -> Path:
    path = Path(raw_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cache_paths(raw: Path, name: str) -> tuple[Path, Path]:
    return raw / f"{name}_X.parquet", raw / f"{name}_y.parquet"


def load_dataset(
    name: str,
    raw_dir: str | Path = "data/raw",
    *,
    use_cache: bool = True,
) -> Dataset:
    """Load a dataset by key, caching the raw frames to ``raw_dir``.

    Args:
        name: a key from :func:`available_datasets` (e.g. ``"breast_cancer"``).
        raw_dir: cache directory (relative paths are resolved under the repo root).
        use_cache: read from / write to the parquet cache. Set ``False`` to force
            a fresh build (useful in tests).
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {available_datasets()}")
    spec = _REGISTRY[name]
    raw = _resolve_raw_dir(raw_dir)
    x_path, y_path = _cache_paths(raw, name)

    if use_cache and x_path.exists() and y_path.exists():
        x = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)[spec.target_name]
    else:
        x, y = spec.builder()
        raw.mkdir(parents=True, exist_ok=True)
        x.to_parquet(x_path)
        y.to_frame().to_parquet(y_path)

    # Ensure declared categoricals carry pandas' `category` dtype so XGBoost's
    # native categorical support (enable_categorical=True) kicks in downstream.
    for col in spec.categorical_features:
        x[col] = x[col].astype("category")

    return Dataset(
        X=x,
        y=y,
        name=name,
        task=spec.task,
        target_name=spec.target_name,
        categorical_features=list(spec.categorical_features),
    )
