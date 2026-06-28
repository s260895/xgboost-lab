"""Leakage-safe data splitting and cross-validation helpers.

The cardinal rule (see README "Design decisions"): **the test set is sacred**.
We carve out a held-out test set that is touched exactly once, plus a separate
validation set that drives early stopping and tuning. Any fitting transform
(encoders, imputers, target encoding) must be fit on **train only** — this
module just produces the splits; the feature modules (Stage 3) enforce the
fit-on-train discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from xgblearn.config import DataConfig
from xgblearn.data.loaders import Dataset, Task


@dataclass
class Splits:
    """Train / validation / test partitions of a dataset."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "val": len(self.X_val),
            "test": len(self.X_test),
        }


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
) -> Splits:
    """Split into train / val / test in two stages.

    ``val_size`` is a fraction of the *non-test* data, so with the defaults
    (test=0.2, val=0.2) the overall proportions are train 64% / val 16% /
    test 20%. Stratification (classification only) preserves the class balance
    in every split.
    """
    strat_full = y if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat_full, random_state=seed
    )

    strat_tv = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=strat_tv,
        random_state=seed,
    )

    return Splits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def split_dataset(dataset: Dataset, cfg: DataConfig) -> Splits:
    """Split a :class:`Dataset` using a :class:`DataConfig`.

    Stratification is applied only for classification tasks (it's meaningless
    for continuous regression targets).
    """
    return train_val_test_split(
        dataset.X,
        dataset.y,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        stratify=cfg.stratify and dataset.is_classification,
        seed=cfg.seed,
    )


def make_cv(
    task: Task,
    n_splits: int = 5,
    *,
    seed: int = 42,
    shuffle: bool = True,
) -> StratifiedKFold | KFold:
    """Return an appropriate CV splitter: stratified for classification, plain
    KFold for regression. (Stage 3 adds Group/TimeSeries variants where entities
    repeat or order matters.)
    """
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
