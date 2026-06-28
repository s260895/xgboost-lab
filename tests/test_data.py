"""Stage 1 data-layer tests: loaders, caching, and leakage-safe splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from xgblearn.data.loaders import available_datasets, load_dataset
from xgblearn.data.splits import train_val_test_split


def test_breast_cancer_available() -> None:
    assert "breast_cancer" in available_datasets()


def test_load_breast_cancer_shape(tmp_path: Path) -> None:
    ds = load_dataset("breast_cancer", raw_dir=tmp_path)
    assert ds.task == "binary"
    assert ds.n_samples == 569
    assert ds.n_features == 30
    assert set(ds.y.unique()) <= {0, 1}
    assert len(ds.X) == len(ds.y)


def test_loader_writes_and_reads_cache(tmp_path: Path) -> None:
    # First load builds + writes the parquet cache.
    load_dataset("breast_cancer", raw_dir=tmp_path)
    assert (tmp_path / "breast_cancer_X.parquet").exists()
    assert (tmp_path / "breast_cancer_y.parquet").exists()

    # Second load reads from cache and yields identical data.
    a = load_dataset("breast_cancer", raw_dir=tmp_path)
    b = load_dataset("breast_cancer", raw_dir=tmp_path)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)


def test_split_partitions_are_disjoint_and_complete(tmp_path: Path) -> None:
    ds = load_dataset("breast_cancer", raw_dir=tmp_path)
    s = train_val_test_split(
        ds.X, ds.y, test_size=0.2, val_size=0.2, stratify=True, seed=42
    )

    idx_train = set(s.X_train.index)
    idx_val = set(s.X_val.index)
    idx_test = set(s.X_test.index)

    # No row appears in more than one partition (no leakage across splits).
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    # Partitions together cover the whole dataset exactly once.
    assert len(idx_train) + len(idx_val) + len(idx_test) == ds.n_samples


def test_split_is_stratified(tmp_path: Path) -> None:
    ds = load_dataset("breast_cancer", raw_dir=tmp_path)
    s = train_val_test_split(
        ds.X, ds.y, test_size=0.2, val_size=0.2, stratify=True, seed=42
    )
    full_rate = ds.y.mean()
    # Class balance preserved within a few percent in every split.
    for y_part in (s.y_train, s.y_val, s.y_test):
        assert abs(y_part.mean() - full_rate) < 0.05
