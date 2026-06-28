"""Training with both XGBoost APIs.

XGBoost ships two interfaces and this repo teaches both:

* **Native API** — ``xgb.train(params, DMatrix, num_boost_round, evals=...)``,
  returning a ``Booster``. We use ``QuantileDMatrix`` (pre-bins the data; lower
  memory; ideal on GPU). Early stopping is a callback.
* **Scikit-learn API** — ``XGBClassifier`` / ``XGBRegressor`` with
  ``.fit(X, y, eval_set=...)``. Plays nicely with sklearn Pipelines and
  search CV. ``QuantileDMatrix`` + ``inplace_predict`` are auto-enabled.

Alias mapping (the same knob, two names):
    learning_rate = eta        n_estimators  = num_boost_round
    reg_alpha     = alpha       reg_lambda    = lambda
    min_split_loss = gamma

Both code paths default to ``device="cuda"``, ``tree_method="hist"`` (set via the
:class:`~xgblearn.config.ModelConfig`). XGBoost early-stops on the **last**
``eval_metric`` by convention; we make that explicit in the native path so the
two APIs behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from xgblearn.config import ModelConfig
from xgblearn.data.splits import Splits


@dataclass
class TrainResult:
    """Outcome of a training run, normalized across the two APIs."""

    model: Any  # xgb.Booster (native) or XGBModel (sklearn)
    api: str  # "native" | "sklearn"
    evals_result: dict[str, dict[str, list[float]]]
    best_iteration: int | None
    best_score: float | None


def _is_classification(objective: str) -> bool:
    return objective.startswith(("binary:", "multi:"))


def train_sklearn(splits: Splits, cfg: ModelConfig) -> TrainResult:
    """Train via the scikit-learn estimator API.

    ``early_stopping_rounds`` is passed to the **constructor** (moved there in
    XGBoost 1.6+); predictions then automatically use ``best_iteration``.
    """
    estimator_cls = XGBClassifier if _is_classification(cfg.objective) else XGBRegressor
    model = estimator_cls(
        n_estimators=cfg.num_boost_round,  # == num_boost_round
        objective=cfg.objective,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        device=cfg.device,
        tree_method=cfg.tree_method,
        enable_categorical=cfg.enable_categorical,
        random_state=cfg.seed,
        **cfg.params,  # learning_rate, max_depth, subsample, ...
    )
    model.fit(
        splits.X_train,
        splits.y_train,
        # First eval_set is "validation_0" (train), second is "validation_1"
        # (val). Early stopping watches the LAST one.
        eval_set=[(splits.X_train, splits.y_train), (splits.X_val, splits.y_val)],
        verbose=False,
    )
    best_it = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return TrainResult(
        model=model,
        api="sklearn",
        evals_result=model.evals_result(),
        best_iteration=int(best_it) if best_it is not None else None,
        best_score=float(best_score) if best_score is not None else None,
    )


def train_native(splits: Splits, cfg: ModelConfig) -> TrainResult:
    """Train via the native ``xgb.train`` API using ``QuantileDMatrix``.

    The validation matrix references the training matrix (``ref=dtrain``) so the
    two share the same bin edges — the correct way to build eval matrices for
    ``QuantileDMatrix``.
    """
    ec = cfg.enable_categorical
    dtrain = xgb.QuantileDMatrix(
        splits.X_train, label=splits.y_train, enable_categorical=ec
    )
    dval = xgb.QuantileDMatrix(
        splits.X_val, label=splits.y_val, ref=dtrain, enable_categorical=ec
    )

    callbacks: list[xgb.callback.TrainingCallback] = []
    if cfg.early_stopping_rounds:
        # Drive early stopping by the last configured metric (XGBoost's own
        # default), matching the sklearn path. save_best=True truncates the
        # booster to its best iteration so predict() needs no iteration_range.
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=cfg.early_stopping_rounds,
                metric_name=cfg.eval_metric[-1],
                data_name="val",
                save_best=True,
            )
        )

    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        cfg.to_xgb_params(),
        dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        callbacks=callbacks or None,
        verbose_eval=False,
    )

    best_it = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)
    return TrainResult(
        model=booster,
        api="native",
        evals_result=evals_result,
        best_iteration=int(best_it) if best_it is not None else None,
        best_score=float(best_score) if best_score is not None else None,
    )


def predict_proba_binary(
    result: TrainResult,
    X: pd.DataFrame,
    *,
    enable_categorical: bool = False,
) -> np.ndarray:
    """Positive-class probabilities for binary classification, either API.

    sklearn predictions already use ``best_iteration``; for the native path,
    ``save_best=True`` truncated the booster, so a plain ``predict`` is correct.
    """
    if result.api == "sklearn":
        return np.asarray(result.model.predict_proba(X))[:, 1]
    # Plain DMatrix is the standard container for inference (QuantileDMatrix is
    # really a training-time construct that wants a `ref` outside of fit).
    dmat = xgb.DMatrix(X, enable_categorical=enable_categorical)
    return np.asarray(result.model.predict(dmat))
