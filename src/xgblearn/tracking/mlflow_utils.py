"""MLflow tracking utilities.

Design decision (see README): **autolog for exploration, manual logging for the
final model.** MLflow's autolog is certified only for ``xgboost <= 3.2.0`` and we
run 3.3.0, so we treat autolog as a convenience for iteration and make manual
logging the reproducible production path. A manually logged run carries:

* explicit params + metrics,
* a model **signature** (inferred input/output schema) and an ``input_example``,
* a logged model in **JSON** format (``model_format="json"``) for cross-version
  portability (the legacy binary format was removed in 3.1),
* reproducibility tags: git commit, seed, and a data fingerprint.
"""

from __future__ import annotations

import hashlib
import subprocess
import warnings
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from xgblearn.config import PROJECT_ROOT

# MLflow 3.x defaults its tracking store to sqlite (./mlflow.db), not the legacy
# ./mlruns file store. We pin that explicitly so `make train` and `make
# mlflow-ui` always agree, and so the Model Registry (which needs a DB backend)
# works out of the box. Artifacts still land under ./mlruns/<experiment>/.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def get_git_commit() -> str | None:
    """Current commit SHA, or ``None`` if not a git repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def data_fingerprint(X: pd.DataFrame, y: pd.Series | None = None) -> str:
    """A short, stable hash of the data — log it so a run is tied to its inputs.

    Uses pandas' row-hashing (order-sensitive), so a different split or row order
    yields a different fingerprint. Truncated to 16 hex chars for readability.
    """
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    if y is not None:
        h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()[:16]


def setup_experiment(name: str, tracking_uri: str | None = None) -> None:
    """Point MLflow at an experiment (and optionally a tracking server).

    With no ``tracking_uri`` this uses :data:`DEFAULT_TRACKING_URI`
    (``sqlite:///mlflow.db``), which the ``make mlflow-ui`` target reads.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(name)


def enable_autolog_guarded() -> bool:
    """Best-effort ``mlflow.xgboost.autolog()``; returns whether it enabled.

    We force it on despite xgboost 3.3.0 being outside the certified band, but
    swallow failures so exploration code never crashes on the autolog path.
    """
    try:
        mlflow.xgboost.autolog(disable_for_unsupported_versions=False, silent=True)
        return True
    except Exception as exc:  # pragma: no cover - depends on version interplay
        warnings.warn(f"mlflow autolog unavailable: {exc}", stacklevel=2)
        return False


def log_run(
    *,
    run_name: str,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    input_example: pd.DataFrame,
    predictions_example: np.ndarray,
    tags: dict[str, str] | None = None,
    model_format: str = "json",
    registered_model_name: str | None = None,
) -> str:
    """Manually log one run (the production path) and return its run id.

    ``mlflow.xgboost.log_model`` accepts both a native ``Booster`` and a
    scikit-learn estimator, so this works for either API.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # log_params stringifies non-scalar values (e.g. an eval_metric list).
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        signature = infer_signature(input_example, predictions_example)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            model_format=model_format,
            registered_model_name=registered_model_name,
        )
        return run.info.run_id
