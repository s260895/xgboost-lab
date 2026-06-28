"""Train a baseline XGBoost model and log it to MLflow.

Config-driven: dataset/splitting from ``configs/data.yaml``, model from
``configs/model_baseline.yaml``. Trains with the scikit-learn API, the native
API, or both, evaluates on validation and the (once-touched) test set, and logs
each run via the manual-logging path.

Examples:
    python scripts/train.py                     # both APIs, baseline config
    python scripts/train.py --api sklearn
    python scripts/train.py --register          # also register the model
    make train
"""

from __future__ import annotations

import argparse

from xgblearn.config import DataConfig, ModelConfig, load_config, set_global_seed
from xgblearn.data.loaders import load_dataset
from xgblearn.data.splits import Splits, split_dataset
from xgblearn.models.evaluate import binary_classification_metrics
from xgblearn.models.train import (
    TrainResult,
    predict_proba_binary,
    train_native,
    train_sklearn,
)
from xgblearn.tracking.mlflow_utils import (
    data_fingerprint,
    get_git_commit,
    log_run,
    setup_experiment,
)

_TRAINERS = {"sklearn": train_sklearn, "native": train_native}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api",
        choices=["sklearn", "native", "both"],
        default="both",
        help="Which XGBoost API to train with.",
    )
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--model-config", default="configs/model_baseline.yaml")
    p.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (default: xgblab-<dataset>-baseline).",
    )
    p.add_argument(
        "--register",
        action="store_true",
        help="Register the model (needs a tracking server with a backend store).",
    )
    return p.parse_args()


def _train_and_log(
    api: str,
    splits: Splits,
    model_cfg: ModelConfig,
    dataset_name: str,
    fingerprint: str,
    register: bool,
) -> tuple[TrainResult, dict[str, float], str]:
    result = _TRAINERS[api](splits, model_cfg)
    ec = model_cfg.enable_categorical

    val_proba = predict_proba_binary(result, splits.X_val, enable_categorical=ec)
    test_proba = predict_proba_binary(result, splits.X_test, enable_categorical=ec)
    metrics = {
        **binary_classification_metrics(
            splits.y_val.to_numpy(), val_proba, prefix="val_"
        ),
        **binary_classification_metrics(
            splits.y_test.to_numpy(), test_proba, prefix="test_"
        ),
    }
    if result.best_iteration is not None:
        metrics["best_iteration"] = float(result.best_iteration)

    params: dict[str, object] = {
        "api": api,
        "objective": model_cfg.objective,
        "eval_metric": ",".join(model_cfg.eval_metric),
        "device": model_cfg.device,
        "tree_method": model_cfg.tree_method,
        "num_boost_round": model_cfg.num_boost_round,
        "early_stopping_rounds": model_cfg.early_stopping_rounds,
        "enable_categorical": model_cfg.enable_categorical,
        "seed": model_cfg.seed,
        **model_cfg.params,
    }
    tags = {
        "git_commit": get_git_commit() or "unknown",
        "dataset": dataset_name,
        "data_fingerprint": fingerprint,
        "api": api,
    }

    sample = splits.X_train.head(5)
    sample_pred = predict_proba_binary(result, sample, enable_categorical=ec)
    run_id = log_run(
        run_name=f"{dataset_name}-baseline-{api}",
        model=result.model,
        params=params,
        metrics=metrics,
        input_example=sample,
        predictions_example=sample_pred,
        evals_result=result.evals_result,
        tags=tags,
        registered_model_name=f"xgblab-{dataset_name}" if register else None,
    )
    return result, metrics, run_id


def main() -> int:
    args = parse_args()

    data_cfg = load_config(args.data_config, DataConfig)
    model_cfg = load_config(args.model_config, ModelConfig)
    set_global_seed(data_cfg.seed)

    dataset = load_dataset(data_cfg.dataset, data_cfg.raw_dir)
    if not dataset.is_classification:
        raise SystemExit(
            f"{dataset.name} is a {dataset.task} task; this baseline script "
            "handles binary classification (see scripts/evaluate.py for regression)."
        )
    splits = split_dataset(dataset, data_cfg)
    fingerprint = data_fingerprint(dataset.X, dataset.y)

    experiment = args.experiment or f"xgblab-{dataset.name}-baseline"
    setup_experiment(experiment)

    print(
        f"dataset      : {dataset.name}  ({dataset.n_samples} rows, {dataset.n_features} feats)"
    )
    print(f"splits       : {splits.sizes}")
    print(f"experiment   : {experiment}")
    print(f"fingerprint  : {fingerprint}")
    print("-" * 60)

    apis = ["sklearn", "native"] if args.api == "both" else [args.api]
    for api in apis:
        _, metrics, run_id = _train_and_log(
            api, splits, model_cfg, dataset.name, fingerprint, args.register
        )
        print(f"[{api:7s}] run={run_id}")
        print(
            f"          val : auc={metrics['val_roc_auc']:.4f} "
            f"pr_auc={metrics['val_pr_auc']:.4f} logloss={metrics['val_logloss']:.4f}"
        )
        print(
            f"          test: auc={metrics['test_roc_auc']:.4f} "
            f"pr_auc={metrics['test_pr_auc']:.4f} logloss={metrics['test_logloss']:.4f}"
        )

    print("-" * 60)
    print("Done. View runs with:  make mlflow-ui   (then open http://localhost:8080)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
