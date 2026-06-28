# xgboost-lab

> A hands-on, learning-oriented XGBoost laboratory — GPU-accelerated on NVIDIA Blackwell, with leakage-safe feature engineering, experiment tracking, and Bayesian hyperparameter tuning baked in.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="XGBoost" src="https://img.shields.io/badge/xgboost-3.3.0-orange.svg">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-CUDA%2012.x%20%7C%20Blackwell%20sm__120-76B900.svg">
  <img alt="MLflow" src="https://img.shields.io/badge/tracking-MLflow-0194E2.svg">
  <img alt="Optuna" src="https://img.shields.io/badge/tuning-Optuna-7B43A1.svg">
  <img alt="Code style" src="https://img.shields.io/badge/lint-ruff-261230.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

---

## What this is

`xgboost-lab` is a single, well-structured repository that takes XGBoost from first principles to a production-aware workflow. It is built to **teach** — every module and notebook pairs a short conceptual intro with heavily-commented, runnable code and a closing "what to notice" note — while still following the engineering practices you'd want in a real ML codebase.

It is deliberately not a minimal starter. The goal is **breadth and depth**: both XGBoost APIs, GPU acceleration on current hardware, a serious feature-engineering module, proper evaluation and interpretation, experiment tracking, and automated hyperparameter tuning — all reproducible and config-driven.

## Why it exists

Most XGBoost tutorials stop at `model.fit()`. This repo goes after the parts that actually decide whether a model is good and trustworthy:

- **Feature engineering done right** — what actually helps tree models (and what's a waste of effort), with leakage demonstrated *and* fixed.
- **GPU correctness on new hardware** — the RTX 50-series (Blackwell) tripped up a lot of ML tooling; this repo documents the working path and verifies it.
- **Honest evaluation** — calibration, imbalance-aware metrics, and the gap between SHAP and permutation importance.
- **Reproducibility** — seeds, pinned versions, config files, and tracked experiments, not loose notebooks.

## Highlights

- ⚡ **GPU-accelerated** with the modern `device="cuda"` API (the removed `gpu_hist`/`gpu_id` params are avoided), verified on an **RTX 5060 Ti (Blackwell, sm_120)**.
- 🧱 **Both XGBoost APIs** — the scikit-learn estimator API and the native `Booster` / `QuantileDMatrix` API, with the alias mapping spelled out.
- 🧪 **Leakage-safe feature engineering** — native categorical support, out-of-fold target encoding, interactions, and monotonic / interaction constraints.
- 📊 **Experiment tracking** with MLflow (autolog + a robust manual-logging path, model signatures, and the model registry).
- 🔎 **Bayesian tuning** with Optuna (TPE + pruning) and nested MLflow runs.
- 🩺 **Interpretation** with SHAP, plus a notebook reproducing the SHAP-vs-permutation-importance divergence.
- 🛠️ **Clean tooling** — `venv`, `ruff`, `mypy`, `pytest`, `pre-commit`, a `Makefile`, and pydantic-loaded YAML configs.

## Hardware & GPU notes

Built and tested on:

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX 5060 Ti (Blackwell, compute capability sm_120) |
| CUDA | 12.x runtime (bundled in the XGBoost wheel) |
| Driver | R570+ branch (any 2025–2026 Game Ready / Studio driver) |
| Python | 3.12 |
| XGBoost | 3.3.0 |

A stock `pip install xgboost` already ships native `sm_120` kernels, so **no source build or nightly is required** on Blackwell. Verify your setup any time with:

```bash
make check-gpu     # runs scripts/check_gpu.py
```

It prints the XGBoost version, confirms `USE_CUDA: True`, and runs a short GPU training loop. CPU-only machines are fully supported too — the GPU smoke test is skippable.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/xgboost-lab.git
cd xgboost-lab

# 2. Create and activate a virtual environment (venv — not conda/uv)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 4. Verify the GPU path (optional but recommended)
make check-gpu

# 5. Download the learning datasets
python scripts/download_data.py

# 6. Train a baseline and open the tracking UI
make train
make mlflow-ui                     # http://localhost:8080
```

## Repository structure

```
xgboost-lab/
├── configs/            # YAML configs (data, models, tuning) loaded via pydantic
├── data/               # raw / interim / processed (gitignored)
├── notebooks/          # numbered, didactic: EDA → modeling → interpretation
├── src/xgblearn/       # installable package
│   ├── data/           # loaders + leakage-safe splits
│   ├── features/       # categorical, transformers, selection
│   ├── models/         # train, tune (Optuna), evaluate
│   ├── interpret/      # SHAP + importance pitfalls
│   └── tracking/       # MLflow utilities
├── scripts/            # CLI entrypoints (check_gpu, download_data, train, tune, evaluate)
├── tests/              # pytest: leakage checks, shapes, determinism, GPU smoke test
└── Makefile            # setup, lint, format, typecheck, test, train, tune, mlflow-ui
```

(The importable package is `xgblearn`; the repo is `xgboost-lab` — same convention as `scikit-learn` the repo vs `sklearn` the import.)

## Learning path

The repo is built to be worked through in order, on progressively harder datasets:

1. **Breast Cancer Wisconsin** — small binary classification; end-to-end pipeline + GPU smoke test.
2. **California Housing** — regression; early stopping, learning curves, monotonic constraints.
3. **Adult / Census Income** — the feature-engineering centerpiece: native categorical vs one-hot vs target encoding, interactions, and constraints.
4. **Credit Card Fraud** — heavy imbalance: `scale_pos_weight`, PR-AUC, threshold tuning, calibration.

Each stage has a matching notebook and reuses the `src/xgblearn` modules so the lessons compound.

## Tooling

| Task | Command |
|---|---|
| Set up environment | `make setup` |
| Lint + format | `make lint` / `make format` |
| Type-check | `make typecheck` |
| Run tests | `make test` |
| Verify GPU | `make check-gpu` |
| Train baseline | `make train` |
| Tune (Optuna) | `make tune` |
| Evaluate | `make evaluate` |
| MLflow UI | `make mlflow-ui` |

## Design decisions

A few deliberate choices worth calling out, since they shape the whole repo:

- **No feature scaling for trees.** Splits are threshold-based and invariant to monotonic transforms — standardizing inputs is a linear-model habit that buys nothing here. The repo says so explicitly rather than copying the boilerplate.
- **Native categorical by default.** `enable_categorical=True` with `tree_method="hist"` over manual one-hot, with target encoding reserved for high-cardinality cases and always fit out-of-fold.
- **The test set is sacred.** Validation drives early stopping and tuning; the held-out test set is touched exactly once.
- **Autolog for exploration, manual logging for the final model.** MLflow autolog is fast for iteration, but the registered model uses explicit params, metrics, a model signature, and a logged data hash + git commit for reproducibility.
- **Tune what matters, in order.** Fix learning rate + early stopping first, then tree structure, then sampling, then regularization — rather than throwing everything into one search.

## Roadmap

- [ ] LightGBM / CatBoost comparison notebook (same datasets, same harness)
- [ ] ONNX export + a minimal inference service
- [ ] CI workflow (lint, type-check, tests) via GitHub Actions

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Acknowledgements

Built as a structured way back into ML engineering, with an emphasis on the parts that don't fit in a quickstart: feature engineering, evaluation, and reproducibility. References drawn from the official XGBoost, MLflow, and Optuna documentation.
