# CLAUDE.md / Project Specification: Comprehensive Learning-Oriented XGBoost Repository

## 0. Purpose of This Document
This is a context/specification document to hand directly to **Claude Code** so it can build a complete, learning-oriented XGBoost machine-learning repository **from scratch** as a fresh git repo. The user is an experienced software engineer (6+ years, MSc Data Science) returning to ML/data science, starting with XGBoost. The objective is **maximal learning coverage** — comprehensive, heavily-commented, didactic code and notebooks — not a minimal starter. Build out everything described below; favor explanatory docstrings and markdown that *teach* the concepts.

**Target environment:** Windows or WSL2/Linux with an NVIDIA **RTX 5060 Ti** (Blackwell, compute capability sm_120). **Python 3.12** managed with **`venv`** (explicitly NOT conda, NOT uv). Stable versions as of June 24, 2026: **xgboost 3.3.0**, MLflow ≥2.19, Optuna ≥4.0, scikit-learn ≥1.5, SHAP ≥0.46.

---

## 1. Critical Environment / GPU Facts (verified against primary sources)

- **XGBoost 3.3.0** was released **June 17, 2026** and **requires Python ≥3.12**, supporting Python 3.12 / 3.13 / 3.14 (PyPI, `pypi.org/project/xgboost`, Author: Hyunsu Cho).
- **GOOD NEWS — XGBoost GPU "just works" on the RTX 5060 Ti / Blackwell.** Unlike PyTorch (whose stable wheels shipped only up to sm_90 and required special `cu128` nightly builds for sm_120), a **stock `pip install xgboost` works on Blackwell**. The verbatim build logic in `dmlc/xgboost` `cmake/Utils.cmake` (master) is:
  - CUDA ≥12.8: `set(CMAKE_CUDA_ARCHITECTURES 50 60 70 80 90 100 120)`
  - CUDA ≥13.0: `set(CMAKE_CUDA_ARCHITECTURES 75 80 90 100 120)` (CUDA 13 dropped sm_50/60/70)
  
  Because `120` (sm_120) is explicitly in the list, the default CUDA-12 wheel contains **native sm_120 cubin kernels**, plus PTX for the newest arch for forward-compatibility. No nightly, no source build needed. No dmlc/xgboost user issue reporting "no kernel image is available" on RTX 50-series was found — consistent with "just works."
- **Requirements:** XGBoost 3.x requires **CUDA 12.0+ runtime** and **compute capability ≥5.0**. You do **NOT** need a locally installed CUDA toolkit — the wheel bundles the CUDA runtime. You only need a recent **NVIDIA driver** (R570 branch or newer for Blackwell; any 2025–2026 Game Ready/Studio driver qualifies). On **Windows** you also need the **Visual C++ Redistributable** (for `vcomp140.dll`/OpenMP).
- **Package variants:** `xgboost` (default, CUDA 12, includes sm_120 — recommended) | `xgboost-cu13` (CUDA 13 build, added in v3.2.0) | `xgboost-cpu` (no GPU, smaller footprint). On WSL2/Linux the GPU wheel is `manylinux_2_28` (needs glibc ≥2.28); the older `manylinux2014` variant has **no GPU support**. Multi-GPU training is Linux-only (irrelevant for a single 5060 Ti).
- **Modern GPU API:** Use `device="cuda"` + `tree_method="hist"`. The old `tree_method="gpu_hist"`, `gpu_id`, `gpu_coord_descent`, and `predictor` parameters were **deprecated in 2.0 and REMOVED in 3.1.0** (per the XGBoost 3.1.0 release notes: *"Removed old GPU-related parameters including use_gpu (pyspark), gpu_id, gpu_hist, and gpu_coord_descent. These parameters have been deprecated in 2.0. Use the device parameter instead."*). Do not use them.
- **Verify GPU:** `xgb.build_info()` should show `USE_CUDA: True`; then train with `device="cuda"` and confirm no "no kernel image" error.
- **Watch for** the benign warning *"Falling back to prediction using DMatrix due to mismatched devices … running on cuda:0, while input data is on cpu."* This is a data-placement issue (data on CPU, booster on GPU during predict), **not** a Blackwell problem. Avoid via `QuantileDMatrix` / consistent device placement / `inplace_predict` with GPU arrays.

### Verification script — create `scripts/check_gpu.py`
```python
import numpy as np, xgboost as xgb
print("xgboost", xgb.__version__)            # expect 3.3.0
print(xgb.build_info())                       # USE_CUDA should be True
X = np.random.rand(200_000, 50).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
dtrain = xgb.QuantileDMatrix(X, label=y)
params = {"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"}
bst = xgb.train(params, dtrain, num_boost_round=50)
print("OK — GPU training completed")
```

---

## 2. Repository Structure
Build this layout (based on Cookiecutter Data Science v2 conventions, adapted for a learning repo):

```
xgboost-learning/
├── README.md                  # Project overview, setup, learning path
├── CLAUDE.md                  # This spec (keep a copy in-repo)
├── pyproject.toml             # Project metadata + tool config (ruff, mypy, pytest)
├── requirements.txt           # Pinned deps (venv-friendly, NOT conda/uv)
├── requirements-dev.txt       # Dev deps
├── Makefile                   # Task runner: setup, lint, format, test, train, mlflow-ui
├── .pre-commit-config.yaml    # ruff (lint+format), mypy, basic hooks
├── .gitignore                 # ignore data/, mlruns/, models/, .venv/, __pycache__
├── .python-version            # 3.12
├── configs/                   # YAML configs
│   ├── data.yaml
│   ├── model_baseline.yaml
│   ├── model_tuned.yaml
│   └── tuning.yaml
├── data/
│   ├── raw/                   # immutable downloads (gitignored)
│   ├── interim/               # intermediate transforms
│   └── processed/             # final modeling datasets
├── notebooks/                 # numbered, didactic: 0.x EDA ... 5.x interpretation
├── src/xgblearn/              # installable package (pip install -e .)
│   ├── __init__.py
│   ├── config.py              # pydantic config loader, seed management
│   ├── data/
│   │   ├── loaders.py         # sklearn/OpenML dataset loaders w/ caching
│   │   └── splits.py          # train/val/test, stratified & grouped CV
│   ├── features/
│   │   ├── categorical.py     # native categorical, target encoding (CV-safe)
│   │   ├── transformers.py    # numeric transforms, interactions
│   │   └── selection.py       # gain vs SHAP-based selection
│   ├── models/
│   │   ├── train.py           # native + sklearn API training
│   │   ├── tune.py            # Optuna + pruning + MLflow
│   │   └── evaluate.py        # metrics, calibration, learning curves
│   ├── interpret/
│   │   └── shap_analysis.py   # SHAP + importance pitfalls
│   └── tracking/
│       └── mlflow_utils.py    # experiment setup, signatures, registry
├── scripts/                   # CLI entrypoints
│   ├── check_gpu.py
│   ├── download_data.py
│   ├── train.py
│   ├── tune.py
│   └── evaluate.py
├── tests/                     # pytest: leakage checks, shapes, determinism
├── models/                    # serialized models (gitignored)
└── mlruns/                    # MLflow local store (gitignored)
```

---

## 3. Dependencies

**requirements.txt**
```
xgboost==3.3.0
scikit-learn>=1.5,<2
pandas>=2.2,<3
numpy>=1.26
mlflow>=2.19
optuna>=4.0
optuna-integration>=4.0   # XGBoostPruningCallback now lives here
shap>=0.46
matplotlib>=3.9
seaborn>=0.13
pyyaml>=6.0
pydantic>=2.7
pyarrow>=16
category_encoders>=2.6    # for target/GLMM encoding demos
```
**requirements-dev.txt:** `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, `ipykernel`, `jupyterlab`.

**Setup (venv only):**
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

---

## 4. Core XGBoost Learning Content (document in code + notebooks)

### 4.1 Gradient boosting fundamentals
- Additive ensemble of shallow trees, each fit to the gradient (and hessian) of the loss of the current ensemble. Contrast with **Random Forest** (bagging: parallel, independent deep trees averaged) vs **boosting** (sequential: each tree corrects prior errors).
- XGBoost specifics to teach: second-order (Newton) optimization using gradients **and** hessians; regularized objective (L1/L2 on leaf weights + complexity penalty γ); grows trees to `max_depth` then prunes by gain (vs classic GBM early stop); **sparsity-aware split finding** with a learned default direction for missing values; histogram-based binning (`hist`).
- vs **LightGBM:** leaf-wise growth + GOSS + EFB, often faster on large data; XGBoost defaults to depth/level-wise. vs **CatBoost:** ordered boosting + sophisticated native categorical (ordered target statistics), strong on heavy categoricals out-of-the-box. Document these tradeoffs in a comparison notebook.

### 4.2 Two APIs (teach both)
- **Native API:** `xgb.train(params, DMatrix/QuantileDMatrix, num_boost_round, evals=...)`. Use `QuantileDMatrix` for memory efficiency (pre-bins data; ideal on GPU). Returns a `Booster`.
- **Scikit-learn API:** `XGBClassifier` / `XGBRegressor` with `.fit(X, y, eval_set=...)`. Integrates with sklearn Pipelines, `GridSearchCV`, etc. `QuantileDMatrix` and `inplace_predict` are auto-enabled. `n_estimators` ≡ `num_boost_round`.
- Document the alias mapping: `learning_rate`≡`eta`, `n_estimators`≡`num_boost_round`, `reg_alpha`≡`alpha`, `reg_lambda`≡`lambda`, `min_split_loss`≡`gamma`.
- Note: `booster=gblinear` is **deprecated since version 3.3.0** (XGBoost 3.3.0 Parameters docs) and will be removed in a future release — focus on `gbtree` (and mention `dart`).

### 4.3 Key hyperparameters (document what each does)
- `learning_rate`/`eta` (0.01–0.3): step shrinkage; lower = more robust but needs more rounds.
- `n_estimators`/`num_boost_round`: number of trees; pair with early stopping.
- `max_depth` (3–10 typical): tree complexity; main overfitting lever.
- `min_child_weight`: min sum of hessian in a child; higher = more conservative.
- `gamma`/`min_split_loss`: min loss reduction to split; higher = more conservative.
- `subsample` (0.5–1): row sampling per tree.
- `colsample_bytree`/`colsample_bylevel`/`colsample_bynode`: column sampling.
- `reg_alpha` (L1), `reg_lambda` (L2): regularization on leaf weights.
- `max_bin`: histogram bins (accuracy vs speed/memory).
- `scale_pos_weight`: imbalance handling (≈ neg/pos ratio).
- `max_delta_step`: useful for extreme imbalance / logistic.
- **Recommended tuning ORDER:** (1) fix a moderate learning_rate + use early stopping to find num_boost_round; (2) tune tree structure (max_depth, min_child_weight, gamma); (3) tune sampling (subsample, colsample_*); (4) tune regularization (alpha, lambda); (5) lower learning_rate and re-fit with more trees.

### 4.4 Early stopping
- **sklearn API:** pass `early_stopping_rounds` to the **constructor** (moved there in 1.6+) and `eval_set` to `.fit()`. Predictions automatically use `best_iteration`.
- **Native API:** `xgb.callback.EarlyStopping(rounds=..., save_best=True, metric_name=..., data_name=...)`.
- Use ~10% of total rounds as patience (e.g., 50 rounds for 1000 trees). Requires a validation set **distinct from the test set**.

### 4.5 Objectives & eval metrics
- Binary: `binary:logistic`; metrics `logloss`, `auc`, `aucpr`, `error`.
- Multiclass: `multi:softprob`; `mlogloss`, `merror`.
- Regression: `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:quantileerror`; metrics `rmse`, `mae`.
- Ranking: `rank:ndcg`, `rank:pairwise` (rewritten LTR implementation in 2.0+, with `lambdarank_pair_method`, unbiased LTR option).
- Count/survival: `count:poisson`, `survival:cox`/`survival:aft`.

### 4.6 Imbalanced data
- `scale_pos_weight` ≈ (# negatives / # positives) as a starting point.
- Prefer `aucpr` (area under precision-recall) and PR curves over plain accuracy/AUC for heavy imbalance.
- Consider `max_delta_step` for very imbalanced logistic. Discuss threshold tuning + calibration over naive resampling.

---

## 5. Feature Engineering Deep Dive (HIGH PRIORITY — user emphasis)
Build a dedicated notebook + `src/xgblearn/features/` modules. Teach which techniques matter for trees vs which don't.

### 5.1 What does NOT help tree models (dispel linear-model habits)
- **Scaling/normalization/standardization is unnecessary** for tree models — splits are threshold-based and invariant to monotonic feature transforms. State this explicitly; it's a common waste of effort.
- Pure monotonic transforms of a single feature (log, Box-Cox) generally don't change tree splits — EXCEPT where they simplify interactions/additivity or aid extrapolation framing.

### 5.2 What DOES matter
- **Categorical handling — teach all three, recommend native:**
  - **Native categorical** (`enable_categorical=True`, `tree_method="hist"`, pandas `category` dtype). Uses optimal partitioning (Fisher) for splits; `max_cat_to_onehot` controls one-hot vs partition threshold; `max_cat_threshold` caps categories per split. **XGBoost 3.1.0 introduced a "re-coder"** (per the 3.1.0 release notes: *"This re-coder saves categories in the trained model and re-codes the data during inference, to keep the categorical encoding consistent … it also supports string-based categories."*). Recommend this as the default — faster, lower memory, good accuracy, simpler pipeline. Empirically faster than manual one-hot/ordinal encoding.
  - **One-hot encoding:** fine for low-cardinality; explodes dimensionality for high-cardinality (avoid for things like ZIP codes).
  - **Target/GLMM encoding** (via `category_encoders`): powerful for high-cardinality, but **DANGEROUS** — must be fit **out-of-fold / inside CV folds** to avoid target leakage. Demonstrate the leakage *and* the correct cross-fitted approach.
- **Missing values:** XGBoost natively learns a default split direction for NaNs (sparsity-aware). Don't blindly impute-then-mask — let XGBoost handle NaN, but teach when explicit imputation + a missingness indicator helps.
- **High-cardinality categoricals:** native categorical, CV-safe target encoding, or frequency/count encoding — discuss tradeoffs.
- **Feature interactions:** trees discover interactions automatically, but hand-crafted ratios/differences/domain features can speed learning and improve performance. Demonstrate building interaction/ratio features.
- **Monotonic constraints:** `monotone_constraints` (e.g., enforce prediction increasing in a feature) for domain knowledge / regulatory needs. Requires `tree_method` hist or approx.
- **Interaction constraints:** `interaction_constraints` (nested lists of feature groups, e.g. `[[0,2],[1,3,4]]`, names supported in Python) restrict which features may interact — reduces overfitting and encodes domain knowledge. Requires hist/approx (older docs note GPU did not support it — verify behavior on 3.3).
- **Feature selection:** compare gain-based importance vs SHAP-based selection; show empirically they can disagree, and choose by goal.

### 5.3 Leakage rules (teach prominently)
- Split BEFORE any fitting transform. Fit encoders/imputers on **train only**; apply to val/test.
- Use sklearn `Pipeline`/`ColumnTransformer` so transforms are learned per-fold inside CV.
- Target encoding must be out-of-fold. Never use target/future info in feature creation. (Classic cautionary example: an ID column that correlates with the target via collection order.)

---

## 6. Model Evaluation & Interpretation
- **Splitting:** train/validation/test (validation for early stopping + tuning; test untouched until the very end). Stratify for classification; group splits when entities repeat.
- **Cross-validation:** `StratifiedKFold` (classification), `KFold` (regression), `GroupKFold`/`TimeSeriesSplit` where appropriate. Show `xgb.cv` for finding `num_boost_round`. CRITICAL: do feature engineering **inside** CV folds.
- **Metrics:** classification — ROC-AUC, PR-AUC/`aucpr`, logloss, F1, confusion matrix, Brier score; regression — RMSE, MAE, R², residual plots.
- **Calibration:** reliability diagrams; `CalibratedClassifierCV` (Platt/isotonic); Brier score. Tree ensembles are often miscalibrated — teach this.
- **Learning curves:** plot train vs validation metric per boosting round (from `evals_result()`) to diagnose over/underfitting.
- **SHAP:** TreeSHAP via `pred_contribs=True`. **XGBoost 3.3.0 uses an in-tree QuadratureTreeSHAP implementation on both CPU and GPU** (per the 3.3.0 GPU Support docs: *"The GPU path uses the same Quadrature-TreeSHAP formulation described by Wettenstein et al. (2026) for exact TreeSHAP feature attributions"*); GPU-accelerated when `device="cuda"`. Produce summary, dependence, and force plots.
- **Importance pitfalls (teach explicitly):** built-in importance types differ (`weight`/`gain`/`cover`/`total_gain`); the default `gain` can mislead; high-cardinality bias exists; **SHAP importance** (audits what the model *uses*) vs **permutation feature importance (PFI)** (measures generalization contribution) can **diverge** — an overfit model trained on pure-noise features shows nonzero SHAP importance but ~zero PFI. Reproduce this divergence and explain when to use which (audit vs insight).

---

## 7. MLflow Integration
- **Local tracking:** default is local `./mlruns`. Provide a Makefile target `mlflow-ui` running `mlflow ui --port 8080`. For Model Registry support, run a server with a backend store, e.g. `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 8080`.
- **Autologging:** `mlflow.xgboost.autolog()` before training — captures params, per-iteration metrics, feature importance, and the model with signature. Works with both native and sklearn APIs; creates child runs for `GridSearchCV`/`RandomizedSearchCV`. **Version caveat:** per MLflow's `mlflow.xgboost` docs, *"Autologging is known to be compatible with the following package versions: 2.1.0 <= xgboost <= 3.2.0. Autologging may not succeed when used with package versions outside of this range."* Since this project uses **xgboost 3.3.0 (outside that band)**, demonstrate autolog but **also implement robust manual logging as the production path**, and document that autolog may need `disable_for_unsupported_versions` handling or a pinned xgboost if it misbehaves.
- **Manual logging (production path):** `mlflow.log_params`, `mlflow.log_metrics`, infer + log a model signature via `mlflow.models.infer_signature`, attach an `input_example`, log data version / git commit / seed, and use `mlflow.xgboost.log_model(..., model_format="json")` for cross-version portability.
- **Model Registry:** register via `registered_model_name` or `mlflow.register_model`; use **aliases** (e.g., `@champion`) instead of hardcoded versions; document the None→Staging→Production→Archived lifecycle.
- **Best practices to bake in:** consistent experiment naming (`project-task-vN`), run naming with timestamps, log seed + data hash for reproducibility; use autolog for exploration and manual logging for the final/registered model.

---

## 8. Hyperparameter Tuning with Optuna
- **Use Optuna** (TPE Bayesian sampler by default) — recommended over GridSearch for XGBoost. Mention alternatives (`RandomizedSearchCV`, Hyperopt, Ray Tune) briefly.
- **Pruning:** `optuna_integration.XGBoostPruningCallback(trial, observation_key)` — the class now lives in the **`optuna-integration`** package (older `optuna.integration` path is legacy). `observation_key` like `validation_0-logloss` for the sklearn API (include the `eval_set` index). Combine with `MedianPruner` or `SuccessiveHalvingPruner`.
- **Search space (recommended starting ranges):**
  - `learning_rate`: loguniform 0.01–0.3
  - `max_depth`: int 3–10
  - `min_child_weight`: int/loguniform 1–10
  - `gamma`: 0–5
  - `subsample`: 0.5–1.0
  - `colsample_bytree`: 0.5–1.0
  - `reg_alpha`: loguniform 1e-3–10
  - `reg_lambda`: loguniform 1e-3–10
  - Fix a high `n_estimators` + early stopping rather than tuning `n_estimators` directly.
- **MLflow integration:** wrap each Optuna trial in a **nested** MLflow run (parent = study); log trial params + objective value. Reduce Optuna stdout verbosity since MLflow tracks everything.
- Keep `device="cuda"` in params during tuning for GPU speed.

---

## 9. Recommended Learning Datasets / Progression
Use easy-to-load datasets (sklearn built-ins / OpenML), progressing in difficulty:
1. **Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`) — small binary classification; quick smoke test for the whole pipeline + GPU check.
2. **California Housing** (`sklearn.datasets.fetch_california_housing`) — regression; RMSE/MAE, residuals, monotonic-constraints demo.
3. **Adult / Census Income** (OpenML `adult`) — tabular classification with rich **categorical** features — the centerpiece for the feature-engineering module (native categorical vs one-hot vs target encoding).
4. **Credit Card Fraud** (OpenML `creditcard`) — highly imbalanced binary; `scale_pos_weight`, `aucpr`, PR curves, threshold tuning, calibration.
Document each in `src/xgblearn/data/loaders.py` with caching to `data/raw/`.

### Suggested build/work sequence
1. Scaffold repo + tooling (pyproject, ruff, pre-commit, Makefile, venv).
2. `scripts/check_gpu.py` — confirm Blackwell GPU works.
3. Data loaders + leakage-safe splits.
4. Baseline XGBoost (both APIs) on breast cancer + MLflow autolog.
5. Regression on California housing + learning curves + early stopping.
6. Feature-engineering module on Adult (categorical strategies, leakage demos, interactions, monotonic/interaction constraints).
7. Optuna tuning + pruning + MLflow nested runs.
8. Imbalanced fraud dataset + metrics/calibration.
9. SHAP + importance-pitfalls notebook.
10. Model registry + reload/inference demo + tests.

---

## 10. Common Mistakes / Gotchas to Document
- Using the removed `tree_method="gpu_hist"` (use `device="cuda"`, `tree_method="hist"`).
- Scaling features for trees (unnecessary).
- Data leakage: fitting encoders/imputers before the split; target encoding without out-of-fold.
- Tuning on the test set; not holding out a true test set.
- Trusting default `gain` importance blindly; conflating SHAP and permutation importance.
- Not using early stopping (over/underfitting `num_boost_round`).
- Ignoring class-imbalance metrics (accuracy on 99/1 data is meaningless).
- Device-mismatch prediction fallback warning (keep data + booster on the same device).
- Using pickle for long-term model storage instead of `save_model(..., .json/.ubj)` — the legacy binary format was removed in 3.1; JSON/UBJ are the stable formats.
- Forgetting determinism: set `random_state`/seed everywhere; log it.

---

## 11. Engineering Practices to Bake In
- `pyproject.toml` configuring **ruff** (lint + format, replacing black+flake8+isort), **mypy**, and **pytest**; line-length 88; build backend hatchling; package under `src/xgblearn`.
- `.pre-commit-config.yaml`: ruff, ruff-format, mypy, trailing-whitespace, end-of-file-fixer, check-added-large-files.
- Type hints throughout `src/`; docstrings that **teach** (this is a learning repo).
- `Makefile` targets: `setup`, `lint`, `format`, `typecheck`, `test`, `check-gpu`, `train`, `tune`, `evaluate`, `mlflow-ui`, `clean`.
- Config-driven runs via YAML in `configs/` loaded with pydantic; centralized seed management in `src/xgblearn/config.py`.
- Reproducibility: global seed, log git commit + data hash to MLflow, pin dependency versions.
- Tests (pytest): assert no leakage (encoders fit only on train), output shapes, deterministic results with a fixed seed, and a GPU smoke test (skippable when no GPU is present).

---

## 12. Summary Guidance for the Coding Agent
Build the repo top-to-bottom following §9's work sequence. Prioritize the **feature-engineering** content (§5) and **GPU correctness** (§1) since those are the user's explicit emphases. Every module and notebook should be self-documenting and pedagogical: short conceptual markdown/docstring intros, then heavily-commented runnable code, then a "gotchas / what to notice" closing note. Default all training to `device="cuda"`, `tree_method="hist"`; default categorical handling to native (`enable_categorical=True`); default tuning to Optuna with pruning + nested MLflow runs; and treat the held-out test set as sacred. When in doubt, prefer the modern XGBoost 3.x scikit-learn API for ergonomics and the native API where memory efficiency (`QuantileDMatrix`) or advanced features are being taught.