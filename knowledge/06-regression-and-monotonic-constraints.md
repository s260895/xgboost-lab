# Regression with XGBoost & monotonic constraints

Stage 2 switches from classification to **regression** (predicting a continuous
number). This note covers what actually changes, the regression metrics, and
**monotonic constraints** — a way to bake domain knowledge into the model.

## 1. How little changes from classification

Boosting is loss-agnostic (`knowledge/02`–`03`): pick a loss, get `g`/`h`, the same
tree machinery runs. Regression just plugs in a different loss:

| | Binary classification | Regression |
|---|---|---|
| objective | `binary:logistic` | `reg:squarederror` |
| loss `L` | cross-entropy | `½(y − F)²` |
| gradient `g` | `p − y` | `F − y` (the residual) |
| hessian `h` | `p(1−p)` | `1` |
| **link function** | sigmoid → probability | **identity** (none) |
| metrics | logloss, AUC, ... | RMSE, MAE, R² |

The two big conceptual simplifications:
- **No link function.** The summed tree score `F(x)` *is* the prediction. There's
  no sigmoid squashing — the model directly outputs the target value.
- **Constant hessian `h = 1`.** Every row has equal curvature, so `cover` (Σh) in a
  node is just the *row count*, and the leaf weight `w* = −G/(H+λ)` is essentially
  the (regularized) mean residual of the leaf.

Everything else — histogram splits, the gain formula, early stopping, both APIs,
no feature scaling — is identical.

## 2. Regression metrics (and what each tells you)

- **RMSE** (root mean squared error): in target units; **squares** errors so large
  misses dominate. The default `reg:squarederror` optimizes this. Sensitive to
  outliers.
- **MAE** (mean absolute error): in target units; treats all errors linearly, so
  it's **robust to outliers**. Optimized by `reg:absoluteerror`.
- **R²** (coefficient of determination): unit-free, `1 − SS_res/SS_tot`. `1.0` is
  perfect; `0.0` means "no better than predicting the mean"; **negative** means
  worse than the mean. Good for "how much variance did I explain?"

Pick the loss to match the metric you care about: if large errors are
disproportionately bad, optimize squared error; if you want robustness to a few
wild targets, consider absolute or Huber (`reg:pseudohubererror`).

### The residual plot

Plot residuals (`y − ŷ`) against the prediction. Healthy = a structureless cloud
around zero. Diagnostic patterns:
- **Funnel** (spread grows with prediction) → heteroscedasticity; consider modeling
  `log(y)`.
- **Curvature** → a systematic nonlinearity the model missed.
- **A hard ceiling/floor** → a *capped target*. California Housing is capped at
  ~5.0 ($500k), so the model can't exceed it and residuals pile up there. (Worth
  knowing — it caps achievable accuracy.)

## 3. Monotonic constraints

Sometimes you *know* the direction of a relationship and want to **guarantee** it:
predicted house value should **not decrease** as median income rises, all else
equal. An unconstrained model, chasing noise, will produce small non-monotonic
**dips** — awkward for trust, and disqualifying in regulated domains (credit,
insurance, pricing) where you must defend that "more income never lowers the
score."

`monotone_constraints` enforces this:
- `{"MedInc": 1}` → prediction is **non-decreasing** in `MedInc`.
- `{"MedInc": -1}` → **non-increasing**.
- `0` (default) → unconstrained.

(You can pass a dict keyed by feature name, or a tuple aligned to columns.)

### How it works (and the cost)

During split finding (`knowledge/04`), XGBoost **rejects any split whose child leaf
values would violate the ordering**, and propagates bounds down the tree so deeper
splits can't undo a parent's monotonic guarantee. Because it filters splits, it
requires `tree_method="hist"` (or `approx`).

The tradeoff: you remove some flexibility, so a constrained model can be slightly
less accurate on the training distribution — but it's **more trustworthy and often
generalizes better** when the constraint reflects a real-world truth (the
constraint is a form of regularization / prior knowledge). In the Stage 2 notebook,
the unconstrained partial-dependence curve on `MedInc` dips in a few places; the
constrained one is monotonically non-decreasing with negligible accuracy change.

> **Related (Stage 3): interaction constraints** (`interaction_constraints`) limit
> *which features may appear together in a tree path* — a different way to inject
> structure. Covered when we hit the Adult dataset.

## 4. Mental checklist for a regression task

1. Objective: `reg:squarederror` (or absolute/Huber if outliers matter).
2. `eval_metric`: `rmse` (and/or `mae`); the **last** one drives early stopping.
3. Don't stratify the split (continuous target).
4. Read the **residual plot**, not just the scalar metrics.
5. Add `monotone_constraints` where you have a defensible directional prior.

---

### Connects to the code
- `configs/model_california.yaml` → `objective: reg:squarederror`,
  `eval_metric: [rmse, mae]`.
- `src/xgblearn/models/evaluate.py` → `regression_metrics`, `plot_residuals`,
  `plot_learning_curve`.
- `src/xgblearn/models/train.py` → `predict_regression`; `monotone_constraints`
  flows through `ModelConfig.params`.
- `notebooks/2.0-california-housing-regression.ipynb` → the full walkthrough +
  monotone partial-dependence demo.

### Terms
- **RMSE / MAE / R²** — the regression metric panel (above).
- **residual** — `y − ŷ`, the signed error of a prediction.
- **heteroscedasticity** — error variance that changes across the prediction range.
- **monotonic constraint** — a forced non-decreasing/non-increasing dependence on a
  feature.
- **partial dependence** — predicted output as one feature varies, others held fixed.
