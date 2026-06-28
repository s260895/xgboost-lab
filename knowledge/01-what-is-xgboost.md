# What is an XGBoost model?

> Short answer: it's a **gradient-boosted decision tree ensemble** — a *sum* of
> many shallow decision trees, built one at a time, where each new tree is fit to
> correct the errors of all the trees before it. It is **not** a random forest,
> even though both are "many trees."

## 1. A single decision tree

A decision tree is a learned flowchart of yes/no questions:

```
                 worst_radius < 16.8 ?
                /                      \
              yes                       no
        texture < 21.5 ?            [predict: malignant]
        /            \
   [benign]      [malignant]
```

Each **leaf** holds a prediction. Trees are intuitive, but a single deep tree
**memorizes** its training data: low bias, **high variance** — nudge the data and
you get a very different tree. One tree alone is weak and unstable. Ensembles
exist to fix that, and there are two opposite philosophies for how.

## 2. Bagging (Random Forest): a committee voting in parallel

A random forest grows **many full, deep trees independently and in parallel**. To
make them differ, each tree sees:

- a random **bootstrap sample** of the rows (sampled with replacement), and
- a random **subset of features** at each split.

Predictions are **averaged** (regression) or **majority-voted** (classification).

Intuition: each deep tree is individually high-variance (it overfits in its own
random way), but their errors are largely **uncorrelated**. Average hundreds of
noisy-but-roughly-unbiased predictors and the noise cancels. **Bagging reduces
variance.** Every tree is a complete, standalone predictor.

## 3. Boosting (XGBoost): a relay of specialists fixing mistakes

XGBoost builds trees **sequentially**, and each new tree's only job is to
**correct the errors the ensemble has made so far**. The trees are deliberately
**shallow** (depth 3–6) — individually "weak learners." The final prediction is
the **sum** of all trees' outputs, not an average.

The loop:

1. **Start** with a constant guess (e.g. the overall log-odds of the positive class).
2. **Measure what's still wrong** — the error of the current ensemble on each row.
3. **Fit a new shallow tree to those errors** (not to the labels directly).
4. **Add that tree**, scaled down by the `learning_rate`, so no tree overcorrects.
5. **Repeat** hundreds of times. Each tree chips at the remaining error.

Because each tree attacks the *residual* error, **boosting reduces bias** (and
regularization keeps variance in check). It's a relay race, not a parallel
committee.

This maps onto the knobs in our configs:

| Knob | Meaning |
|---|---|
| `num_boost_round` (= `n_estimators`) | how many trees in the relay |
| `learning_rate` (= `eta`) | how big a correction each tree may contribute (smaller = more careful, needs more trees) |
| `max_depth` | how complex each weak learner is |
| early stopping | stop adding trees once validation error stops improving |

## 4. Side by side

| | Random Forest (bagging) | XGBoost (boosting) |
|---|---|---|
| Trees built | Independently, in parallel | Sequentially, one after another |
| Each tree's job | Predict the whole answer | Fix the previous trees' errors |
| Tree depth | Deep (full-grown) | Shallow (weak learners) |
| Combined by | Averaging / voting | **Summing** |
| Mainly reduces | **Variance** | **Bias** |
| Tuning | Forgiving | Sensitive (LR × #trees × depth interact) |

## 5. So what is XGBoost, precisely?

A **gradient-boosted decision tree ensemble**: a sum of many shallow trees, built
one at a time, where each tree is fit to the **gradient of the loss** of the
ensemble so far. XGBoost specifically also uses the **second derivative (hessian)**
— Newton-style optimization (see note 02).

The **"X" (eXtreme)** is the engineering that makes this fast and accurate:
built-in L1/L2 **regularization**, **histogram binning** (`tree_method="hist"`),
**sparsity-aware** missing-value handling, and GPU support. The core idea is
decades-old gradient boosting; XGBoost is the industrial-strength implementation.

---

### Connects to the code
- `src/xgblearn/models/train.py` — both APIs train this same ensemble; the
  docstring lists the `learning_rate`/`eta` alias mapping.
- `configs/model_baseline.yaml` — the knobs in the table above.
- `notebooks/1.0-breast-cancer-baseline.ipynb` — the learning curve shows each
  successive tree shaving off error.

### Terms
- **bias** — error from the model being too simple to capture the true pattern.
- **variance** — error from the model being too sensitive to the particular
  training sample.
- **weak learner** — a model only slightly better than guessing (here, a shallow tree).
- **bootstrap sample** — a same-size sample drawn *with replacement*.
