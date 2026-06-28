# Knowledge base

A running, plain-English (with math where it matters) reference for XGBoost and
the ML/data-science concepts around it. This is the "why" companion to the code
in `src/xgblearn/` and the notebooks — written to be read in order, but each note
stands on its own.

These notes are built up conversationally as the project grows. When a concept
comes up while building a stage, it gets distilled here so the understanding
compounds instead of evaporating.

## Index

1. [What is an XGBoost model?](01-what-is-xgboost.md) — decision trees, bagging
   (random forest) vs. boosting, and what "gradient-boosted trees" actually means.
2. [The additive model & Newton boosting](02-boosting-additive-model-and-newton.md)
   — what gets "summed" (incl. for classification), link functions, and the
   gradient/hessian (second-order) math at XGBoost's core.
3. [Loss functions & cross-entropy](03-loss-functions-and-cross-entropy.md) —
   what a loss is, why classification uses logloss, what cross-entropy measures,
   and the full derivation of `g = p − y` and `h = p(1−p)`.
4. [How trees split & histograms](04-how-trees-split-and-histograms.md) — the
   exact greedy split search, the histogram method (`hist` / `QuantileDMatrix`),
   missing-value handling, and how a tree grows and prunes.
5. [Bias, variance & the knobs](05-bias-variance-and-the-knobs.md) — the
   bias–variance tradeoff, what every hyperparameter does to it, a tuning
   cheat-sheet, and the diagnosis→action playbook.
6. [Regression & monotonic constraints](06-regression-and-monotonic-constraints.md)
   — what changes for regression (identity link, `g=F−y`, `h=1`), RMSE/MAE/R²,
   residual plots, and enforcing domain knowledge with `monotone_constraints`.

## Hands-on tools

- `scripts/inspect_tree.py` — trains a tiny shallow model and prints a real tree
  (split conditions, leaf weights, gain, cover), plus a feature-importance plot.
  The concrete companion to notes 02/04/05.

## Conventions

- Math is written in readable plain text (e.g. `w* = -Σg / (Σh + λ)`), not LaTeX,
  so it reads fine in a terminal or a plain Markdown viewer.
- Each note ends with a short **"connects to the code"** section pointing at the
  relevant module/config, and a **"terms"** glossary of anything introduced.
