# Bias, variance, and what every knob actually does

This is the practical payoff of notes 01–04: a mental model for *why* a model
under- or over-fits, and which hyperparameter to reach for in each case. If you
internalize one thing about tuning XGBoost, make it this.

## 1. The bias–variance decomposition

For a model's expected error on unseen data, you can (for squared error, exactly;
in spirit, generally) split it into three parts:

```
expected error  =  bias²  +  variance  +  irreducible noise
```

- **Bias** — error from the model being **too simple** to capture the real pattern.
  High bias = **underfitting**: it's wrong in a consistent, structural way.
- **Variance** — error from the model being **too sensitive** to the particular
  training sample. High variance = **overfitting**: retrain on slightly different
  data and you'd get a very different model.
- **Irreducible noise** — randomness in the data itself; no model can beat it.

You can't drive both bias and variance to zero — pushing one down usually pushes
the other up. The art is finding the sweet spot for *your* data.

### How it looks on the learning curve (your MLflow `train-`/`val-` charts)

| What you see | Diagnosis | Direction |
|---|---|---|
| train **and** val both high (loss won't drop) | **high bias** / underfit | make the model *more* powerful |
| train low, val much higher (big gap) | **high variance** / overfit | *regularize* / simplify |
| train low, val low, small gap | good fit | stop here |

This is exactly why we log train vs. val curves: the **gap** between them is a
direct read on variance, and the **floor** they sit at is a read on bias.

## 2. How boosting moves along the tradeoff

Boosting is unusual: adding trees **reduces bias** (each tree fixes more residual
error), and left unchecked it eventually **increases variance** (later trees start
fitting noise). So a boosted model walks *from* underfit *toward* overfit as rounds
increase — and **early stopping is the tool that halts at the sweet spot** (the
minimum of the validation curve). Everything else is about controlling *how fast*
and *how far* it can travel.

## 3. Each knob, and which way it pushes

Group the levers by what they control.

### Capacity / how hard each tree works
- **`max_depth`** — the **main overfitting lever**. Deeper trees model
  higher-order feature interactions → lower bias, higher variance. Typical 3–10;
  start ~6. Too deep overfits fast.
- **`num_boost_round`** (`n_estimators`) — more trees → lower bias, eventually
  higher variance. Don't tune this directly; set it high and let **early stopping**
  pick the count.
- **`learning_rate`** (`eta`) — shrinks each tree's contribution. **Lower LR
  generalizes better** (smoother, less greedy) but needs *more* trees to reach the
  same fit. The classic move: small LR + many trees + early stopping. LR and
  `num_boost_round` trade off directly.

### Conservativeness of splits (raise these to fight overfitting)
- **`min_child_weight`** — minimum total hessian `H` in a child (note 02/04).
  Higher = refuses to carve out small/uncertain groups → **higher bias, lower
  variance**.
- **`gamma`** (`min_split_loss`) — minimum `Gain` to keep a split. Higher = prunes
  more aggressively → simpler trees, **lower variance**.
- **`reg_lambda`** (L2) and **`reg_alpha`** (L1) — penalize leaf weights (the `+λ`
  in `w* = −G/(H+λ)`, and L1's soft-thresholding). Higher = smaller, smoother leaf
  values → **lower variance**. L1 can zero out leaves entirely (sparsity).

### Randomness / decorrelation (regularize by showing each tree less)
- **`subsample`** — fraction of **rows** per tree (0.5–1.0). <1 injects randomness
  that decorrelates trees → **lower variance** (a boosting cousin of bagging). Too
  low starves each tree → higher bias.
- **`colsample_bytree` / `bylevel` / `bynode`** — fraction of **columns** sampled
  per tree / per level / per split. Same effect: less correlated trees, lower
  variance; also speeds training.

### Resolution
- **`max_bin`** — histogram bins (note 04). More bins = finer thresholds →
  marginally lower bias, slightly higher variance and cost. 256 is a fine default.

## 4. The cheat-sheet

| Knob | ↑ increase it → | Fixes... |
|---|---|---|
| `max_depth` | more capacity (↓bias, ↑variance) | underfitting |
| `num_boost_round` | more capacity (↓bias, ↑variance) | underfitting (via early stop) |
| `learning_rate` | faster/greedier fit (↑variance) | *lower* it for overfitting |
| `min_child_weight` | more conservative (↑bias, ↓variance) | overfitting |
| `gamma` | more pruning (↑bias, ↓variance) | overfitting |
| `reg_lambda` / `reg_alpha` | more shrinkage (↓variance) | overfitting |
| `subsample` | *lower* it → more randomness (↓variance) | overfitting |
| `colsample_by*` | *lower* it → more randomness (↓variance) | overfitting |
| `max_bin` | finer splits (↓bias, ↑variance) | underfitting (marginal) |

> Mnemonic: **underfitting** → grow capacity (`max_depth` ↑, more rounds, `max_bin`
> ↑). **Overfitting** → either *regularize* (`gamma`/`lambda`/`alpha`/
> `min_child_weight` ↑) or *add randomness* (`subsample`/`colsample` ↓), and lower
> `learning_rate`.

## 5. A diagnosis → action playbook

1. **Read the curves.** Big train–val gap → variance problem. High train loss →
   bias problem.
2. **Underfitting?** Increase `max_depth`, allow more rounds, lower
   `min_child_weight`/`gamma`. Make sure early stopping isn't cutting too soon.
3. **Overfitting?** Lower `learning_rate` (+ more rounds), reduce `max_depth`,
   raise `min_child_weight`/`gamma`/`reg_lambda`, drop `subsample`/`colsample`.
4. **Slow but fine?** Lower `max_bin` or `colsample` to speed up.

## 6. The recommended tuning *order* (why Stage 5 tunes in stages)

Don't throw all knobs into one giant search — they interact, and order tames that:

1. **Fix a moderate `learning_rate`** (~0.1) and use **early stopping** to find the
   right number of trees.
2. **Tree structure:** `max_depth`, `min_child_weight`, `gamma`.
3. **Sampling:** `subsample`, `colsample_bytree`.
4. **Regularization:** `reg_alpha`, `reg_lambda`.
5. **Finally, lower the `learning_rate`** and refit with more trees for the final
   model.

This is exactly the order encoded in `configs/tuning.yaml` and used in Stage 5
(Optuna).

---

### Connects to the code
- `configs/model_baseline.yaml` / `configs/model_tuned.yaml` — every knob above.
- `configs/tuning.yaml` — the search ranges and the staged order from §6.
- MLflow `train-*` vs `val-*` curves — your live bias/variance readout.

### Terms
- **bias** — structural error from an over-simple model (underfit).
- **variance** — instability from over-sensitivity to the training sample (overfit).
- **regularization** — any constraint that trades a little bias for less variance.
- **early stopping** — halting boosting at the validation-loss minimum.
