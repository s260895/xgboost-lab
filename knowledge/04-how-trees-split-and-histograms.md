# How a tree picks its splits (and why histograms make it fast)

Note 02 gave the **gain** formula but waved at "the tree-builder tries candidate
splits and keeps the best." This note opens that box: the exact greedy algorithm,
the histogram approximation (`tree_method="hist"`, what `QuantileDMatrix`
pre-computes), the missing-value handling, and how a tree grows and prunes.

## 1. The job at a single node

Recall the building blocks (note 02). For the rows `I` at a node, with
`g_i = ∂L/∂F` and `h_i = ∂²L/∂F²` per row, define the **leaf quality score**

```
score(I) = G² / (H + λ),     where  G = Σ_{i∈I} g_i,   H = Σ_{i∈I} h_i
```

A split sends `I` into a left set `I_L` and right set `I_R`. Its **gain** is the
improvement in total score, minus the per-split penalty `γ`:

```
Gain = ½ [ G_L²/(H_L+λ)  +  G_R²/(H_R+λ)  −  G²/(H+λ) ]  −  γ
```

The node's task: **over every feature and every threshold, find the split with the
maximum Gain.** Keep it only if `Gain > 0`.

## 2. The exact greedy algorithm

For one feature:

1. **Sort** the rows by that feature's value.
2. Sweep the threshold left→right. Maintain running `G_L, H_L` (accumulate as rows
   move to the left side); the right side is just `G_R = G − G_L`,
   `H_R = H − H_L`. So each candidate threshold is **O(1)** given the running sums.
3. Evaluate `Gain` at each split point (between consecutive distinct values); track
   the best.

Do this for every feature; take the global best `(feature, threshold)`.

**Cost.** The sweep is cheap, but the *sort* dominates: roughly
`O(#features · n·log n)` work, repeated at every node. On large data, re-sorting
each feature at each node is the bottleneck. That's what the histogram method kills.

## 3. The histogram method (`tree_method="hist"`)

Idea: **bin each feature once, then work on bins instead of raw values.**

1. **Up front** (once, before training), bucket each feature's values into at most
   `max_bin` bins (default 256). This is what **`QuantileDMatrix` precomputes** —
   and why a validation matrix passes `ref=dtrain`, to reuse the *same* bin edges.
2. **At a node**, build a **histogram** per feature: for each bin, accumulate the
   sum of `g` and sum of `h` of the rows falling in it. Building costs
   `O(n · #features)` — a single linear pass, **no sorting**.
3. **Find the split** by sweeping the ≤ `max_bin` bins (accumulating `G_L, H_L`
   across bins) and scoring `Gain`. That's `O(max_bin · #features)` — independent
   of `n`.

Two multipliers make it even faster:

- **Histogram subtraction trick.** A node's histogram equals the sum of its two
  children's histograms. So build the histogram for the **smaller** child directly,
  then get the larger child for free: `hist_big = hist_parent − hist_small`. Halves
  the histogram-building work at every level.
- **Cache- and GPU-friendly.** Fixed-width integer bins and dense histogram arrays
  vectorize well and map cleanly onto GPU threads — this is why `device="cuda"` +
  `hist` is fast on Blackwell.

### Is binning "lossy"? A little, and it's chosen well.

Binning means the split threshold can only land on a bin edge, not any raw value —
a small approximation. But the edges aren't naive equal-width cuts; XGBoost uses a
**weighted quantile sketch**: edges are placed so each bin holds roughly equal
**total hessian** (≈ equal "amount of confidence-weighted data"). Quantile edges
put resolution where the data actually is. With `max_bin = 256` the accuracy cost
is typically negligible, and you trade it for a large speed/memory win.

> **`max_bin` tradeoff:** more bins → finer thresholds, marginally better accuracy,
> more memory and time; fewer bins → faster, more regularized, coarser. 256 is a
> good default.

## 4. Missing values — sparsity-aware split finding

XGBoost does **not** require you to impute. At each split it learns a **default
direction** for missing values:

- Enumerate splits using only the rows that *have* the feature.
- Try assigning **all missing rows to the left**, compute Gain; try **all to the
  right**, compute Gain; keep whichever direction scores higher.
- Store that default direction in the node. At inference, a missing value simply
  follows it.

This is both principled (the direction is chosen to minimize loss) and efficient
(missing rows are handled as one block, not one-by-one). It also makes XGBoost fast
on **sparse** data (one-hot columns, lots of zeros): zeros are treated as "absent"
and skipped in the histogram, so cost scales with *non-missing* entries.

## 5. How the whole tree grows (and stops)

- **Growth policy.** XGBoost's default is **depth-wise** (`grow_policy="depthwise"`):
  split every node at the current depth before going deeper, up to `max_depth`.
  (LightGBM defaults to **leaf-wise** / best-first, splitting the single
  highest-gain leaf anywhere — often faster but more prone to deep, overfit
  branches. XGBoost offers this as `grow_policy="lossguide"`.)
- **Pruning with `γ` (`gamma` / `min_split_loss`).** A split survives only if
  `Gain > 0`, i.e. its raw quality improvement exceeds `γ`. XGBoost grows to
  `max_depth` and then prunes branches whose gain doesn't clear the bar
  (post-pruning), rather than stopping greedily at the first bad split — a split
  that looks weak can enable a strong one below it.
- **`min_child_weight`.** A candidate split is rejected if either child's total
  hessian `H` falls below this floor. For logloss `h = p(1−p)`, so it demands a
  minimum amount of confident data per leaf — preventing splits that isolate a few
  uncertain points.
- **Leaf values.** Once the structure is fixed, each leaf gets
  `w* = −G/(H+λ)` (note 02), and the tree is added as `F ← F + η·w*`.

## 6. Categorical splits (preview of Stage 3)

With `enable_categorical=True`, a categorical feature isn't split by a `<`
threshold but by **partitioning categories** into a left/right group. XGBoost uses
the same `G`/`H` gain machinery: it sorts categories by `G/H` and finds the best
partition (optimal for the score), falling back to one-hot when the cardinality is
below `max_cat_to_onehot`. Same engine, different split shape — covered in depth in
Stage 3.

---

### Connects to the code
- `configs/model_baseline.yaml` → `tree_method: hist`; `max_bin` (if set),
  `gamma`, `min_child_weight`, `max_depth` are the knobs above.
- `src/xgblearn/models/train.py` → `QuantileDMatrix(...)` does the pre-binning;
  the val matrix uses `ref=dtrain` to share bin edges; `enable_categorical`
  toggles partition splits.
- `scripts/check_gpu.py` → trains with `device="cuda"` + `hist`, the GPU-friendly
  histogram path.

### Terms
- **exact greedy** — evaluate every sorted threshold for every feature; precise but
  sort-bound.
- **histogram method** — bin features once, build per-bin `g`/`h` sums, sweep bins.
- **`max_bin`** — number of histogram buckets per feature (default 256).
- **weighted quantile sketch** — algorithm placing bin edges at hessian-weighted
  quantiles, so bins carry equal "confidence mass."
- **histogram subtraction** — derive a child histogram as parent minus sibling.
- **default direction** — the learned side that missing values follow at a split.
- **depth-wise vs leaf-wise** — grow all nodes per level vs. always split the
  globally best leaf.
