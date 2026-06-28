# The additive model & Newton boosting

This note answers two questions:

1. When we say the prediction is a **sum** of trees, what's being added — and how
   does that work for *classification*, not just regression?
2. What does "each tree fits the **gradient** (and **hessian**) of the loss" mean,
   concretely, in calculus terms?

## 1. The additive model — what gets summed

A trained XGBoost model is literally a sum of trees:

```
F(x) = base_score + η·f₁(x) + η·f₂(x) + ... + η·f_K(x)
```

- Each `f_k` is one tree. **A tree's output is a real number** — the *leaf weight*
  of whichever leaf `x` falls into. (Not a class, not a probability — a number.)
- `η` is the `learning_rate` (shrinkage), applied to every tree.
- `base_score` is the initial constant guess (round 0).

So `F(x)` is a single real number, the **raw score** (a.k.a. the *margin* or
*logit*). The "summing" is identical for every task. What differs is the **link
function** that maps that raw score into the target space at the very end:

| Task | Objective | Link applied to F(x) | Final output |
|---|---|---|---|
| Regression | `reg:squarederror` | identity | `ŷ = F(x)` |
| Binary classification | `binary:logistic` | sigmoid | `p = 1 / (1 + e^(−F(x)))` |
| Multiclass | `multi:softprob` | softmax over K scores | class probabilities |

### Why summing "works" for classification

This is the key insight that resolves the confusion. For classification we **do
not** sum probabilities (that would be nonsense — they wouldn't stay in [0,1]).
Instead:

- The trees sum to a raw score `F(x)` living in **log-odds (logit) space**, which
  ranges over all reals (−∞ to +∞). Adding more trees just pushes the score up or
  down — always valid.
- Only at the end do we squash it into a probability with the **sigmoid**:
  `p = 1/(1+e^(−F))`. `F = 0 → p = 0.5`; large positive `F → p → 1`; large
  negative `F → p → 0`.

This is exactly how **logistic regression** works: it sums a linear combination
`w·x` to get a logit, then sigmoids it. XGBoost just replaces the linear `w·x`
with a sum of trees. Same skeleton, more expressive body.

`base_score` for binary classification is the **log-odds of the prior**: if 63% of
training labels are positive, `base_score = log(0.63/0.37) ≈ 0.53`, so before any
tree fires the model already predicts the base rate.

## 2. Gradient boosting = gradient descent in *function space*

Ordinary gradient descent minimizes a loss `L(θ)` by nudging parameters:
`θ ← θ − η·∇_θ L`. We step opposite the gradient because that's the direction of
steepest decrease.

Gradient **boosting** does the same thing, but the "parameter" being optimized is
the *whole function* `F`. We can't take a derivative w.r.t. a function directly,
so we evaluate it at each training point: the quantity that tells us "which way to
nudge the prediction for row i to reduce its loss" is

```
g_i = ∂ L(y_i, F(x_i)) / ∂ F(x_i)        ← the gradient for row i
```

The direction of steepest descent for row `i` is `−g_i`. So:

> **Each boosting round fits a new tree to predict the negative gradient** (the
> "pseudo-residuals"), then adds it. The tree points the ensemble in the direction
> that most reduces the loss.

### "Fit the residuals" is just the squared-error special case

For regression with squared-error loss `L = ½(y − F)²`:

```
g = ∂L/∂F = (F − y)      →   −g = (y − F) = the ordinary residual
```

That's why boosting is often first taught as "each tree fits the residual error of
the last." True — but only for squared-error loss. The *general* rule is "fit the
negative gradient," which works for any differentiable loss.

### For binary logloss the gradient is beautifully clean

Loss: `L = −[y·log(p) + (1−y)·log(1−p)]` with `p = sigmoid(F)`. Differentiating
w.r.t. `F` (chain rule through the sigmoid) collapses to:

```
g_i = p_i − y_i              (predicted probability minus the label)
h_i = p_i · (1 − p_i)        (the second derivative — see below)
```

So the pseudo-residual for classification is simply *how far off the predicted
probability is*. Clean gradients like this are why logistic loss is the default.

## 3. The hessian — Newton's method, and XGBoost's signature formulas

Classic gradient boosting (e.g. sklearn's `GradientBoosting`) uses only the
gradient `g`. **XGBoost also uses the second derivative**, the **hessian**:

```
h_i = ∂² L(y_i, F(x_i)) / ∂ F(x_i)²       ← curvature of the loss for row i
```

This makes it **Newton's method** in function space rather than plain gradient
descent. Newton's step is `−g/h` (gradient divided by curvature) instead of
`−η·g`: take big steps where the loss is flat, small careful steps where it's
sharply curved. Better-scaled steps → faster, more stable convergence.

### Where the formulas come from (the one derivation worth knowing)

At each round, approximate the loss of adding tree `f` with a **2nd-order Taylor
expansion** around the current predictions, plus a regularization term
`Ω(f) = γT + ½λ·Σ wⱼ²` (T = number of leaves, wⱼ = leaf weights):

```
Obj(f) ≈ Σ_i [ g_i·f(x_i) + ½·h_i·f(x_i)² ]  +  γT + ½λ·Σ_j wⱼ²
```

For a *fixed tree shape*, every row in leaf `j` gets the same value `wⱼ`. Let
`Gⱼ = Σ g_i` and `Hⱼ = Σ h_i` over the rows in leaf `j`. The leaf's objective is a
simple quadratic in `wⱼ`:

```
Objⱼ = Gⱼ·wⱼ + ½(Hⱼ + λ)·wⱼ²
```

Minimize (set derivative to 0) → the **optimal leaf weight**:

```
wⱼ* = − Gⱼ / (Hⱼ + λ)
```

Substitute back → the **best achievable objective** for that leaf:

```
Objⱼ* = − ½ · Gⱼ² / (Hⱼ + λ)
```

That `Gⱼ²/(Hⱼ+λ)` is the **similarity / quality score** of a leaf. The **gain** of
splitting a node into left/right children is the improvement in this score:

```
Gain = ½ [ G_L²/(H_L+λ)  +  G_R²/(H_R+λ)  −  (G_L+G_R)²/(H_L+H_R+λ) ]  −  γ
```

The tree-builder tries candidate splits and keeps the one with the highest `Gain`;
a split is only kept if `Gain > 0`. **This single mechanism powers everything** —
the loss enters *only* through `g_i` and `h_i`, so swapping the objective (logloss
→ squared error → Poisson → ...) changes how `g` and `h` are computed and nothing
else. That's the elegance of the design.

### This is where the hyperparameters live

Reading the formulas, several knobs stop being magic:

| Knob | Role in the math |
|---|---|
| `reg_lambda` (λ) | the `+λ` in the denominator — shrinks leaf weights toward 0 (L2). |
| `gamma` / `min_split_loss` (γ) | the `−γ` in `Gain` — a split must clear this bar, or it's pruned. |
| `min_child_weight` | a floor on `Hⱼ` (total hessian in a child). For logloss, `h = p(1−p)`, so it demands a minimum amount of "confident" data per leaf — a conservativeness lever. |
| `learning_rate` (η) | scales each tree's contribution when added: `F ← F + η·f`. |
| `reg_alpha` (α) | L1 on leaf weights (not shown above; adds a soft-threshold to wⱼ*). |

## Putting it together — one boosting round

1. With the current ensemble, compute each row's prediction → `p_i` (after the link).
2. From the loss, compute `g_i` and `h_i` (for logloss: `g = p − y`, `h = p(1−p)`).
3. Grow a tree: score candidate splits with the `Gain` formula (sums of `g`, `h`).
4. Set each leaf's value to `wⱼ* = −Gⱼ/(Hⱼ+λ)`.
5. Add the tree, scaled by `η`: `F ← F + η·f`.
6. Repeat until validation stops improving (early stopping).

---

### Connects to the code
- `configs/model_baseline.yaml` — `learning_rate`, `reg_lambda`, `reg_alpha`,
  `gamma`, `min_child_weight` are exactly the symbols above.
- `src/xgblearn/config.py::ModelConfig.to_xgb_params` — `objective` selects which
  `g`/`h` formulas XGBoost uses.
- `notebooks/1.0-...ipynb` — the `val-logloss` learning curve is `L` decreasing as
  rounds add trees; early stopping halts when it stops dropping.

### Terms
- **logit / log-odds** — `log(p/(1−p))`; maps a probability in (0,1) to all reals.
- **sigmoid** — `1/(1+e^(−z))`; the inverse of the logit, maps reals back to (0,1).
- **link function** — the function mapping the raw summed score to the target space.
- **gradient (g)** — first derivative of the loss w.r.t. the prediction.
- **hessian (h)** — second derivative; the *curvature* of the loss.
- **pseudo-residual** — the negative gradient `−g`; what each tree is fit to.
- **Newton's method** — optimization using both slope and curvature; step `−g/h`.
- **Taylor expansion** — local polynomial approximation of a function; here 2nd-order.
