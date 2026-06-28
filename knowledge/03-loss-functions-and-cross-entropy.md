# Loss functions & cross-entropy (where `g = p − y` comes from)

Note 02 used the gradient `g = p − y` and hessian `h = p(1−p)` for classification
without deriving them. This note steps back to the **loss function** — what it is,
why classification uses **logloss / cross-entropy**, what cross-entropy actually
measures, and then derives `g` and `h` from first principles.

## 1. What a loss function is, and why it's the real objective

A **loss function** `L(y, ŷ)` scores how wrong a single prediction `ŷ` is against
the truth `y`. Training = finding the model that **minimizes the average loss**.
The loss *defines* what "good" means; pick the wrong loss and you optimize the
wrong thing.

For boosting specifically, the loss has a second, structural job: recall from note
02 that every tree is fit to the **negative gradient of the loss**, and split
quality uses the **gradient + hessian**. So the loss isn't just a scorecard — *it
is the thing whose derivatives drive the entire algorithm.* That immediately rules
out some "obvious" choices:

> **Why not just optimize accuracy?** Accuracy is a step function of the raw score
> `F`: nudging `F` a little usually doesn't flip any label, so the gradient is
> **zero almost everywhere** (and undefined at the threshold). Zero gradient = no
> learning signal. We need a loss that is **smooth and differentiable in `F`**, so
> every prediction contributes a useful "nudge." Logloss is that loss.

## 2. Cross-entropy / logloss for binary classification

For binary labels `y ∈ {0,1}` and predicted positive-class probability `p`, the
**binary cross-entropy** (a.k.a. **log loss**) is:

```
L(y, p) = − [ y·log(p) + (1−y)·log(1−p) ]
```

Read it by cases:
- If `y = 1`: `L = −log(p)`. Predict `p→1` → loss → 0; predict `p→0` → loss → ∞.
- If `y = 0`: `L = −log(1−p)`. Symmetric.

So logloss **rewards confident-and-right** and **savagely punishes
confident-and-wrong** (the `−log` blows up near a confidently wrong prediction).
That asymmetry is exactly what you want: a model that says "99% malignant" and is
wrong should hurt far more than one that hedged at 55%.

### What "cross-entropy" is measuring (the information view)

The name comes from information theory:

- **Entropy** `H(p) = −Σ p·log p` — the average "surprise" (in bits/nats) of
  outcomes drawn from a distribution `p`. Predictable distributions have low
  entropy; coin-flip-uniform has the most.
- **Cross-entropy** `H(p, q) = −Σ p·log q` — the average surprise when the *true*
  distribution is `p` but you *encode/score* using your predicted distribution
  `q`. It's minimized, for fixed `p`, exactly when `q = p`.
- **KL divergence** `KL(p‖q) = H(p,q) − H(p) ≥ 0` — the *extra* surprise you pay
  for using the wrong distribution `q` instead of the truth `p`.

For one labeled example the true distribution is a spike: `y=1` means "[P(0),P(1)]
= [0,1]". Plugging that into cross-entropy collapses the sum to a single term and
gives exactly the logloss formula above. So **minimizing logloss = making your
predicted distribution `q` as close as possible (in KL) to the true label** — and,
in expectation over data, driving `p` toward the true conditional probability.

### Two more reasons logloss is the right default

- **It's a proper scoring rule.** Its expected value is minimized *only* when your
  predicted probability equals the true probability. So honest, **calibrated**
  probabilities are optimal — the loss can't be gamed by over/under-confidence.
  (Calibration is its own topic — Stage 4.)
- **It's maximum likelihood.** A 0/1 label is a Bernoulli draw with likelihood
  `p^y·(1−p)^(1−y)`. The negative log-likelihood over the dataset is
  `−Σ[y·log p + (1−y)·log(1−p)]` — *identical* to total logloss. Minimizing logloss
  is doing maximum-likelihood estimation.

## 3. Deriving `g = p − y` and `h = p(1−p)`

The trees output a raw score `F`; the probability is `p = σ(F) = 1/(1+e^(−F))`
(sigmoid). We need the derivatives of `L` **with respect to `F`** (that's what
boosting steps in), so we chain through `p`.

**Useful fact — the sigmoid derivative:**

```
σ(F) = 1/(1+e^(−F))      ⇒      dσ/dF = σ(F)·(1−σ(F)) = p(1−p)
```

**Gradient** (first derivative of `L` w.r.t. `F`), via the chain rule
`dL/dF = (dL/dp)·(dp/dF)`:

```
dL/dp = −[ y/p − (1−y)/(1−p) ]
dp/dF = p(1−p)

dL/dF = −[ y/p − (1−y)/(1−p) ] · p(1−p)
      = −[ y(1−p) − (1−y)p ]
      = −[ y − y·p − p + y·p ]
      = −[ y − p ]
      =  p − y                       ✓   so   g = p − y
```

The `p(1−p)` from the sigmoid derivative cancels the `p` and `(1−p)` denominators
from `dL/dp` — that algebraic cancellation is *why* logistic loss + sigmoid pairs
so cleanly, and why `g` is just "predicted minus actual."

**Hessian** (second derivative): differentiate `g = p − y = σ(F) − y` w.r.t. `F`:

```
h = dg/dF = dσ/dF = p(1−p)            ✓   so   h = p(1−p)
```

Note `h = p(1−p)` is largest at `p = 0.5` (most uncertain → most curvature, big
information) and shrinks toward 0 as `p → 0` or `1` (confident predictions carry
little additional curvature). This is exactly the "confidence weight" that
`min_child_weight` thresholds (note 02).

## 4. The same recipe for other objectives

Every objective is just a different `L`, yielding different `g`/`h` that plug into
the *same* tree machinery:

| Objective | Loss `L` | gradient `g` | hessian `h` |
|---|---|---|---|
| `reg:squarederror` | `½(y−F)²` | `F − y` (the residual) | `1` |
| `binary:logistic` | binary cross-entropy | `p − y` | `p(1−p)` |
| `multi:softprob` | categorical cross-entropy | `pₖ − 1{y=k}` per class | `pₖ(1−pₖ)` |
| `reg:absoluteerror` | `|y−F|` | `sign(F−y)` | (const; uses LAD leaf fix) |
| `count:poisson` | Poisson deviance | `e^F − y` | `e^F` |

This is the unifying payoff: choosing an objective only changes the two-line `g`/`h`
computation; split-finding, leaf weights, and gain (note 04) are untouched.

---

### Connects to the code
- `configs/model_baseline.yaml` → `objective: binary:logistic`,
  `eval_metric: [logloss, auc]`.
- `src/xgblearn/models/evaluate.py` → `log_loss` is this same `L`, reported as the
  `logloss` metric; `brier` is a related proper score (squared error on probs).
- The `val-logloss` MLflow curve is the mean of this `L` over the validation set
  per boosting round.

### Terms
- **cross-entropy / logloss** — `−Σ p·log q`; for a 0/1 label reduces to the binary
  formula above.
- **entropy** — average surprise of a distribution, `−Σ p·log p`.
- **KL divergence** — extra surprise from using the wrong distribution; `≥ 0`.
- **proper scoring rule** — a loss minimized (in expectation) only by the true
  probability; rewards calibration.
- **maximum likelihood (MLE)** — choosing parameters that make the observed data
  most probable; equivalent here to minimizing logloss.
- **sigmoid `σ`** — `1/(1+e^(−F))`, with the tidy derivative `σ(1−σ)`.
