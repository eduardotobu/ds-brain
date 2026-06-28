---
id: 20260627-1001
type: permanent
subtype: concept
created: 2026-06-27
modified: 2026-06-27
aliases:
  - L1
  - L2
  - Ridge
  - Lasso
  - Elastic Net
  - weight decay
tags:
  - ml
  - optimization
  - generalization
  - regularization
---
# Regularization reduces model complexity by adding a penalty to the loss function, trading variance for bias to improve generalization

Adding a penalty term to the training loss discourages the model from fitting
noise. The result generalizes better to unseen data, at the cost of slightly
higher training error. This is the primary lever for controlling the
bias-variance tradeoff in practice.

## Core insight
A model minimizing only training loss has no incentive to stay simple — it
will overfit whenever it can. Regularization adds that incentive explicitly
via a scalar λ. At λ=0 you get standard empirical risk minimization; at high
λ the model collapses toward the prior (zero weights).

## The two main forms

### L2 — Ridge / weight decay
Penalizes the **sum of squared weights**:

$$
\mathcal{L}_{reg} = \mathcal{L}_{train} + \lambda \sum_j w_j^2
$$

Shrinks all weights toward zero, but rarely to exactly zero. Stable under
correlated features — distributes weight among them rather than arbitrarily
picking one.

### L1 — Lasso
Penalizes the **sum of absolute weights**:

$$
\mathcal{L}_{reg} = \mathcal{L}_{train} + \lambda \sum_j |w_j|
$$

Drives many weights to *exactly* zero — performs implicit feature selection.
Preferred when you suspect only a sparse subset of features matter.

### Elastic Net
Combines both:

$$
\mathcal{L}_{reg} = \mathcal{L}_{train} + \lambda_1 \sum_j |w_j| + \lambda_2 \sum_j w_j^2
$$

Best of both worlds: L1 sparsity + L2 stability. Default choice when you have
many correlated features.

## Bayesian interpretation
Regularization is MAP estimation with a prior on weights:
- L2 ↔ Gaussian prior (zero-mean)
- L1 ↔ Laplace prior (zero-mean, heavier tails → sparsity)

λ encodes "how strongly you believe weights should be small."

## Why this matters in practice
- **Linear models**: Ridge/Lasso are the first thing to reach for before
  moving to tree ensembles.
- **Neural nets**: L2 is called *weight decay* and is almost always enabled
  (e.g., `weight_decay=1e-4` in PyTorch Adam). Dropout is a stochastic
  regularizer operating on activations, not weights.
- **λ is always tuned by cross-validation.** Use `RidgeCV` / `LassoCV` in
  sklearn — they handle this automatically over a logarithmic grid.

## Common mistakes
- Not scaling features first — L1/L2 are **not scale-invariant**. A feature
  with large magnitude dominates the penalty unfairly.
- Penalizing the bias term — by convention, exclude it from regularization.
- Treating regularization as a fix for data leakage or label noise — it only
  addresses variance, not bias from bad data.

## Caveats & limits
- Tree-based methods (RF, XGBoost) have their own regularization
  (`max_depth`, `min_samples`, subsampling) — L1/L2 don't apply directly.
- Very high λ causes underfitting — always validate on a held-out set.

## Links
- Builds on:      [[Bias-Variance Tradeoff defines the fundamental tension in generalization]]
- Addresses:      [[Overfitting occurs when a model learns noise instead of signal]]
- Used in:        [[Linear Regression models a continuous target as a weighted sum of features]]
- Used in:        [[Logistic Regression estimates class probability via sigmoid of a linear combo]]
- Related to:     [[Feature Selection reduces dimensionality by removing irrelevant features]]
- Contrasts with: [[Early Stopping is an implicit regularizer for iterative optimizers]]
- See MOC:        [[MOC - Example]]

## Sources
- [[ESL — Elements of Statistical Learning — Hastie et al.]] Ch. 3
- [[ISLR — Introduction to Statistical Learning — James et al.]] Ch. 6
- [[Pattern Recognition and Machine Learning — Bishop]] Ch. 3
