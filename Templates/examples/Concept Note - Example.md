---
id: 202606031500
title: "Bias-Variance Tradeoff"
created: 2026-06-03
updated: 2026-06-07
tags:
  - concept
  - ml
type: concept
status: evergreen
area: fundamentals
---
# Bias-Variance Tradeoff

## Definition

The total expected error of a model decomposes into three components: bias² (underfitting), variance (overfitting), and irreducible noise. Reducing one typically increases the other.

## Intuition

Imagine throwing darts at a target. **Bias** = how far your average throw is from the bullseye. **Variance** = how spread out your throws are. A simple model (e.g., linear) has high bias (always misses the same way) but low variance (consistent). A complex model (e.g., deep tree) has low bias (can hit anywhere) but high variance (inconsistent between games).

## Formal description

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$
- $\sigma^2$ = irreducible noise in the data

## When it applies

- Model selection: choosing between simple vs. complex models
- Hyperparameter tuning: regularization strength controls the tradeoff directly
- Ensemble methods: bagging reduces variance, boosting reduces bias
- Learning curves: diagnosing whether more data or more capacity will help

## When it fails

- Modern deep learning seems to violate it (double descent phenomenon)
- Very large models can have low bias AND low variance if sufficiently over-parameterized
- The decomposition assumes fixed model class; in practice we search over architectures

## Common pitfalls

- Confusing training error with bias (training error = 0 doesn't mean bias = 0)
- Thinking "more complex = always better" without considering sample size
- Ignoring that regularization is a bias-variance knob, not just "a thing you add"

## Code snippet

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Demonstrate bias-variance with tree depth
X, y = make_regression(n_samples=200, noise=10)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mse')
    print(f"depth={str(depth):>4} | MSE={-scores.mean():.1f} ± {scores.std():.1f}")
    # Low depth → high bias (bad mean) low variance (low std)
    # High depth → low bias (good mean) high variance (high std)
```

## Connections

- Prerequisites: [[Mean Squared Error]], [[Expected Value]]
- Used by: [[Regularization]], [[Ensemble Methods]], [[Cross-Validation]]
- Related: [[Double Descent]], [[Overfitting]]
- Contrast with: [[Bias-Variance in Deep Learning]]

## References

- Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*, Ch. 2.9
- Geman et al. (1992) — "Neural Networks and the Bias/Variance Dilemma"
