---
title: Gradient Descent
type: concept
domain: optimization
subfield: first-order methods
difficulty: beginner
status: evergreen
created: 2026-06-07
updated: 2026-06-07
tags:
  - ml/concept
  - optimization
  - training
aliases:
  - GD
  - steepest descent
  - batch gradient descent
---

# Gradient Descent

> [!abstract] TL;DR
> Iteratively move parameters in the direction of steepest decrease of a differentiable loss function. It's the workhorse optimizer behind almost every model trained today.

## Intuition
- Imagine standing on a foggy hill and wanting to reach the valley. You can't see far, but you can feel which way is downhill under your feet. Take a step that way, reassess, repeat.
- The "feel under your feet" is the **gradient** $\nabla_\theta \mathcal{L}(\theta)$: the direction of steepest *ascent* of the loss. We move in the **opposite** direction.
- It does **not** apply (or works poorly) when:
  - the loss is non-differentiable everywhere it matters,
  - the loss has dominating saddle points or pathological curvature,
  - the step size (learning rate) is set so badly that you overshoot or crawl forever.

## Formal Definition
Given a differentiable loss $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ and a learning rate $\eta > 0$:

$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta \, \nabla_\theta \mathcal{L}(\theta_t)
$$

**Symbols**
- $\theta_t \in \mathbb{R}^d$ — parameters at iteration $t$
- $\mathcal{L}(\theta)$ — scalar loss to minimize
- $\nabla_\theta \mathcal{L}(\theta_t)$ — gradient vector of $\mathcal{L}$ at $\theta_t$
- $\eta$ — learning rate (step size)

## Derivation / Key Steps
1. **First-order Taylor expansion** of the loss around the current point:
$$
\mathcal{L}(\theta + \Delta\theta) \approx \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^\top \Delta\theta.
$$
2. To decrease the loss most rapidly under a small-step constraint $\|\Delta\theta\| = \eta$, choose $\Delta\theta$ anti-parallel to the gradient:
$$
\Delta\theta = -\eta \, \frac{\nabla_\theta \mathcal{L}(\theta)}{\|\nabla_\theta \mathcal{L}(\theta)\|}.
$$
3. Dropping the normalization (folding it into $\eta$) gives the standard update rule.
4. **Convergence (convex, $L$-smooth case):** with $\eta \le 1/L$,
$$
\mathcal{L}(\theta_t) - \mathcal{L}(\theta^\star) \;\le\; \frac{\|\theta_0 - \theta^\star\|^2}{2\eta t} \;=\; \mathcal{O}(1/t).
$$
   If $\mathcal{L}$ is also $\mu$-strongly convex, convergence becomes **linear**: $\mathcal{O}((1 - \mu/L)^t)$.

> [!note] Assumptions
> - $\mathcal{L}$ is differentiable (and ideally $L$-smooth: $\nabla \mathcal{L}$ is $L$-Lipschitz).
> - Learning rate is chosen appropriately for the problem's curvature.
> - For the global-optimum guarantees above, $\mathcal{L}$ is convex. In non-convex settings (deep nets), GD only finds stationary points.

## Properties
- [x] Works on convex **and** non-convex losses (with weaker guarantees in the latter).
- [x] Differentiable loss required.
- [ ] Not scale-invariant — feature scaling materially affects convergence.
- [x] Deterministic given fixed initialization and data (vanilla / batch version).
- [x] Memory-cheap: only stores $\theta_t$ and the current gradient.
- **Per-iteration cost:** $\mathcal{O}(n \cdot d)$ for a dataset of $n$ samples and $d$ parameters (because the gradient sums over all examples).

## Worked Example
Minimize $\mathcal{L}(\theta) = (\theta - 3)^2$ starting from $\theta_0 = 0$ with $\eta = 0.1$.

Gradient: $\nabla \mathcal{L}(\theta) = 2(\theta - 3)$.

```python
import numpy as np

def loss(theta):       return (theta - 3) ** 2
def grad(theta):       return 2 * (theta - 3)

theta, eta = 0.0, 0.1
for t in range(25):
    theta = theta - eta * grad(theta)
    print(f"t={t:02d}  theta={theta:.6f}  loss={loss(theta):.6f}")
```

**Result / interpretation:** $\theta$ converges geometrically to the true minimum $\theta^\star = 3$. Each step closes ~20% of the remaining gap because $|1 - \eta \cdot 2| = 0.8$, matching the linear-convergence bound for a strongly convex quadratic.

## Connections
- **Generalizes:** the "move opposite the derivative" rule from single-variable calculus.
- **Special case of:** [[Mirror Descent]] (with Euclidean geometry), [[Proximal Gradient Methods]] (with zero regularizer).
- **Used by:** [[Example Models - Linear Regression]], [[Logistic Regression]], [[Neural Networks]], [[Backpropagation]] (which *computes* the gradient that GD then *uses*).
- **Contrasts with:**
  - [[Newton's Method]] — uses second-order curvature (the Hessian), faster near the optimum but expensive.
  - [[Coordinate Descent]] — updates one parameter at a time.
  - [[Stochastic Gradient Descent]] — approximates the gradient on a mini-batch for scalability.
- **Prerequisites:** [[Differentiability]], [[Convex Functions]], [[Lipschitz Smoothness]].

## Common Pitfalls
- **Learning rate too large** → loss oscillates or diverges. Symptom: loss curve zig-zags upward.
- **Learning rate too small** → painfully slow convergence; can look like the model isn't learning.
- **Unscaled features** → the loss surface becomes an elongated ellipse, and vanilla GD bounces across the narrow axis. Always standardize inputs (or use adaptive methods).
- **Confusing "vanilla / batch GD" with SGD** — batch GD uses the full dataset per step; SGD uses one sample (or a mini-batch). They have very different noise and convergence properties.
- **Assuming convergence to a global minimum** in non-convex problems. GD only guarantees a stationary point.

## Open Questions
- For deep non-convex losses, *why* does plain SGD generalize so well despite landing in non-global minima? (Active research: implicit regularization, flat minima, edge of stability.)
- When is line search worth it over a tuned constant $\eta$ in modern large-scale training?

## References
- 📄 Cauchy, A.-L. (1847). *Méthode générale pour la résolution des systèmes d'équations simultanées.* — the original.
- 📚 Boyd & Vandenberghe, *Convex Optimization*, Ch. 9 — clean treatment of descent methods and step-size rules.
- 📚 Nocedal & Wright, *Numerical Optimization*, Ch. 3 — line searches and convergence theory.
- 🎥 Andrew Ng, *Machine Learning Specialization* — Course 1, Week 2 — intuitive introduction.
- 🔗 Sebastian Ruder, *"An overview of gradient descent optimization algorithms"* (2017).

## Revisit Log
- 2026-06-07 — created from canonical sources.
