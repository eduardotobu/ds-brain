---
title: "Linear Regression"
type: model
family: linear
task: regression
supervision: supervised
year: 1805
authors: [Adrien-Marie Legendre, Carl Friedrich Gauss]
paper: "Legendre (1805), Nouvelles méthodes pour la détermination des orbites des comètes"
status: production-ready
created: 2026-06-07
updated: 2026-06-07
tags: [ml/model, linear, regression, baseline]
aliases: [OLS, Ordinary Least Squares, Linear Least Squares]
---

# Linear Regression

> [!abstract] One-liner
> Predict a continuous target as a linear combination of input features by minimizing squared error — the simplest, most interpretable supervised model and the baseline every regression problem should start with.

## Problem It Solves
- **Input:** feature vector $x \in \mathbb{R}^d$ (numeric, with categoricals one-hot encoded).
- **Output:** continuous scalar $\hat{y} \in \mathbb{R}$.
- **Why use this instead of simpler alternatives:** it's already nearly the simplest non-trivial model. The real comparison is *upward*: prefer linear regression over fancier models when you need interpretability, calibrated coefficients with uncertainty, low latency, or a sanity baseline.

## Model Definition
**Hypothesis / functional form**
$$
\hat{y} \;=\; f_\theta(x) \;=\; w^\top x + b \;=\; \sum_{j=1}^{d} w_j x_j + b
$$

**Parameters:** $\theta = \{w \in \mathbb{R}^d,\; b \in \mathbb{R}\}$ — one weight per feature plus an intercept.

**Inductive bias / assumptions (the classical "OLS" assumptions):**
- **Linearity** of the relationship between $E[y \mid x]$ and $x$.
- **Independence** of observations.
- **Homoscedasticity** — constant error variance across $x$.
- **No (perfect) multicollinearity** among features.
- **Normality of residuals** — only needed for valid $t$- and $F$-tests / confidence intervals, *not* for point estimation.

## Loss & Objective
Mean squared error over $n$ training points $\{(x_i, y_i)\}_{i=1}^n$:

$$
\mathcal{L}(w, b) \;=\; \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - (w^\top x_i + b)\bigr)^2
$$

- **Regularization:**
  - **Ridge ($L_2$):** add $\lambda \|w\|_2^2$ → shrinks coefficients, handles collinearity. See [[Ridge Regression]].
  - **Lasso ($L_1$):** add $\lambda \|w\|_1$ → sparse coefficients / feature selection. See [[Lasso Regression]].
  - **Elastic Net:** mix of both.
- **Constraints:** none in vanilla form. Non-negativity or monotonicity can be added (NNLS, isotonic regression).

## Training / Optimization
Two canonical routes:

**1. Closed-form (Normal Equations).** Stack data into $X \in \mathbb{R}^{n \times d}$ (with a column of ones for the intercept) and $y \in \mathbb{R}^n$:
$$
\hat{\theta} \;=\; (X^\top X)^{-1} X^\top y
$$
In practice, solve via QR or SVD decomposition for numerical stability — never literally invert $X^\top X$.

**2. Iterative ([[Example Concepts - Gradient Descent]] / SGD).** Required when $n$ or $d$ is too large for the closed form.

- **Update rule (full-batch GD):**
$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta \cdot \frac{2}{n} X^\top (X\theta_t - y)
$$
- **Complexity:**
  - Closed form: train $\mathcal{O}(n d^2 + d^3)$, predict $\mathcal{O}(d)$.
  - GD: train $\mathcal{O}(n d)$ per iteration, predict $\mathcal{O}(d)$.
- **Memory:** $\mathcal{O}(n d)$ to hold $X$; $\mathcal{O}(d^2)$ for the normal-equation route.
- **Parallelism / scalability:** trivially parallelizable over rows; for $d$ in the millions use SGD or mini-batch SGD; for $n$ in the billions, fit on sufficient statistics ($X^\top X$, $X^\top y$) computed in a distributed pass.

## Hyperparameters
| Name | Typical range | Effect | Notes |
|------|---------------|--------|-------|
| `fit_intercept` | `True` / `False` | Whether to learn $b$ | Almost always `True` unless data is pre-centered. |
| `alpha` (ridge $\lambda$) | $10^{-4}$ – $10^{2}$ | Shrinkage strength | Tune on log scale via CV. |
| `alpha` (lasso $\lambda$) | $10^{-4}$ – $10^{1}$ | Sparsity strength | Larger ⇒ more zero coefficients. |
| `l1_ratio` (elastic net) | $0$ – $1$ | Mix of L1 vs L2 | $0$ = ridge, $1$ = lasso. |
| `solver` | `svd`, `cholesky`, `lsqr`, `sag` | Numerical method | `svd` is most robust to ill-conditioning. |

## Inference
- **How predictions are produced:** a single dot product $\hat{y} = w^\top x + b$ — microseconds per sample.
- **Uncertainty / calibration:** under classical assumptions, $\hat{w} \sim \mathcal{N}(w, \sigma^2 (X^\top X)^{-1})$, giving closed-form coefficient standard errors, $t$-statistics, $p$-values, and prediction intervals.
- **Latency considerations:** essentially free; suitable for edge devices and high-QPS services.

## Strengths
- ✅ Extremely **interpretable** — each $w_j$ is the expected change in $y$ per unit change in $x_j$, holding others fixed.
- ✅ Closed-form solution → fast, deterministic training.
- ✅ Strong statistical theory (CIs, hypothesis tests, $R^2$).
- ✅ Tiny model artifact, near-zero inference cost.
- ✅ Great **baseline**: if a fancy model can't beat it meaningfully, the fancy model is wrong.

## Weaknesses & Failure Modes
- ⚠️ Assumes linearity; misses interactions and non-linear effects unless you engineer them.
- ⚠️ Sensitive to **outliers** (squared loss). Use [[Huber Regression]] or remove them.
- ⚠️ Coefficients become unstable under **multicollinearity** (huge variance, sign flips). Use ridge or drop collinear features.
- ⚠️ Heteroscedastic errors → biased standard errors; remedy with robust (sandwich) SEs or weighted least squares.
- ⚠️ Extrapolation outside the training range is unreliable.

## Data Requirements
- **Size / dimensionality:** as a rule of thumb, want $n \gtrsim 10 \cdot d$ for stable estimates; with ridge/lasso you can push into $d > n$.
- **Preprocessing:**
  - Standardize features when using regularization (the penalty is scale-dependent).
  - One-hot encode categoricals (drop one level to avoid the dummy-variable trap).
  - Impute missing values; OLS can't handle NaNs.
- **Leakage risks:** target leakage from features computed using future information; "data snooping" from doing feature selection on the full dataset before CV.
- **Class imbalance handling:** N/A (regression). For skewed targets, consider $\log(y)$ transform or quantile regression.

## Evaluation
- **Primary metrics:** RMSE, MAE, $R^2$ (and adjusted $R^2$ for model comparison).
- **Baselines to beat:**
  - Predict the mean ($R^2 = 0$ baseline).
  - A single-feature regression on the most correlated feature.
- **Diagnostic plots:**
  - Residuals vs fitted — should be a structureless blob; funnels indicate heteroscedasticity, curves indicate misspecification.
  - Q-Q plot of residuals — checks normality for inference.
  - Leverage / Cook's distance — flags influential points.
  - Learning curve — diagnose under- vs over-fitting.
- **Cross-validation strategy:** plain $k$-fold (typically $k=5$ or $10$); time-series split if data is temporal.

## Minimal Implementation
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# toy data: y = 2*x1 - 1*x2 + 3 + noise
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 2))
y = 2 * X[:, 0] - 1 * X[:, 1] + 3 + rng.normal(scale=0.5, size=500)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_tr, y_tr)
pred  = model.predict(X_te)

print("coef:", model.coef_, "intercept:", model.intercept_)
print(f"RMSE: {mean_squared_error(y_te, pred, squared=False):.4f}")
print(f"R^2 : {r2_score(y_te, pred):.4f}")

# with L2 regularization
ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
```

## Variants & Extensions
- [[Ridge Regression]] — adds $L_2$ penalty for stability under collinearity.
- [[Lasso Regression]] — adds $L_1$ penalty for sparsity / feature selection.
- [[Elastic Net]] — combines $L_1$ and $L_2$.
- [[Polynomial Regression]] — linear in *transformed* features for non-linear fits.
- [[Generalized Linear Models]] — extends to non-Gaussian targets via a link function.
- [[Bayesian Linear Regression]] — places priors on $w$, yields posterior over predictions.
- [[Quantile Regression]] — models conditional quantiles instead of the mean.
- [[Huber Regression]] — robust to outliers.

## Related Models
- **Predecessor:** the [[Method of Least Squares]] itself (Legendre 1805, Gauss 1809).
- **Successor:** [[Generalized Linear Models]] (1972) — same linear predictor, broader response distributions.
- **Competitor:** [[Gradient Boosting Machines]], [[Random Forest]] (when non-linear interactions dominate).
- **Underlying concepts:** [[Example Concepts - Gradient Descent]], [[Maximum Likelihood Estimation]] (OLS = MLE under Gaussian errors), [[Bias-Variance Tradeoff]], [[Regularization]].

## When to Use (Decision Notes)
- **Reach for this when:**
  - You need an interpretable, defensible model (regulated industries, scientific reporting).
  - You have small data and risk overfitting with flexible models.
  - You need calibrated uncertainty on coefficients.
  - You need a baseline before reaching for tree ensembles or neural nets.
  - Latency / model-size budgets are tight.
- **Avoid when:**
  - Relationships are clearly non-linear and you can't or won't engineer features.
  - There are strong, unknown feature interactions that tree ensembles would handle automatically.
  - You have many highly collinear features and no regularization budget — at minimum switch to ridge.

## Production Considerations
- **Serving format:** pickle / joblib for sklearn; a hand-rolled dot product if you want zero dependencies. ONNX export is trivial.
- **Retraining cadence:** depends on drift; for stable phenomena, quarterly or yearly is often fine. Linear models are cheap enough to retrain nightly.
- **Drift / monitoring signals:**
  - Distribution shift in feature means/variances (PSI, KS test).
  - Residual mean drifting away from zero.
  - Coefficient stability across retraining windows.
- **Explainability tools:** the coefficients **are** the explanation. Optionally pair with SHAP for per-prediction attributions (SHAP for a linear model has a closed form: $\phi_j = w_j (x_j - \bar{x}_j)$).

## References
- 📄 Legendre, A.-M. (1805). *Nouvelles méthodes pour la détermination des orbites des comètes.*
- 📄 Gauss, C. F. (1809). *Theoria motus corporum coelestium.* — introduces the Gaussian-error justification.
- 📚 Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 3 — definitive ML treatment.
- 📚 Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models* — applied / statistical perspective.
- 🛠️ [scikit-learn — `LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- 🛠️ [statsmodels — `OLS`](https://www.statsmodels.org/stable/regression.html) (for inference-grade output).
- 🎥 Andrew Ng, *Machine Learning Specialization*, Course 1 — accessible derivation and intuition.

## Revisit Log
- 2026-06-07 — created as a canonical example of the Model template.
