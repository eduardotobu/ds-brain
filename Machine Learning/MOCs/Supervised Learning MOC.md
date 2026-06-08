---
id: 202606071811
title: Supervised Learning MOC
created: 2026-06-07
updated: 2026-06-07
tags:
  - moc
  - ml
  - supervised-learning
type: moc
status: 🌱 seedling
---
# MOC: Supervised Learning

## Overview

Supervised learning is the branch of ML where a model learns a mapping from inputs $X$ to outputs $Y$ given a labeled training set $\{(x_i, y_i)\}_{i=1}^n$. The field splits cleanly along two axes: **task type** (classification vs. regression) and **model family** (linear, tree-based, kernel, or neural). Mastering supervised learning means understanding not just individual algorithms but the *bias–variance tradeoff* that governs all of them — every design decision is a negotiation between underfitting and overfitting. This MOC is the entry point to that landscape: start with the foundational concepts, pick a model family, study its canonical architectures, and then follow the open questions into active research territory.

## Foundational concepts

- [[The bias–variance tradeoff governs all supervised learning]]
- [[Empirical risk minimization is the formal basis for supervised training]]
- [[Generalization requires controlling model complexity, not just training loss]]
- [[The i.i.d. assumption underlies most supervised learning theory]]
- [[Loss functions encode the task: cross-entropy for classification, MSE for regression]]
- [[Regularization prevents overfitting by penalizing hypothesis complexity]]
- [[Maximum likelihood estimation unifies many supervised objectives]]
- [[The curse of dimensionality makes distance-based methods brittle in high dimensions]]
- [[Overfitting and underfitting]]
- [[Train–validation–test split discipline]]
- [[Data leakage silently invalidates supervised learning experiments]]

## Models & Techniques

### Linear Models
- [[Linear Regression]]  - closed-form solution via the normal equations
- [[Logistic Regression — probabilistic linear classifier via sigmoid]]
- [[Ridge regression adds L2 penalty to shrink coefficients toward zero]]
- [[Lasso regression induces sparsity via L1 penalty]]
- [[Support Vector Machines maximize the margin between classes]]
- [[Kernel trick maps inputs to high-dimensional feature spaces implicitly]]

### Tree-Based Methods
- [[Decision Tree — axis-aligned partitions learned by greedy splitting]]
- [[Random Forest — variance reduction via bagging and feature subsampling]]
- [[Gradient Boosting builds an additive model by fitting residuals sequentially]]
- [[XGBoost — regularized gradient boosting with second-order Taylor approximation]]
- [[LightGBM — histogram-based gradient boosting for large datasets]]

### Neural Networks
- [[Multilayer Perceptron — universal approximator via stacked affine + nonlinear layers]]
- [[Convolutional Neural Network — translation-equivariant feature extractor]]
- [[Recurrent Neural Network — sequential processing with hidden state]]
- [[Transformer — attention-based sequence model with global receptive field]]
- [[Batch Normalization reduces internal covariate shift and stabilizes training]]
- [[Dropout regularizes by training an ensemble of thinned networks]]
- [[Residual connections enable gradient flow in very deep networks]]

### Probabilistic & Bayesian Methods
- [[Gaussian Naive Bayes — generative classifier under feature independence assumption]]
- [[Gaussian Process Regression — nonparametric Bayesian regression with uncertainty]]
- [[Calibration ensures predicted probabilities reflect true frequencies]]

### Instance-Based Methods
- [[k-Nearest Neighbors — prediction by majority vote or average of neighbors]]

## Key papers

<!-- Seminal first, then influential follow-ups. -->

- [[Vapnik & Cortes 1995 — Support-Vector Networks]]
- [[Breiman 2001 — Random Forests]]
- [[Friedman 2001 — Greedy Function Approximation: A Gradient Boosting Machine]]
- [[Chen & Guestrin 2016 — XGBoost: A Scalable Tree Boosting System]]
- [[LeCun et al. 1998 — Gradient-Based Learning Applied to Document Recognition]]
- [[Vaswani et al. 2017 — Attention Is All You Need]]
- [[He et al. 2016 — Deep Residual Learning for Image Recognition]]
- [[Srivastava et al. 2014 — Dropout: A Simple Way to Prevent Neural Networks from Overfitting]]
- [[Ioffe & Szegedy 2015 — Batch Normalization: Accelerating Deep Network Training]]
- [[Guo et al. 2017 — On Calibration of Modern Neural Networks]]

## Experiments & Projects

<!-- Link experiment logs as you create them. -->

- [[Effect of regularization strength on logistic regression generalization gap]]
- [[Comparing gradient boosting implementations on tabular benchmark datasets]]
- [[Ablation: does batch normalization help more than dropout on small datasets?]]
- [[Learning curve analysis: when does more data help vs. a better model?]]

## Open questions

- [[Does label smoothing improve calibration independently of accuracy?]]
- [[When does gradient boosting outperform neural networks on tabular data?]]
- [[How much does feature scaling matter for tree-based vs. linear models?]]
- [[Is there a principled way to choose between cross-entropy and focal loss?]]

## Learning path

*Suggested reading order for someone new to supervised learning.*

1. Start with [[The bias–variance tradeoff governs all supervised learning]] — it is the unifying lens for everything else.
2. Read [[Empirical risk minimization is the formal basis for supervised training]] to understand what "learning" formally means.
3. Study [[Linear Regression — closed-form solution via the normal equations]] and [[Logistic Regression — probabilistic linear classifier via sigmoid]] as the simplest instantiations.
4. Add [[Regularization prevents overfitting by penalizing hypothesis complexity]] (Ridge, Lasso, early stopping).
5. Move to [[Decision Tree — axis-aligned partitions learned by greedy splitting]], then [[Random Forest — variance reduction via bagging and feature subsampling]].
6. Study [[Gradient Boosting builds an additive model by fitting residuals sequentially]] and read [[Chen & Guestrin 2016 — XGBoost: A Scalable Tree Boosting System]].
7. Enter neural networks via [[Multilayer Perceptron — universal approximator via stacked affine + nonlinear layers]].
8. Add depth: [[Residual connections enable gradient flow in very deep networks]], [[Batch Normalization reduces internal covariate shift and stabilizes training]], [[Dropout regularizes by training an ensemble of thinned networks]].
9. Specialize: follow [[MOC: CNNs]], [[MOC: Transformers]], or [[MOC: Tree-Based Methods]] depending on your task.
10. Close the loop with [[MOC: Evaluation & Metrics]] — a model is only as good as your ability to measure it.

## Narrative

Supervised learning began as a statistical problem: given observations, fit a model. The first generation of solutions — linear regression, logistic regression, discriminant analysis — were elegant but brittle, constrained by linearity and by the curse of dimensionality. The introduction of the [[Kernel trick maps inputs to high-dimensional feature spaces implicitly]] and the [[Support Vector Machines maximize the margin between classes]] in the 1990s offered a theoretically grounded escape from linearity without explicit feature engineering, but SVMs scaled poorly with data size.

Tree-based methods — [[Decision Tree — axis-aligned partitions learned by greedy splitting]], [[Random Forest — variance reduction via bagging and feature subsampling]], and eventually [[Gradient Boosting builds an additive model by fitting residuals sequentially]] — attacked the problem differently: by combining many weak learners, they achieved strong performance on structured/tabular data with minimal preprocessing and natural handling of categorical features. To this day, gradient boosting variants like [[XGBoost — regularized gradient boosting with second-order Taylor approximation]] and [[LightGBM — histogram-based gradient boosting for large datasets]] dominate tabular benchmarks.

Neural networks, largely dormant through the 1990s, returned with force after 2012 when deep [[Convolutional Neural Network — translation-equivariant feature extractor]]s demonstrated that with sufficient data and compute, learned representations outperform hand-crafted features. The key enablers were [[Residual connections enable gradient flow in very deep networks]], [[Batch Normalization reduces internal covariate shift and stabilizes training]], and [[Dropout regularizes by training an ensemble of thinned networks]]. The [[Transformer — attention-based sequence model with global receptive field]] then dethroned both CNNs and RNNs on sequence tasks and is now expanding into vision and tabular domains. The open question is whether a unified architecture will eventually dominate all of supervised learning, or whether tabular data will remain tree-country for the foreseeable future.

## To-do

- [ ] Create Concept Notes for every foundational concept linked above
- [ ] Create Model Cards for all architectures listed under Models & Techniques
- [ ] Create Paper Notes for all key papers listed above
- [ ] Add Dataview query: `WHERE type = "concept" AND status = "seedling"` to track concept debt
- [ ] Expand [[MOC: Linear Models]], [[MOC: Tree-Based Methods]], [[MOC: Neural Networks]], [[MOC: Evaluation & Metrics]] as child MOCs
- [ ] Link this MOC into [[MOC: Machine Learning]] as a child
- [ ] Review open questions and convert the most tractable one into a [[Research Question]] note