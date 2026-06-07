---
id: 202606041600
title: "Does pre-training eliminate the bias-variance tradeoff in fine-tuning?"
created: 2026-06-04
tags:
  - question
  - ml
type: research-question
status: open
area: transfer-learning
---

# Does pre-training eliminate the bias-variance tradeoff in fine-tuning?

## Question

When fine-tuning a large pre-trained model on a small downstream task, does the pre-training act as such a strong prior that variance is negligible — effectively breaking the classical bias-variance tradeoff? Or does it just shift the curve?

## Why it matters

If true, it changes how we think about model selection for small datasets: instead of "use a simpler model to avoid overfitting", the answer becomes "use the largest pre-trained model and fine-tune carefully". This directly affects our churn model architecture decisions.

## Hypotheses

- H1: Pre-training drastically reduces variance (the model starts near a good solution, so different training runs converge similarly). The tradeoff still exists but is shifted far right — you can use much more complex models before variance kicks in.
- H2: The tradeoff is just hidden — variance manifests as sensitivity to fine-tuning hyperparameters (learning rate, number of epochs) rather than sensitivity to training data.
- H3: Double descent explains it — pre-trained models are in the "interpolation regime" where more parameters = less overfitting.

## How to test

- [ ] Run the DistilBERT churn experiment 10 times with different seeds → measure prediction variance across runs
- [ ] Compare variance of fine-tuned DistilBERT vs. trained-from-scratch small transformer on same data
- [ ] Read Nakkiran et al. 2020 "Deep Double Descent" — does it address transfer learning?
- [ ] Check if anyone has done bias-variance decomposition on fine-tuned models empirically

## Related work

- [[Bias-Variance Tradeoff]] — classical formulation
- [[Double Descent]] — challenges the classical view
- Nakkiran et al. 2020 — "Deep Double Descent"
- Neyshabur et al. 2020 — on generalization in over-parameterized models

## Connections

- Arose from: [[Fine-tune DistilBERT for churn classification]]
- Blocks: [[Deciding model complexity for small-data projects]]
- Related: [[Bias-Variance Tradeoff]], [[Transfer Learning]]

## Answer

<!-- Fill when resolved. Change status to "resolved". Include evidence. -->

