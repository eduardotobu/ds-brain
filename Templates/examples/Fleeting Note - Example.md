---
id: 202606071423
title: "Dropout at inference as uncertainty estimation"
created: 2026-06-07 14:23
tags:
  - fleeting
  - ml
type: fleeting
status: inbox
---

# Dropout at inference as uncertainty estimation

## Quick capture

Saw a mention that keeping dropout active during inference (MC Dropout) gives you a cheap approximation of Bayesian uncertainty. Run the same input N times → variance of outputs ≈ model uncertainty. Could be useful for our credit scoring model where we need confidence intervals without switching to a full Bayesian NN.

## Context

Reading Gal & Ghahramani 2016 while debugging the calibration issues in the churn model. The model is overconfident on edge cases — MC Dropout could flag those without retraining.

## Next step

- [ ] Process → convert into concept note, model card, or experiment log
