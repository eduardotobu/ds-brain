---
title: "{{title}}"
type: model
family:        # e.g. linear, tree-based, neural net, probabilistic, ensemble
task:          # classification / regression / clustering / generative / ranking / RL
supervision:   # supervised / unsupervised / self-supervised / semi-supervised / RL
year:
authors:
paper:
status: studying   # studying / understood / production-ready
created: {{date:YYYY-MM-DD}}
updated: {{date:YYYY-MM-DD}}
tags: [ml/model]
aliases: []
---

# {{title}}

> [!abstract] One-liner
> What the model is and what it’s for, in a single sentence.

## Problem It Solves
- Input: …
- Output: …
- Why use this instead of simpler alternatives:

## Model Definition
**Hypothesis / functional form**
$$
\hat{y} = f_\theta(x) = \dots
$$

**Parameters:** $\theta = \{\dots\}$

**Inductive bias / assumptions:**
- …
- …

## Loss & Objective
$$
\mathcal{L}(\theta) = \dots
$$

- Regularization:
- Constraints:

## Training / Optimization
- Algorithm: (e.g. SGD, Adam, EM, coordinate descent, MLE)
- Update rule:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$
- Complexity: train $O(\cdot)$, predict $O(\cdot)$
- Memory:
- Parallelism / scalability notes:

## Hyperparameters
| Name | Typical range | Effect | Notes |
|------|---------------|--------|-------|
|      |               |        |       |

## Inference
- How predictions are produced:
- Uncertainty / calibration:
- Latency considerations:

## Strengths
- ✅ …
- ✅ …

## Weaknesses & Failure Modes
- ⚠️ …
- ⚠️ …

## Data Requirements
- Size / dimensionality:
- Preprocessing (scaling, encoding, imputation):
- Leakage risks:
- Class imbalance handling:

## Evaluation
- Primary metric(s):
- Baselines to beat:
- Diagnostic plots (residuals, ROC, calibration, learning curves):
- Cross-validation strategy:

## Minimal Implementation
```python
# fit + predict with a canonical library (sklearn / pytorch / statsmodels / xgboost ...)
```

## Variants & Extensions
- [[ ]] — what it changes
- [[ ]] — what it changes

## Related Models
- **Predecessor:** [[ ]]
- **Successor:** [[ ]]
- **Competitor:** [[ ]]
- **Underlying concepts:** [[ ]], [[ ]]

## When to Use (Decision Notes)
- Reach for this when:
- Avoid when:

## Production Considerations
- Serving format (pickle, ONNX, TorchScript, native):
- Retraining cadence:
- Drift / monitoring signals:
- Explainability tools (SHAP, LIME, feature importance):

## References
- 📄 Original paper —
- 📚 Textbook reference —
- 🛠️ Library docs —
- 🎥 Lecture / talk —

## Revisit Log
- {{date:YYYY-MM-DD}} — created
