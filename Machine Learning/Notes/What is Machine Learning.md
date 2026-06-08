---
id: 202606071816
title: What is Machine Learning
created: 2026-06-07
updated: 2026-06-07
tags:
  - concept
  - ml
type: concept
status: 🌿 growing
area: fundamentals
---

# What is Machine Learning?

## Definition

Machine learning as a discipline aims to design understand and apply computer programs that learn from experience (i.e., data) for the purpose of modeling, prediction, or control.
a
## Intuition

In traditional programming, you provide the computer with data and explicit rules (code) to output an answer. In machine learning, you provide the computer with data and the *answers*, and the system outputs the *rules*. For example, instead of writing complex geometric rules to identify a cat in a photo, you feed the system thousands of images labeled "cat" and "not cat," allowing it to discover the distinguishing features on its own.

## Formal description

According to Tom Mitchell (1997), a computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$. 
Mathematically, it is often framed as finding a function $f$ that maps inputs $X$ to outputs $Y$ while minimizing an error term $\epsilon$: 
$$ Y = f(X) + \epsilon $$

## When it applies

- **Complex rule environments:** Tasks where manual rule-writing is nearly impossible (e.g., natural language processing, computer vision, voice recognition).
- **Dynamic systems:** Environments that change over time, requiring the system to continuously adapt to new data (e.g., fraud detection, stock market forecasting).
- **Scale:** Discovering hidden, high-dimensional patterns in massive datasets that humans cannot process manually.

## When it fails

- **Data scarcity:** When you do not have enough high-quality, representative data to train the model.
- **Deterministic tasks:** When 100% predictable outcomes and absolute precision are required (e.g., simple payroll calculators, basic arithmetic).
- **Strict interpretability needs:** When stakeholders require an exact logical trace of how a decision was made, and black-box models (like deep neural networks) are used.

## Common pitfalls

- **Overfitting:** The model memorizes the training data perfectly but fails to generalize to new, unseen data.
- **Underfitting:** The model is too simple to capture the underlying patterns in the data.
- **Data Leakage:** Information from outside the training dataset is used to create the model, leading to overly optimistic performance metrics.
- **Bias:** Training the model on unrepresentative or prejudiced data, which the system then amplifies.

## Code snippet

```python
from sklearn.ensemble import RandomForestClassifier

# 1. Instantiate the model 
model = RandomForestClassifier()

# 2. "Learn" from the data (Experience E)
model.fit(X_train, y_train)

# 3. Make predictions on new data (Task T)
predictions = model.predict(X_test)
```

## Connections

- Prerequisites: [[Basic Programming]], [[Mathematics for Machine Learning MOC]]
- Used by: [[Supervised Learning MOC]], [[Deep Learning MOC]]
- Related: [[Statistics]], [[Artificial Intelligence MOC]]
- Contrast with: [[Traditional Software Engineering]], [[Symbolic AI]]

## References

- Mitchell, T. M. (1997). _Machine Learning_. McGraw-Hill.
