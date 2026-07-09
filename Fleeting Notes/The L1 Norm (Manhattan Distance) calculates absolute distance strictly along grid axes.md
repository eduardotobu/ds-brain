---
id: "20260708230454"
type: permanent
subtype: concept
created: 2026-07-08 23:04:54
aliases:
  - L1
  - Lasso
  - Manhattan
  - Distance
tags:
  - linear-algebra
  - norm
Up: "[[Linear Algebra for Machine Learning]]"
---
# The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes

![[file-20260708230734771.jpg]]

In vector spaces, a "norm" is a function that measures the size, length, or magnitude of a vector. The **L1 Norm**—commonly referred to as **Manhattan Distance** or Taxicab Geometry—measures this distance by strictly summing the absolute differences of its components.

It is called the "Manhattan Distance" because it calculates distance as if you were walking the grid-like streets of Manhattan: you cannot travel diagonally through city blocks; you can only move horizontally and vertically along the axes.

### Mathematical Representation

For a single vector $\mathbf{x}$, the L1 norm (denoted by the subscript $1$) is the sum of the absolute values of its individual scalar components:

$$||\mathbf{x}||_1 = \sum_{i=1}^n |x_i|$$

When calculating the **Manhattan Distance** between two separate vectors ($\mathbf{x}$ and $\mathbf{y}$), you calculate the L1 norm of their difference:

$$d_1(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_1 = \sum_{i=1}^n |x_i - y_i|$$

### Python Representation

In Python, you can calculate the L1 norm using NumPy's `linalg.norm` function and specifying the `ord=1` parameter, or by manually summing the absolute differences.

``` python
import numpy as np

# Define two 2D vectors (points on a grid)
x = np.array([1, 2])
y = np.array([4, 6])

# 1. Calculating Manhattan Distance manually
manhattan_dist_manual = np.sum(np.abs(x - y))

# 2. Calculating it using NumPy's built-in norm function
manhattan_dist_np = np.linalg.norm(x - y, ord=1)

print(f"Point X: {x}, Point Y: {y}")
print(f"Manhattan Distance: {manhattan_dist_manual}") 
# Output: |1-4| + |2-6| = |-3| + |-4| = 3 + 4 = 7.0
```

### Semantic Meanings in ML

The geometric properties of the L1 norm make it extremely valuable in specific machine learning scenarios:

- **Lasso Regularization (L1 Regularization):** When added as a penalty to a neural network or regression model's loss function, L1 regularization penalizes the absolute size of the [[Weight Vector]]. Because of its diamond-shaped geometric constraint, it forces many less-important feature weights to become **exactly zero**. This acts as an automatic feature selector, producing _sparse_ models that are faster and easier to interpret.
- **Mean Absolute Error (MAE):** The L1 norm forms the basis of the MAE loss function. Unlike L2/MSE (which squares errors), MAE measures the direct absolute distance. This makes MAE **robust to outliers**, as massive errors aren't exponentially magnified by a square function.
- **High-Dimensional Distance:** As the dimensionality of a dataset grows massive (e.g., thousands of features), diagonal distances (Euclidean/L2) begin to lose their meaning—a phenomenon part of the "Curse of Dimensionality." Manhattan distance often remains a more reliable metric for clustering algorithms (like K-Means or K-Nearest Neighbors) in high-dimensional spaces.

**Related notes to create/link:**

- The diagonal counterpart: [[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points]]
- Generating sparse models: [[L1 Regularization (Lasso) forces feature weights to exactly zero, creating sparse models]]
- Dealing with outliers: [[Mean Absolute Error is robust against extreme dataset outliers]]