---
id: "20260708231508"
type: permanent
subtype: concept
created: 2026-07-08 23:15:08
aliases:
  - L2
  - Norm
  - Ridge
  - Euclidean
  - Distance
tags:
  - linear-algebra
  - machine-learning
Up: "[[Linear Algebra for Machine Learning]]"
---
# The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points

While the [[The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes|L1 Norm]] calculates distance by moving strictly along grid axes, the **L2 Norm** calculates the "as the crow flies" distance. It is the shortest, straight-line distance between two points in a vector space.

Also known as **Euclidean Distance**, it is simply the $n$-dimensional generalization of the Pythagorean theorem ($a^2 + b^2 = c^2$). Because it relies on squaring the components, it heavily penalizes large values or large errors compared to smaller ones.

### Mathematical Representation

![[file-20260708231716878.jpg]]

For a single vector $\mathbf{x}$, the L2 norm (denoted by the subscript $2$, or often with no subscript as it is the default norm) is the square root of the sum of the squared vector values:

$$||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

When calculating the **Euclidean Distance** between two separate vectors ($\mathbf{x}$ and $\mathbf{y}$), you calculate the L2 norm of their difference:

$$d_2(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

### Python Representation

In Python, the L2 norm is the default behavior of NumPy's `linalg.norm` function. You can also calculate it manually using standard array operations.

``` python
import numpy as np

# Define two 2D vectors
x = np.array([1, 2])
y = np.array([4, 6])

# 1. Calculating Euclidean Distance manually
euclidean_dist_manual = np.sqrt(np.sum(np.square(x - y)))

# 2. Calculating it using NumPy's built-in norm function (ord=2 is default)
euclidean_dist_np = np.linalg.norm(x - y)

print(f"Point X: {x}, Point Y: {y}")
print(f"Euclidean Distance: {euclidean_dist_manual}") 
# Output: sqrt((1-4)^2 + (2-6)^2) = sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = sqrt(25) = 5.0
```

### Semantic Meanings in ML

The L2 norm is arguably the most ubiquitous distance metric in machine learning, forming the backbone of numerous algorithms and optimization techniques:

- **Mean Squared Error (MSE):** The L2 norm is the foundation of MSE, the default loss function for regression problems. Because errors are squared, MSE harshly penalizes large prediction errors (outliers) while being more forgiving of tiny errors.
- **L2 Regularization (Ridge / Weight Decay):** When added to a loss function, the L2 penalty ($||\mathbf{w}||_2^2$) constrains the model's weights. Unlike L1 regularization which creates a diamond constraint, L2 creates a smooth, circular geometric constraint. This shrinks all weights smoothly and evenly without forcing any of them to exactly zero, preventing a single feature from dominating the model.
- **Spatial Algorithms (KNN & K-Means):** Algorithms like K-Nearest Neighbors and K-Means Clustering default to Euclidean distance to find the "closest" data points. However, due to the _Curse of Dimensionality_, L2 distances can become mathematically meaningless in highly sparse, high-dimensional spaces, prompting a switch to L1 or Cosine similarity.

**Related notes to create/link:**

- The grid-based counterpart: [[The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes]]
- Distributing weights evenly: [[L2 Regularization (Ridge) shrinks feature weights evenly without forcing them to zero]]
- Regression optimization: [[Mean Squared Error harshly penalizes large prediction outliers]]
- Distance independent of magnitude: [[Cosine Similarity measures the angle between vectors instead of magnitude]]