---
id: "20260708231916"
type: permanent
subtype: concept
created: 2026-07-08 23:19:16
aliases:
  - L2 Regularization
  - Ridge Regularization
tags:
  - ml
  - linear-algebra
  - regularization
Up: "[[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points]]"
---
# L2 Regularization (Ridge) shrinks feature weights evenly without forcing them to zero

If [[L1 Regularization (Lasso) forces feature weights to exactly zero, creating sparse models|L1 Regularization (Lasso)]] acts like a scalpel that cuts away irrelevant features by forcing their weights to exactly zero, **L2 Regularization (Ridge)** acts like a compressive weight blanket. It prevents overfitting by penalizing large weights, shrinking them evenly across the board, but rarely reducing them to absolute zero.

Geometrically, the L2 penalty creates a circular (or spherical) constraint in the vector space. When the model tries to minimize its error while staying within this circular boundary, the loss contour usually touches the circle at a point that is _off_ the axes. Because it does not hit the sharp corners of the axes like the L1 diamond does, the resulting weights are small but retain non-zero values.

### Mathematical Representation

To apply Ridge Regularization, we add the squared L2 norm of the weight vector ($||\mathbf{w}||_2^2$) multiplied by a regularization hyperparameter $\lambda$ (lambda) to the standard loss function $J(\mathbf{w})$:

$$J_{L2}(\mathbf{w}) = J(\mathbf{w}) + \lambda ||\mathbf{w}||_2^2$$

Expanded to show the scalar components:

$$J_{L2}(\mathbf{w}) = J(\mathbf{w}) + \lambda \sum_{i=1}^n w_i^2$$

_(Note: In calculus-based optimization, you will often see a $\frac{1}{2}$ added in front of the penalty term. This is just a mathematical trick so that when the derivative is taken, the exponent 2 cancels out the fraction, leaving cleaner math)._

### Python Representation

In traditional machine learning, this is implemented using the `Ridge` model in Scikit-Learn. In deep learning frameworks like PyTorch, L2 regularization is often referred to as **weight decay** and is built directly into the optimizer (like Adam or SGD).


``` PYTHON
import numpy as np
from sklearn.linear_model import Ridge

# Simulated dataset: 3 samples, 4 features
X = np.array([[1, 2, 0.5, 8], 
              [2, 4, 1.1, 7], 
              [3, 6, 0.9, 9]])
y = np.array([10, 20, 30])

# Initialize Ridge with a regularization strength (alpha represents lambda)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

# Print the learned weight vector
print(f"Learned Weights: {ridge_model.coef_}")
# Output might look like: [ 1.2,  2.4,  0.8, -0.5]
# Notice how all features retain a weight (none are exactly 0.0)
```

### Semantic Meanings in ML

Ridge regularization is the default choice for many continuous ML models due to its mathematical stability:

- **Handling Multicollinearity:** If two features are highly correlated (linearly dependent), an unregularized model might assign massive, opposing weights to them. Ridge regularization mathematically prevents this. It distributes the weight evenly between the correlated features, leading to much more stable and reliable predictions.
- **Fixing Singular Matrices:** In Ordinary Least Squares (OLS) regression, if your design matrix $\mathbf{X}$ is singular (non-invertible), the equation crashes. The Ridge penalty alters the math to $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$. Adding the [[The identity matrix acts as the multiplicative identity, leaving any vector or matrix unchanged|identity matrix]] $\mathbf{I}$ guarantees the matrix becomes full rank and invertible, artificially "curing" the singularity.
- **Retaining Dense Information:** Sometimes, every single feature in your dataset contains a tiny, useful signal (e.g., pixel values in an image). You don't want to throw any away. L2 is perfect here because it prevents any single pixel from dominating the model while keeping all pixels active.

**Related notes to create/link:**

- The base distance metric: [[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points]]
- The sparse alternative: [[L1 Regularization (Lasso) forces feature weights to exactly zero, creating sparse models]]
- Curing singular matrices algebraically: [[Adding a scalar to the diagonal of a matrix guarantees invertibility]]
- Implementation in Neural Networks: [[Weight Decay in PyTorch is mathematically equivalent to L2 Regularization]]
