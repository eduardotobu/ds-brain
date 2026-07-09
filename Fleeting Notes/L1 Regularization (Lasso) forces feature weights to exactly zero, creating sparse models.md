---
id: "20260708230926"
type: permanent
subtype: concept
created: 2026-07-08 23:09:26
aliases:
  - L1
  - Regularization
  - Lasso
  - Manhattan
tags:
  - regularization
  - linear-algebra
  - machine-learning
Up: "[[The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes]]"
---
# L1 Regularization (Lasso) forces feature weights to exactly zero, creating sparse models

In machine learning, models often memorize the training data (overfitting) by assigning large weights to noisy or irrelevant features. **Lasso Regularization** (Least Absolute Shrinkage and Selection Operator) prevents this by adding an "L1 penalty" to the model's loss function.

Because the L1 penalty calculates distance using absolute values, its geometric constraint forms a diamond shape in the vector space. When the model tries to minimize its error while staying within this diamond constraint, the optimal solution almost always hits the sharp corners of the diamond—which lie exactly on the axes. This mathematical quirk forces the weights of less important features to become **exactly zero**.

### Mathematical Representation

When training a model without regularization, the goal is to minimize a standard loss function, such as Mean Squared Error, denoted as $J(\mathbf{w})$.

With Lasso Regularization, we add the L1 norm of the weight vector ($||\mathbf{w}||_1$) multiplied by a tuning parameter $\lambda$ (lambda) to the loss function:

$$J_{L1}(\mathbf{w}) = J(\mathbf{w}) + \lambda ||\mathbf{w}||_1$$

Expanded to show the scalar components:

$$J_{L1}(\mathbf{w}) = J(\mathbf{w}) + \lambda \sum_{i=1}^n |w_i|$$

- If $\lambda = 0$, there is no regularization (standard model).
- As $\lambda$ increases, the penalty grows, and more weights are driven identically to $0$.

### Python Representation

While you can implement this manually in PyTorch by adding `torch.abs(weights).sum()` to your loss, it is most commonly used in Scikit-Learn for linear models.


``` python
import numpy as np
from sklearn.linear_model import Lasso

# Simulated dataset: 3 samples, 4 features
X = np.array([[1, 2, 0.5, 8], 
              [2, 4, 1.1, 7], 
              [3, 6, 0.9, 9]])
y = np.array([10, 20, 30])

# Initialize Lasso with a regularization strength (alpha represents lambda)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)

# Print the learned weight vector
print(f"Learned Weights: {lasso_model.coef_}")
# Output might look like: [ 5.0,  0.0,  0.0,  0.0]
# Notice how the L1 penalty forced features 2, 3, and 4 to exactly 0.0
```

### Semantic Meanings in ML

Lasso is highly favored in specific data environments for its unique properties:
- **Automatic Feature Selection:** Because it zeros out irrelevant weights, Lasso automatically selects only the most important features. If you feed a Lasso model 10,000 genomic markers, it might return only the 15 markers that actually predict the disease.
- **Model Interpretability:** A sparse model (a model with mostly zero weights) is vastly easier for a human to understand and audit than a dense model where every single feature contributes a tiny fraction to the prediction.
- **High-Dimensional Data ($n > m$):** When you have more features than actual data rows (like in text processing or bioinformatics), standard regression fails because the matrix is [[Matrix singularity determines if a linear system is complete, redundant, or contradictory|singular]]. Lasso effectively drops the dimensionality until the system is solvable.

**Related notes to create/link:**

- The base distance metric: [[The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes]]
- The circular constraint alternative: [[L2 Regularization (Ridge) shrinks feature weights evenly without forcing them to zero]]
- Combining both constraints: [[Elastic Net combines L1 and L2 penalties for balanced regularization]]
- Controlling the penalty: [[Hyperparameter tuning finds the optimal lambda for regularization]]