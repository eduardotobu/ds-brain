---
id: "20260719190917"
type: permanent
subtype: concept
created: 2026-07-19 19:09:17
aliases: []
tags:
  - linear-algebra
Up:
---
# Matrix-vector multiplication represents a linear transformation mapping a vector from one space to another


Matrix-vector multiplication is the fundamental operation for applying a linear transformation to a vector, mapping it from an input space of $n$ dimensions to an output space of $m$ dimensions. Geometrically, this operation can rotate, stretch, shrink, or shear the input vector. Algebraically, the operation can be conceptualized in two distinct ways: computing the dot product of the input vector with each row of the matrix, or computing a linear combination of the matrix's columns weighted by the elements of the input vector.

We can see a vector-matrix multiplication as n dot-products (the vector dot product for each matrix line). So we can represent system of equations like this too: 
![[file-20260719191042897.jpg]]
### Mathematical Representation

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ be an $m \times n$ matrix and $\mathbf{x} \in \mathbb{R}^n$ be an $n \times 1$ column vector. The matrix-vector product is a new column vector $\mathbf{y} \in \mathbb{R}^m$, denoted as $\mathbf{y} = \mathbf{A}\mathbf{x}$.

**1. The Row Perspective (Dot Product):**

The $i$-th component of the output vector $\mathbf{y}$ is the dot product of the $i$-th row of $\mathbf{A}$ and the vector $\mathbf{x}$:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j$$

**2. The Column Perspective (Linear Combination):**

The output vector $\mathbf{y}$ is a linear combination of the column vectors of $\mathbf{A}$, denoted as $\mathbf{a}_j$, where the scalar weights are the corresponding elements of $\mathbf{x}$:

$$\mathbf{A}\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n$$

### Python Representation

``` python
import numpy as np

# Initialize a 3x2 matrix A (m=3, n=2)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Initialize a 2x1 vector x (n=2)
x = np.array([10, 20])

# Perform matrix-vector multiplication using the @ operator (or np.dot)
y = A @ x

print(f"Matrix A (shape {A.shape}):\n{A}")
print(f"Vector x (shape {x.shape}):\n{x}")
print(f"Resulting Vector y (shape {y.shape}):\n{y}")
# Output:
# Resulting Vector y (shape (3,)):
# [ 50 110 170]

# Verification via Column Perspective (Linear Combination):
col_1 = A[:, 0]
col_2 = A[:, 1]
y_linear_combo = (x[0] * col_1) + (x[1] * col_2)
print(f"Verified via Linear Combination:\n{y_linear_combo}")
```

### Semantic Meanings in ML

- **Dense (Fully Connected) Layers:** In deep learning, the core operation of a standard neural network layer is a matrix-vector multiplication representing $f(\mathbf{x}) = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$. The weight matrix $\mathbf{W}$ transforms the input features $\mathbf{x}$ into a new latent representation space.
- **Dimensionality Reduction:** Algorithms like Principal Component Analysis (PCA) rely on multiplying a data vector by a projection matrix containing eigenvectors. If a $k \times n$ matrix projects an $n$-dimensional vector, it compresses the data into a lower-dimensional $k$-space while attempting to preserve maximum variance.
- **Word Embeddings and Lookup Tables:** Multiplying a one-hot encoded vector by an embedding matrix effectively "selects" the corresponding column of the matrix, transforming a sparse, high-dimensional vocabulary representation into a dense, lower-dimensional semantic embedding.
- **Markov Chains and PageRank:** In probability and graph theory algorithms, multiplying a state vector by a stochastic transition matrix updates the probabilities of moving from one state (or webpage) to the next in a sequence.
