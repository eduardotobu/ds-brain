---
id: "20260809150000"
type: permanent
subtype: concept
created: 2026-08-09
aliases:
  - Identity Matrix
  - I
tags:
  - linear-algebra
Up: "[[Matrices are collections of vectors representing tabular data or weights]]"
---
# The identity matrix acts as the multiplicative identity, leaving any vector or matrix unchanged

The **identity matrix** ($\mathbf{I}$) is a special square matrix with $1$s along its main diagonal and $0$s everywhere else. It is the matrix equivalent of the number $1$ in scalar arithmetic: multiplying any vector or matrix by $\mathbf{I}$ returns that exact same vector or matrix, completely unchanged.

Geometrically, $\mathbf{I}$ represents the **"do nothing" linear transformation**. Its columns are simply the standard basis vectors ($\hat{e}_1, \hat{e}_2, \dots, \hat{e}_n$), so every basis vector maps to itself and no point in space ever moves.

### Mathematical Representation

For an $n \times n$ identity matrix, the entry at row $i$, column $j$ is defined using the Kronecker delta:

$$(\mathbf{I}_n)_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

For example, the $3 \times 3$ identity matrix is:

$$\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Its defining property, for any compatible matrix $\mathbf{A}$ and vector $\mathbf{x}$:

$$\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A} \qquad \mathbf{I}\mathbf{x} = \mathbf{x}$$

### Key Properties

- **Multiplicative identity:** $\mathbf{I}$ is the unique matrix satisfying $\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$ for every square $\mathbf{A}$ of matching size.
- **Non-singular by definition:** $\det(\mathbf{I}) = 1$ and $\text{rank}(\mathbf{I}_n) = n$ — it is always full rank, never [[Matrix singularity determines if a linear system is complete, redundant, or contradictory|singular]].
- **Self-inverse:** $\mathbf{I}^{-1} = \mathbf{I}$.
- **RREF target:** A non-singular square matrix always reduces to $\mathbf{I}$ under [[Gaussian elimination and row reduction|Gaussian elimination]] — this is exactly what [[Row Echelon Form and RREF expose the structural foundation of a matrix|RREF]] looks like for a complete system.
- **Basis vectors:** Its columns are the standard basis vectors of $\mathbb{R}^n$, which is why it leaves every vector's coordinates untouched.

### Python Representation

``` python
import numpy as np

# Create a 3x3 identity matrix
I = np.eye(3)
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

A = np.array([[2, 1, 4],
              [1, 3, 7],
              [0, 2, 4]])

# Multiplying by I leaves A unchanged
print(np.allclose(A @ I, A))  # True
```

### Semantic Meanings in ML

- **Computing Matrix Inverses:** Augmenting a square matrix $\mathbf{A}$ with $\mathbf{I}$ and row-reducing to RREF is the standard algorithmic way to compute $\mathbf{A}^{-1}$ — the left side becomes $\mathbf{I}$ while the right side becomes the inverse.
- **Regularization (Ridge):** Adding a scaled identity matrix to a design matrix's Gram matrix, $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$, guarantees the result is full rank and invertible even when $\mathbf{X}^T\mathbf{X}$ alone is singular — this is exactly how [[L2 Regularization (Ridge) shrinks feature weights evenly without forcing them to zero|L2 Regularization]] "cures" multicollinearity.
- **Initialization and Residual Connections:** Some weight matrices are initialized at (or near) $\mathbf{I}$ so that a layer starts by passing its input through unchanged, letting the network learn a small deviation from identity rather than an arbitrary transformation (e.g., residual/skip connections).
- **Eigen-decomposition Baseline:** Every eigenvalue of $\mathbf{I}$ equals $1$, and every non-zero vector is one of its eigenvectors — a useful reference point when reasoning about how other matrices stretch or rotate space differently.

**Related notes to create/link:**
- Reversing transformations: [[Matrix Inversion requires non-zero determinants and full rank]]
- Faster way to know if a matrix is singular or not singular: [[The determinant measures how a matrix scales space and determines invertibility]]
