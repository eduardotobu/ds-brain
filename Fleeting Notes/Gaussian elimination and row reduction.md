---
id: "20260705164230"
type: permanent
subtype: algorithm
created: 2026-07-05 16:42:30
aliases: []
tags:
  - linear-algebra
family:
Up: "[[Systems of linear equations]]"
---
# Gaussian elimination systematically solves linear systems by reducing matrices to upper triangular form

**Gaussian Elimination** (also known as row reduction) is a systematic, step-by-step algorithmic procedure used to solve systems of linear equations, find the inverse of a matrix, and determine a matrix's rank.

The algorithm takes an **augmented matrix**—which merges the coefficient matrix $\mathbf{A}$ and target vector $\mathbf{b}$ from $\mathbf{Ax}=\mathbf{b}$—and applies a sequence of **Elementary Row Operations** to transform it into an upper triangular structure called **Row Echelon Form (REF)**. From there, a process called **back-substitution** is used to find the exact values of the unknown variables.

### The Three Elementary Row Operations

To reduce a matrix without altering the underlying solution of the linear system, you can only perform three valid operations on its rows:

1. **Swapping:** Switch the positions of two rows.
2. **Scaling:** Multiply all elements in a single row by a non-zero scalar.
3. **Row Addition/Subtraction:** Add or subtract a multiple of one row to another row.

One important property is that **if you perform any of these operations on a matrix it will preserve singularity**: If it's a singular matrix it will still be a singular matrix and if it's a non-singular matrix it will still be a non-singulat matrix.

### Mathematical Pipeline

Given a system packed into an augmented matrix $[\mathbf{A} | \mathbf{b}]$:

$$\begin{bmatrix} 2 & 1 & | & 5 \\ -1 & 3 & | & 8 \end{bmatrix} \xrightarrow{\text{Forward Elimination}} \begin{bmatrix} 2 & 1 & | & 5 \\ 0 & 3.5 & | & 10.5 \end{bmatrix} \quad \text{(Row Echelon Form)}$$

Once in REF, the last row easily solves for the final variable ($3.5x_2 = 10.5 \rightarrow x_2 = 3$). You then substitute $x_2$ back into the top row to solve for $x_1$.

If you continue eliminating elements _above_ the pivots until the left side becomes the [[The identity matrix acts as the multiplicative identity, leaving any vector or matrix unchanged|identity matrix]], the matrix reaches **Reduced Row Echelon Form (RREF)**, exposing the solution vector directly.

### Python Representation

While you can use `SymPy` to explicitly see the step-by-step RREF matrix, in practical machine learning pipelines, we use `SciPy` or `NumPy`. Under the hood, modern computers don't perform raw Gaussian elimination because it is computationally expensive ($O(n^3)$) and prone to floating-point rounding errors. Instead, they use **LU Decomposition** (Lower-Upper), which is a vectorized, computationally efficient cousin of Gaussian elimination.


``` python
import numpy as np
import scipy.linalg as la
from sympy import Matrix

# --- 1. Educational/Exact View (SymPy) ---
# Create an augmented matrix
augmented_matrix = Matrix([[2, 1, 5], 
                           [-1, 3, 8]])
# Compute RREF
rref_matrix, pivots = augmented_matrix.rref()
print("RREF Matrix:\n", rref_matrix)

# --- 2. Production/ML View (SciPy LU Decomposition) ---
A = np.array([[2, 1], 
              [-1, 3]])

# P = Permutation (swapping), L = Lower triangular, U = Upper triangular (REF)
P, L, U = la.lu(A)
print("\nUpper Triangular Matrix (U) from LU decomposition:\n", U)
```

### Semantic Meanings and Relevance in ML

While data scientists rarely write Gaussian elimination loops from scratch, its core rules govern how ML systems compute behind the scenes:

- **Identifying Feature Redundancy (Rank Deficiency):** If you run row reduction on a dataset's design matrix $\mathbf{X}$ and an entire row becomes zeroes, it proves that one of your features is a perfect linear combination of the others (multicollinearity). This tells you the matrix is _singular_ and cannot be inverted.
- **The Necessity of Pivoting (Numerical Stability):** If a diagonal element during row reduction is extremely close to zero, dividing by it causes massive floating-point errors. This is why deep learning libraries implement _partial pivoting_ (swapping rows to put the largest number on the diagonal), ensuring numerical stability during backpropagation.
- **Determinants and Normalizing Flows:** In advanced generative architectures like Normalizing Flows, models must compute the determinant of a matrix. Row reduction simplifies this drastically: the determinant of a matrix in Row Echelon Form is simply the product of its diagonal elements (pivots).

**Related notes to create/link:**

- The modern implementation: [[LU Decomposition factorizes matrices into lower and upper structures]]
- Dealing with singular matrices: [[Matrix Inversion requires non-zero determinants and full rank]]
- Detecting multicollinearity: [[Matrix Rank determines the number of independent features]]