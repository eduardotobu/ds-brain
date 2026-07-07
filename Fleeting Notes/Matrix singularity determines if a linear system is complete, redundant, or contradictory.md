---
id: "20260705170314"
type: permanent
subtype: concept
created: 2026-07-05 17:03:14
aliases: []
tags:
  - linear-algebra
Up: "[[Systems of linear equations]]"
---
# Matrix singularity determines if a linear system is complete, redundant, or contradictory

In linear algebra, **singularity** describes whether a matrix (or system of equations) contains enough independent, non-conflicting information to be solved uniquely. We classify linear systems into two primary categories based on the relationship between the number of equations ("sentences") and the unique pieces of information they provide.

![[file-20260705130743425.jpg]]

### 1. Non-Singular Systems (Complete Systems)

A **non-singular system** carries exactly as many unique pieces of information as there are equations. It represents a **complete system** where there are enough constraints to pin down an answer without any redundancy or conflict.

**<u>Only square matrices can be non-singular</u>**

- **Geometric Behavior:** The hyperplanes intersect at exactly one point.
- **Solutions:** It has a **unique solution**.
- **Determinant:** $\det(\mathbf{A}) \neq 0$ (the matrix can be inverted).

$$\begin{aligned} x + y &= 5 \\ x - y &= 1 \end{aligned} \implies \text{Unique solution: } (3, 2)$$

### 2. Singular Systems

A **singular system** fails to provide a unique solution because the equations lack a clean balance of independent information. Singular systems fall into two distinct sub-types:

#### A. Redundant Systems (Dependent)

In a **redundant system**, two or more equations convey the exact same mathematical information, meaning you have fewer pieces of unique information than equations.

- **Geometric Behavior:** The hyperplanes completely overlap or blend into a shared line/subspace.
- **Solutions:** It has **infinitely many solutions**.
- **Measurement:** We can measure exactly _how_ redundant a system is by analyzing its [[Matrix Rank determines the number of independent features|Matrix Rank]]. A lower rank indicates higher redundancy.

$$\begin{aligned} x + y &= 5 \\ 2x + 2y &= 10 \quad \text{(This sentence gives no new information)} \end{aligned}$$

#### B. Contradictory Systems (Inconsistent)

In a **contradictory system**, the equations mathematically conflict with one another, making it impossible for all conditions to be true at the same time.

- **Geometric Behavior:** The hyperplanes run parallel or fail to cross at any mutual point in space.
- **Solutions:** It has **no solution**.

$$\begin{aligned} x + y &= 5 \\ x + y &= 10 \quad \text{(An impossible contradiction)} \end{aligned}$$

# Singularity vs. Non-Singularity in Matrices

A fundamental concept in linear algebra is distinguishing between singular and non-singular matrices. This distinction dictates whether a matrix retains mathematical information during a transformation or loses it.

## Comparative Table

Below is the comparative table covering the algebraic and geometric properties of both matrices:

| Property | Singular Matrix | Non-Singular Matrix |
| :--- | :--- | :--- |
| **Determinant** | Equal to zero ($\vert A \vert = 0$) | Not equal to zero ($\vert A \vert \neq 0$) |
| **Inverse** | Does not have an inverse matrix | Has an inverse matrix ($A^{-1}$ exists) |
| **Rank** | Less than the size of the matrix (rank-deficient) | Equal to the size of the matrix (full rank) |
| **Rows and Columns** | Are linearly dependent (one is a linear combination of another) | Are linearly independent |
| **System of Equations** | Has infinite solutions or no solution | Guaranteed to have a unique solution |
| **Eigenvalues** | At least one eigenvalue is zero | No eigenvalue is zero |

## The Geometric Concept

A square matrix represents a linear transformation. When you apply a **non-singular matrix** (whose determinant is $\neq 0$), the mathematical space maintains all its original dimensions. Because coordinate information is not lost during this transformation, there is always an exact way back, which explains why the inverse matrix ($A^{-1}$) always exists.

In contrast, a **singular matrix** (determinant equal to $0$) literally "collapses" the mathematical space into a lower dimension. For example, it transforms a whole three-dimensional (3D) space and squashes it flat, leaving it as a plane (2D) or a simple straight line (1D). Since multiple starting points end up squashed into the same destination, it is impossible to know where each one came from. Since there is no path back, it is mathematically impossible to calculate an inverse matrix.

### Python Representation

In NumPy, attempting to solve a singular system (whether redundant or contradictory) using `np.linalg.solve()` will throw a `LinAlgError` because a singular matrix cannot be inverted. We can screen for singularity by checking if the matrix determinant is zero.

``` python
import numpy as np

# --- 1. Non-Singular (Complete) ---
A_complete = np.array([[1, 1], [1, -1]])
print("Complete Matrix Det:", np.linalg.det(A_complete)) # Output: -2.0 (Non-zero)

# --- 2. Singular (Redundant or Contradictory Coefficient Matrix) ---
A_singular = np.array([[1, 1], [2, 2]])
print("Singular Matrix Det:", np.linalg.det(A_singular)) # Output: 0.0

# Attempting to solve a singular system causes a crash
try:
    b = np.array([5, 10])
    np.linalg.solve(A_singular, b)
except np.linalg.LinAlgError as e:
    print(f"Error caught: {e}") # Output: Singular matrix
```

### Semantic Meanings in ML

Recognizing singularity is highly critical when training models:

- **Multicollinearity in Datasets:** If your dataset contains redundant features (e.g., a house's size listed in both `square_feet` and `square_meters`), your design matrix becomes singular. This completely breaks classical closed-form algorithms like Ordinary Least Squares (OLS) Linear Regression because you cannot mathematically compute $(\mathbf{X}^T\mathbf{X})^{-1}$.
- **Underdetermined Problems (High-Dimensional Data):** When you have more features than data samples ($n > m$), like in genomic sequencing or medical imaging, your linear system becomes structurally redundant (singular) with infinite possible parameter combinations. Models must utilize regularization techniques (L1/L2) to force a single, stable solution.

**Related notes to create/link:**

- Measuring information content: [[Matrix Rank determines the number of independent features]]
- When inversion breaks down: [[Matrix Inversion requires non-zero determinants and full rank]]
- Resolving singular systems practically: [[Regularization prevents overfitting in singular data spaces]]
- Linear dependence and independence: [[Linear independence ensures vectors provide non-redundant information in a vector space]]
- Faster way to know if a matrix is singular or not singular: [[The determinant measures how a matrix scales space and determines invertibility]]