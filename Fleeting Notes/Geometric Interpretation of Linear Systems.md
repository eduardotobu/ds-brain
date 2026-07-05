---
id: "20260705003128"
type: permanent
subtype: concept
created: 2026-07-05 00:31:28
aliases: []
tags:
  - linear-algebra
Up: "[[Linear Algebra for Machine Learning]]"
---

# Geometric Interpretation of Linear Systems

A system of linear equations (commonly written as $\mathbf{Ax} = \mathbf{b}$) is not just an algebraic puzzle; it has a profound geometric meaning. Understanding this geometry is essential for grasping how optimization algorithms find solutions and how data space transformations operate in machine learning.

We can view a linear system through two distinct geometric lenses: the **Row Picture** and the **Column Picture**.

### 1. The Row Picture: Intersecting Hyperplanes

In the row picture, each individual equation in the system represents a geometric constraint.

- In a 2D space, an equation represents a **line**.
- In a 3D space, an equation represents a **plane**.
- In an $n$-dimensional space, an equation represents an $(n-1)$-dimensional **hyperplane**.

Solving the system via the row picture means searching for the exact coordinate in space where all these hyperplanes intersect.

- **Unique Solution:** The hyperplanes cross at a single, distinct point.
- **No Solution:** The hyperplanes are parallel or do not share a single mutual intersection point.
- **Infinite Solutions:** The hyperplanes overlap perfectly along a shared line or subspace.

### 2. The Column Picture: Linear Combinations of Vectors

The column picture shifts the focus from drawing boundaries to **stretching and combining vectors**. Instead of looking at equations, it looks at the matrix $\mathbf{A}$ as a collection of column vectors. It asks: _Can we scale these column vectors by certain amounts ($\mathbf{x}$) so that when we add them together, they land exactly on the target vector $\mathbf{b}$?_

Mathematically, it represents a linear combination:

$$x_1 \begin{bmatrix} a_{1,1} \\ a_{2,1} \end{bmatrix} + x_2 \begin{bmatrix} a_{1,2} \\ a_{2,2} \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

This view is highly favored in machine learning because it introduces the concept of **vector spaces and spans**, which dictate whether a model has the mathematical capacity to represent or fit a given dataset.

### Python Representation

In code, solving for the vector $\mathbf{x}$ maps to finding the intersection point or the required vector coefficients using NumPy's linear algebra submodule (`np.linalg`).

``` python 
import numpy as np

# Representing the system: 
#  2x +  y = 5
# -1x + 3y = 8
A = np.array([[2, 1], 
              [-1, 3]])
b = np.array([5, 8])

# Solving for x (finds the coordinate point / scalar coefficients)
x = np.linalg.solve(A, b)

print(f"Solution vector x: {x}")
# Returns the exact weights needed to combine columns of A to reach b
```

### Semantic Meanings in ML

- **Decision Boundaries:** Linear classification algorithms like Support Vector Machines (SVMs) or Logistic Regression explicitly calculate a hyperplane (the Row Picture) to geometrically separate different classes of data points in high-dimensional space.
- **Dimensionality Reduction:** Techniques like Principal Component Analysis (PCA) rely heavily on the Column Picture to find the optimal lower-dimensional subspace (span) to project high-dimensional feature vectors onto without losing critical variance.

**Related notes to create/link:**

- Stacking vectors into matrices: [[Matrices are collections of vectors]]
- Understanding vector reach: [[The span of vectors defines the reach of a vector space]]
- Overdetermined systems in ML: [[Linear Least Squares solves unsolvable linear systems]]