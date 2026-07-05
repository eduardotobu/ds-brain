---
id: "20260705003128"
type: permanent
subtype: concept
created: 2026-07-05 00:31:28
aliases: []
tags:
  - linear-algebra
Up: "[[Systems of linear equations]]"
---

# Geometric interpretation of linear systems

We can view a linear system through two distinct geometric lenses: the **Row Picture** and the **Column Picture**.

### 1. The Row Picture: Intersecting Hyperplanes

In the row picture, each individual equation in the system represents a geometric constraint.

- In a 2D space, an equation represents a **line**.
- In a 3D space, an equation represents a **plane**.
- In an $n$-dimensional space, an equation represents an $(n-1)$-dimensional **hyperplane**.

Solving the system via the row picture means searching for the exact coordinate in space where all these hyperplanes intersect.
- **Unique Solution (Complete System):** The hyperplanes cross at a single, distinct point.
- **Infinite Solutions (Redundant System):** The hyperplanes overlap perfectly along a shared line or subspace.
- **No Solution (Contradictory System):** The hyperplanes are parallel or do not share a single mutual intersection point.

![[file-20260705130743425.jpg]]

If we convert the constants $b$ of the linear system to 0, we can see that a contradictory system becomes a redundant system that pass through origin, so this is what generalizes them as singular.

### 2. The Column Picture: Linear Combinations of Vectors

The column picture shifts the focus from drawing boundaries to **stretching and combining vectors**. Instead of looking at equations, it looks at the matrix $\mathbf{A}$ as a collection of column vectors. It asks: _Can we scale these column vectors by certain amounts ($\mathbf{x}$) so that when we add them together, they land exactly on the target vector $\mathbf{b}$?_

Mathematically, it represents a linear combination:

$$x_1 \begin{bmatrix} a_{1,1} \\ a_{2,1} \end{bmatrix} + x_2 \begin{bmatrix} a_{1,2} \\ a_{2,2} \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

This view is highly favored in machine learning because it introduces the concept of **vector spaces and spans**, which dictate whether a model has the mathematical capacity to represent or fit a given dataset.

### Semantic Meanings in ML

- **Decision Boundaries:** Linear classification algorithms like Support Vector Machines (SVMs) or Logistic Regression explicitly calculate a hyperplane (the Row Picture) to geometrically separate different classes of data points in high-dimensional space.
- **Dimensionality Reduction:** Techniques like Principal Component Analysis (PCA) rely heavily on the Column Picture to find the optimal lower-dimensional subspace (span) to project high-dimensional feature vectors onto without losing critical variance.