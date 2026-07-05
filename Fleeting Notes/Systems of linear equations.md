---
id: "20260705001301"
type: permanent
subtype: concept
created: 2026-07-05 00:13:01
aliases: []
tags:
  - linear-algebra
Up: "[[Linear Algebra for Machine Learning]]"
---

# Systems of linear equations

At its historical and practical core, linear algebra was developed to systematically solve **systems of linear equations**—sets of multiple equations with multiple unknown variables that must be satisfied simultaneously.

A system of equations is not just an algebraic puzzle to solve by substituting variables; it is an invitation to shift our thinking from isolated numbers to vector spaces and geometric transformations.

### Mathematical Representation

A system of $m$ linear equations with $n$ unknowns ($x_1, x_2, \dots, x_n$) is written algebraically as:

$$\begin{matrix} a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n = b_m \end{matrix}$$

Linear algebra allows us to compress this messy list of equations into a clean, single matrix equation:

$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

Where $\mathbf{A}$ is the $m \times n$ matrix of coefficients, $\mathbf{x}$ is the $n$-dimensional vector of unknowns, and $\mathbf{b}$ is the $m$-dimensional vector of constants.
More detail on: [[Matrix form of linear equations]]

### The Two Geometric Views

To truly understand a system of equations, we look at it through two distinct geometric perspectives:
[[Geometric interpretation of linear systems]]

### Matrix Singularity

[[Matrix singularity determines if a linear system is complete, redundant, or contradictory]]


### Python Representation

In machine learning and data science, we rarely solve these systems by hand using row reduction (Gaussian elimination). Instead, we offload the computation to optimized numerical libraries like NumPy.

``` python
import numpy as np

# Solving the system:
#  2x + 1y = 8
#  1x + 3y = 9

# Coefficient Matrix A
A = np.array([
    [2, 1],
    [1, 3]
])

# Constant Vector b
b = np.array([8, 9])

# Solving for x and y (Ax = b)
x = np.linalg.solve(A, b)

print(f"Solution vector [x, y]: {x}") 
# Output will be [3. 2.] because 2(3)+2=8 and 3+3(2)=9
```

### Why This Matters in Machine Learning

Every time a model learns, it is essentially trying to solve or approximate a massive system of equations.

- **Linear Regression:** Finding the best-fitting line through a dataset is mathematically equivalent to solving an overdetermined system of equations ($\mathbf{A}\mathbf{x} = \mathbf{b}$) where there are more equations (data points) than unknowns (features). Since a perfect intersection doesn't exist, we look for the closest possible vector.
- **Optimization:** Neural network training relies on moving weights in a direction that satisfies a complex array of mathematical constraints.


**Related notes to create/link:**

- Combining vectors geometrically: [[Linear Combinations span vector spaces]]
- When a system cannot be perfectly solved: [[The Normal Equation solves overdetermined systems in Linear Regression]]


