---
id: "20260705115336"
type: permanent
subtype: concept
created: 2026-07-05 11:53:36
aliases: []
tags:
  - linear-algebra
Up: "[[Systems of linear equations]]"
---
# Matrix form of linear equations

Instead of writing out a massive system of linear equations line by line, linear algebra allows us to pack an entire system into a single, elegant expression: **$\mathbf{Ax} = \mathbf{b}$**.

Converting an algebraic system of equations into its matrix form is the primary step required to feed data into machine learning algorithms. It shifts our perspective from isolated equations to entire transformations, allowing modern computing hardware (like GPUs) to perform massive parallel computations.

### Mathematical Representation

A system of $m$ linear equations with $n$ unknowns can be completely decomposed into three distinct mathematical structures:

1. **The Coefficient Matrix ($\mathbf{A}$):** An $m \times n$ matrix holding all the scalar weights/coefficients that multiply the variables.
2. **The Variable Vector ($\mathbf{x}$):** An $n$-dimensional column vector holding the unknowns we wish to solve for.
3. **The Constant Vector ($\mathbf{b}$):** An $m$-dimensional column vector holding the output targets.

**Algebraic System:**

$$\begin{aligned} a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n &= b_1 \\ a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n &= b_2 \\ &\vdots \\ a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n &= b_m \end{aligned}$$

**Matrix Form ($\mathbf{Ax} = \mathbf{b}$):**

$$\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$$

### Python Representation

In NumPy, we represent the coefficient matrix $\mathbf{A}$ as a 2D array and the vectors as 1D or 2D column arrays. We can evaluate the equation using the matrix multiplication operator (`@`).


``` python
import numpy as np

# Representing the system:
# 3x + 2y = 7
# 1x - 4y = -7

# Coefficient Matrix A (2x2)
A = np.array([
    [3,  2],
    [1, -4]
])

# Supposing we already solved the system and found x = [1, 2]
x = np.array([1, 2])

# Performing matrix-vector multiplication (Ax) to find b
b = A @ x

print(f"Target vector b: {b}")  # Output: [7, -7]
```

### Semantic Meanings in ML

The structure of $\mathbf{Ax} = \mathbf{b}$ (and its transposed equivalent $\mathbf{Xw} = \mathbf{y}$) underpins almost all core machine learning models:

- **Linear Regression Predictions:** When a model generates a prediction, it computes $\mathbf{\hat{y}} = \mathbf{Xw}$, where $\mathbf{X}$ is the matrix of dataset features (the design matrix), $\mathbf{w}$ is the vector of learned weights, and $\mathbf{\hat{y}}$ represents the target predictions.
- **Neural Network Layers:** A fully connected neural network layer transforms input activations ($\mathbf{a}$) into the next layer's raw inputs using a weight matrix ($\mathbf{W}$) and a bias scalar vector ($\mathbf{b}$): $\mathbf{z} = \mathbf{Wa} + \mathbf{b}$.
- **Data Transformations:** In computer vision or embeddings, multiplying a matrix by a vector acts as a geometric function that scales, rotates, or projects the data vector into an entirely new feature space.