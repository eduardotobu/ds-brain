---
id: "20260719172950"
type: permanent
subtype: concept
created: 2026-07-19 17:29:50
aliases:
  - vector sum
tags:
  - linear-algebra
Up: "[[Vectors are the foundational data structure of Machine Learning]]"
---
# Vector Addition and Subtraction define the fundamental arithmetic of linear spaces

Vector addition and subtraction represent the most basic geometric and algebraic operations in a vector space. Geometrically, addition corresponds to the concatenation of displacement vectors, often visualized via the parallelogram rule, where the sum $\mathbf{u}+\mathbf{v}$ is the diagonal of the parallelogram formed by $\mathbf{u}$ and $\mathbf{v}$. Subtraction represents the difference between two points, resulting in a vector that points from the terminal end of the subtrahend to the terminal end of the minuend.

### Mathematical Representation
#### Vector Addition

Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ be two column vectors. Addition is defined component-wise as:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$
![[file-20260719173344995.jpg|256]]

#### Vector Substraction

Vector subtraction is defined as the addition of the negative vector, where $-\mathbf{v}$ is a reflection of $\mathbf{v}$ across the origin:

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v}) = \begin{bmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{bmatrix}$$

![[file-20260719173630147.jpg|262]]


These operations are closed under the field of scalars, satisfying commutativity ($\mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}$) and associativity ($\mathbf{u}+(\mathbf{v}+\mathbf{w})=(\mathbf{u}+\mathbf{v})+\mathbf{w}$).

### Python Representation

``` python
import numpy as np

# Initialize vectors
u = np.array([3, 5])
v = np.array([1, -2])

# Addition
addition_result = u + v
# Subtraction
subtraction_result = u - v

print(f"Vector Addition: {addition_result}")       # Output: [4 3]
print(f"Vector Subtraction: {subtraction_result}") # Output: [2 7]
```

### Semantic Meanings in ML

- **Embedding Space Semantics:** In models like Word2Vec, vector addition captures relationships (e.g., $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$), allowing semantic reasoning through linear algebra.
- **Optimization and Residuals:** In gradient descent and backpropagation, the update rule $\mathbf{w}_{new} = \mathbf{w}_{old} - \eta \nabla L$ is a direct application of vector subtraction, shifting weights in the direction of the negative gradient to minimize loss.
- **Data Normalization:** Centering data around the origin requires subtracting the mean vector $\boldsymbol{\mu}$ from each feature vector $\mathbf{x}$, represented as $\mathbf{x}' = \mathbf{x} - \boldsymbol{\mu}$.
- **Neural Network Layers:** Fully connected layers represent a linear transformation followed by an addition of a bias vector $\mathbf{b}$, expressed as $f(\mathbf{x}) = \mathbf{Wx} + \mathbf{b}$, highlighting addition's role in shifting hyperplanes.

### Related notes to create/link

- [[Linear Combinations and Span]]
- [[Basis and Dimension]]
- [[Dot Product and Orthogonality]]
- [[Matrix-Vector Multiplication]]