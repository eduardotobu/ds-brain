---
id: "20260719173824"
type: permanent
subtype: concept
created: 2026-07-19 17:38:24
aliases:
  - vector multiplication
tags:
  - linear-algebra
Up: "[[Vectors are the foundational data structure of Machine Learning]]"
---
# Scalar multiplication scales a vector's magnitude without altering its orientation along the line of action.

Scalar multiplication is a fundamental operation in linear algebra where a vector $\mathbf{v}$ is scaled by a real number $c$ (the scalar). Geometrically, this operation stretches or shrinks the vector by a factor of $\vert{}c\vert{}$. If $c > 0$, the vector maintains its direction; if $c < 0$, the vector's direction is reversed. If $c = 0$, the result is the zero vector $\mathbf{0}$. This operation defines the structure of a vector space by allowing vectors to be resized while preserving the underlying line of action.

### Mathematical Representation

Given a scalar $c \in \mathbb{R}$ and a vector $\mathbf{v} \in \mathbb{R}^n$, the product $c\mathbf{v}$ is computed by multiplying each component of $\mathbf{v}$ by $c$:

$$c\mathbf{v} = c\begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

Scalar multiplication satisfies the following axioms for all scalars $c, d$ and vectors $\mathbf{u}, \mathbf{v}$:

- Distributivity over vector addition: $c(\mathbf{u}+\mathbf{v}) = c\mathbf{u} + c\mathbf{v}$
- Distributivity over scalar addition: $(c+d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$
- Compatibility of scalar multiplication: $c(d\mathbf{v}) = (cd)\mathbf{v}$
- Identity element: $1\mathbf{v} = \mathbf{v}$

### Python Representation

``` python
import numpy as np

# Initialize a vector
v = np.array([2, -4])
scalar = 2.5

# Perform scalar multiplication
scaled_v = scalar * v

print(f"Original Vector: {v}")      # Output: [ 2 -4]
print(f"Scaled Vector: {scaled_v}") # Output: [ 5. -10.]

# Example of direction reversal with negative scalar
reversed_v = -1.0 * v
print(f"Reversed Vector: {reversed_v}") # Output: [-2.  4.]
```

### Semantic Meanings in ML

- **Gradient Descent Learning Rate:** In optimization, the update rule $\mathbf{w}_{new} = \mathbf{w}_{old} - \eta \nabla L$ uses scalar multiplication to scale the gradient vector $\nabla L$ by the learning rate $\eta$, controlling the step size of parameter updates.
- **Neural Network Weights:** During backpropagation, error signals are propagated back through layers, often involving scalar multiplication of the activation derivatives, which scales the magnitude of the weight adjustments.
- **Feature Normalization:** When scaling features to a specific range (e.g., $[0, 1]$ or standardized units), each feature vector is multiplied by a scalar inverse of the range or standard deviation.
- **Attention Mechanisms:** In transformer architectures, the "scaled" dot-product attention uses scalar multiplication to multiply the dot product of queries and keys by $1/\sqrt{d_k}$ to prevent gradient explosion.

### Related notes to create/link
