---
id: "20260719174159"
type: permanent
subtype: concept
created: 2026-07-19 17:41:59
aliases:
  - element-wise multiplication
tags:
  - linear-algebra
Up: "[[Vectors are the foundational data structure of Machine Learning]]"
---
# The Hadamard product for vectors computes their element-wise multiplication to scale individual components independently.

The Hadamard product for vectors is a binary operation that takes two vectors of identical dimensions and produces a third vector of the same dimension by multiplying their corresponding components. Unlike the inner (dot) product, which collapses vectors into a single scalar representing their projection and angle, the Hadamard product preserves the dimensionality of the vector space. Geometrically, it acts as an anisotropic scaling operation; one vector acts as a set of independent scaling factors applied to the standard basis directions of the other vector.
In practice, rather than interpreting it as a spatial shape, it is best viewed geometrically as a **masking or filtering tool**.
### Mathematical Representation

Given two column vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, the Hadamard product (often denoted by $\circ$ or $\odot$) is defined mathematically as:

$$\mathbf{u} \circ \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{bmatrix} \circ \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} u_1 v_1 \\ u_2 v_2 \\ \vdots \\ u_n v_n \end{bmatrix}$$

The resulting vector $\mathbf{w} = \mathbf{u} \circ \mathbf{v}$ has components $w_i = u_i v_i$. The operation inherits several properties from scalar multiplication, including commutativity ($\mathbf{u} \circ \mathbf{v} = \mathbf{v} \circ \mathbf{u}$), associativity ($(\mathbf{u} \circ \mathbf{v}) \circ \mathbf{w} = \mathbf{u} \circ (\mathbf{v} \circ \mathbf{w})$), and distributivity over vector addition ($\mathbf{u} \circ (\mathbf{v} + \mathbf{w}) = \mathbf{u} \circ \mathbf{v} + \mathbf{u} \circ \mathbf{w}$).

### Python Representation

``` python
import numpy as np

# Initialize two 1D arrays (vectors) of the same size
u = np.array([2, -3, 4])
v = np.array([5, 2, -1])

# The Hadamard product for vectors in NumPy uses the standard * operator
hadamard_result = u * v

print(f"Vector u: {u}")
print(f"Vector v: {v}")
print(f"Hadamard Product (u ∘ v): {hadamard_result}") 
# Output:
# Vector u: [ 2 -3  4]
# Vector v: [ 5  2 -1]
# Hadamard Product (u ∘ v): [ 10  -6  -4]
```

### Semantic Meanings in ML

- **Feature Weighting and Attention:** The Hadamard product is used to apply learned attention weights or feature importance masks to a feature vector. Multiplying a feature vector by a weight vector between $[0, 1]$ effectively "pays attention" to or "ignores" specific features.
- **Adaptive Learning Rate Optimizers:** In optimization algorithms like Adam or RMSProp, the Hadamard product is used extensively. The update step relies on element-wise multiplication and division (e.g., dividing the gradient vector element-wise by the square root of the moving average of squared gradients) to adapt the learning rate for each parameter individually.
- **Dropout Regularization:** During neural network training, dropout is implemented by taking the Hadamard product of the layer's activation vector and a binary mask vector generated from a Bernoulli distribution. This randomly zeros out components to prevent co-adaptation.
- **Gating in Vector Spaces:** In architectures processing sequential data (like GRUs), gating vectors determine how much of the previous hidden state vector to retain, applying the Hadamard product between a sigmoid-activated gate vector and the hidden state vector.