---
id: "202606291403"
type: permanent
subtype: concept
created: 2026-06-29
aliases: []
tags:
  - linear-algebra
Up:
---
# Tensors are multi-dimensional arrays that generalize vectors and matrices

While vectors and matrices are specific, fixed-dimension structures, a **tensor is the generalized mathematical concept for an $n$-dimensional array of numbers**.

In the context of machine learning, you can think of a tensor as a container that can house data in any number of dimensions (often called _axes_ or _ranks_). In fact, the structures we already use are just specific types of tensors:

- A **Scalar** (a single number) is a 0th-order tensor.
- A [[Vectors are the foundational data structure of Machine Learning|Vector]] is a 1st-order tensor.
- A [[Matrices are collections of vectors representing tabular data or weights|Matrix]] is a 2nd-order tensor.
- Arrays with 3 or more dimensions are simply referred to as $n$-order tensors.

### Mathematical Representation

Mathematically, tensors with 3 or more dimensions are often denoted by bold typeface or sans-serif letters, like $\boldsymbol{\mathsf{T}}$. Because they exist in three or more dimensions, writing them out on flat paper is difficult, so they are usually defined by how you index their individual elements.

For a 3rd-order tensor $\boldsymbol{\mathsf{T}}$, an individual scalar value inside it would be accessed using three indices (e.g., row, column, and depth):

$$t_{i,j,k}$$

### Python Representation

This concept is so foundational that the most popular deep learning frameworks are named after it (TensorFlow, PyTorch tensors). In code, a tensor is just an array nested within other arrays.

``` python
import numpy as np

# Representing a 3rd-order tensor (e.g., 2 matrices stacked together)
T = np.array([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])

# Checking the dimensionality (returns 3 for a 3D tensor)
print(T.ndim) 

# Checking the shape (returns (2, 2, 3): 2 depths, 2 rows, 3 columns)
print(T.shape)
```

### Semantic Meanings in ML

Higher-order tensors are essential for deep learning and dealing with unstructured data. Common applications include:

- **Color Images (3D Tensors):** While a grayscale image is a 2D matrix, a color image adds a third dimension for the color channels (Red, Green, Blue). An image's shape is typically `(height, width, 3)`.
- **Video (4D Tensors):** Video data adds the element of time. It is represented as a sequence of color images: `(frames, height, width, 3)`.
- **Batches of Data:** In training neural networks, we rarely feed one item at a time. We feed data in batches. A batch of 32 color images becomes a 4D tensor with the shape `(32, height, width, 3)`.

**Related notes to create/link:**

- Deep learning frameworks: [[PyTorch and TensorFlow handle tensor gradients automatically]]
- Reshaping data: [[Tensor reshaping modifies dimensions without altering data]]
- Operations across axes: [[Broadcasting aligns tensors of different shapes for arithmetic]]