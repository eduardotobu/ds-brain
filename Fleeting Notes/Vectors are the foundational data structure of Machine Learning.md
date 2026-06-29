---
id: "202606291151"
type: permanent
subtype: concept
created: 2026-06-29
aliases:
  - vectors
tags:
  - linear-algebra
Up: "[[Linear Algebra for Machine Learning]]"
---
# Vectors are the foundational data structure of Machine Learning

In physics, a vector represents magnitude and direction, but in computer science and machine learning, a **vector is simply an ordered, one-dimensional list of numbers (scalars)**.

Because algorithms cannot understand raw text, images, or concepts, we use vectors to translate the real world into a mathematical format that models can process. In this context, an $n$-dimensional vector represents a single data point plotted in an $n$-dimensional space, where $n$ is the number of features.

### Mathematical Representation

Mathematically, a vector is typically denoted by a bold lowercase letter and represented as a column of values:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

### Python Representation

In machine learning workflows, vectors are almost exclusively handled using **NumPy** arrays or framework-specific tensors (like PyTorch or TensorFlow). A standard Python list can hold the numbers, but a NumPy array provides the underlying C-based mathematical operations required for linear algebra.

``` python
import numpy as np

# Representing a 3-dimensional vector (e.g., [x_1, x_2, x_3])
x = np.array([1.5, 2.0, 5.1])

# Checking the dimensionality (returns 1 for a 1D vector)
print(x.ndim) 

# Checking the shape (returns (3,) meaning 3 elements in the 1st dimension)
print(x.shape)
```

### Semantic Meanings in ML

Depending on where the vector is used in a model, it takes on different semantic meanings:

- When representing the input data (e.g., the square footage, age, and price of a house), it is a [[Feature Vector]].
- When representing the learned parameters of a model, it is a [[Weight Vector]].
- When representing the semantic meaning of text or complex data, it is called an [[Embedding]].

**Related notes to create/link:**
- Vectors are a 1st-order tensor: [[Tensors are multi-dimensional arrays that generalize vectors and matrices]]
- Combining vectors creates a matrix: [[Matrices are collections of vectors]]
- How vectors are compared for similarity: [[Dot Product calculates vector similarity]]
- Array operations in Python: [[NumPy Vectorization speeds up computation]]