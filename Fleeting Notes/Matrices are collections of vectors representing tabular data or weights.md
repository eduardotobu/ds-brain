---
id: "202606291244"
type: permanent
subtype: concept
created: 2026-06-29
aliases:
  - matrices
tags:
  - linear-algebra
Up: "[[Linear Algebra for Machine Learning]]"
---
# Matrices are collections of vectors representing tabular data or weights

If a vector is a single row or column of numbers, a **matrix is a two-dimensional grid of numbers**. You can think of a matrix as multiple [[Vectors are the foundational data structure of Machine Learning|vectors]] stacked together. Because it has two axes (rows and columns), a matrix is classified as a 2nd-order [[Tensors are multi-dimensional arrays that generalize vectors and matrices|tensor]].

In machine learning, matrices are the primary structure used to store entire datasets and the complex network of weights inside algorithms.

### Mathematical Representation

Mathematically, matrices are usually denoted by bold uppercase letters (e.g., $\mathbf{A}$, $\mathbf{W}$). A matrix with $m$ rows and $n$ columns is said to have a shape of $m \times n$.

$$\mathbf{A} = \begin{bmatrix} a_{1,1} & a_{1,2} & \dots & a_{1,n} \\ a_{2,1} & a_{2,2} & \dots & a_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m,1} & a_{m,2} & \dots & a_{m,n} \end{bmatrix}$$

Each individual scalar inside the matrix is identified by its row and column index (e.g., $a_{2,1}$ is the element in the second row, first column).

### Python Representation

Just like vectors, matrices are handled using 2D NumPy arrays or framework tensors.

``` python
import numpy as np

# Representing a 2x3 matrix (2 rows, 3 columns)
A = np.array([
    [1.0, 2.5, 3.1],
    [4.2, 5.0, 6.8]
])

# Checking the dimensionality (returns 2 for a 2D matrix)
print(A.ndim) 

# Checking the shape (returns (2, 3))
print(A.shape)
```

### Semantic Meanings in ML

Depending on its place in the pipeline, a matrix usually represents one of the following:

- **The Design Matrix (Dataset):** Often denoted as $\mathbf{X}$, this is your entire tabular dataset. Each row is an individual [[Feature Vector]] (a single data point, like one house), and each column represents a specific feature across all data points.
- **Weight Matrices:** In neural networks, the connections between neurons in one layer and the next are stored in a weight matrix, often denoted as $\mathbf{W}$. Processing data through a neural network heavily relies on [[Matrix Multiplication]].
- **Images:** A single grayscale image is processed by computer vision models as a 2D matrix, where each number represents the intensity of a specific pixel.

**Related notes to create/link:**

- Multiplying matrices: [[Matrix Multiplication transforms vector spaces]]
- Flipping rows and columns: [[Matrix Transposition swaps axes]]
- Adding dimensions: [[Color images are 3D Tensors]]