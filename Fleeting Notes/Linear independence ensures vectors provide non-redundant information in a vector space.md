---
id: "20260705180606"
type: permanent
subtype: concept
created: 2026-07-05 18:06:06
aliases: []
tags:
  - linear-algebra
Up: "[[Matrix singularity determines if a linear system is complete, redundant, or contradictory]]"
---
# Linear independence ensures vectors provide non-redundant information in a vector space

A set of vectors is either **linearly independent** or **linearly dependent**. This concept tells us whether adding a new vector to a group actually expands our geometric reach (span) or if it just introduces redundant information.

- **Linearly Independent:** No vector in the set can be created by combining, stretching, or shrinking the other vectors in the set. Each vector points in an entirely unique directional dimension.
- **Linearly Dependent:** At least one vector in the set is redundant. It can be perfectly reconstructed as a linear combination of the remaining vectors.

### Mathematical Representation

Formally, a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$ is **linearly independent** if and only if the vector equation:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n = \mathbf{0}$$

can _only_ be satisfied by setting all the scalar coefficients to zero ($c_1 = c_2 = \dots = c_n = 0$). This is called the **trivial solution**.

If there is _any_ way to satisfy the equation where at least one coefficient ($c_i$) is not zero, the vectors are **linearly dependent**. For example, if $2\mathbf{v}_1 + 3\mathbf{v}_2 = \mathbf{v}_3$, then $\mathbf{v}_3$ depends completely on $\mathbf{v}_1$ and $\mathbf{v}_2$, making the set dependent.

### Python Representation

To programmatically check if a set of vectors is linearly independent, stack them into a matrix and compute the matrix rank using NumPy. If the rank equals the number of vectors, they are linearly independent. If the rank is less than the number of vectors, dependency exists.

``` python
import numpy as np

# Define three 3D vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([5, 7, 9]) # Notice: v3 = v1 + v2 (Linearly Dependent!)

# Stack them horizontally to form a matrix
A = np.column_stack((v1, v2, v3))

# Calculate the rank (number of linearly independent columns)
rank = np.linalg.matrix_rank(A)
num_vectors = A.shape[1]

print(f"Number of vectors: {num_vectors}")
print(f"Matrix Rank: {rank}")

if rank == num_vectors:
    print("The vectors are Linearly Independent.")
else:
    print("The vectors are Linearly Dependent (Redundant information exists).")
# Output: Matrix Rank: 2, meaning one vector is completely redundant.
```

### Semantic Meanings in ML

- **The Curser of Dimensionality & Redundant Features:** If you are training a model to predict house prices, and your feature matrix $\mathbf{X}$ has columns for `size_in_square_feet` and `size_in_square_meters`, these columns are linearly dependent ($v_2 = c \cdot v_1$). This redundancy adds zero predictive power while wasting computation time and memory.
- **Multicollinearity and Model Failure:** When features are highly dependent, the closed-form calculation for Linear Regression—$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$—fails because the matrix $\mathbf{X}^T\mathbf{X}$ becomes [[Matrix singularity determines if a linear system is complete, redundant, or contradictory|singular]] (non-invertible).
- **Dimensionality Reduction (PCA):** Principal Component Analysis works by taking a highly dependent set of data columns and projecting them onto a new set of orthogonal (perpendicular) axes that are guaranteed to be completely **linearly independent**, squeezing out the statistical noise.

**Related notes to create/link:**

- Measuring independent dimensions: [[Matrix Rank determines the number of independent features]]
- Geometric coverage of vectors: [[The span of vectors defines the reach of a vector space]]
- Forcing independence in models: [[Principal Component Analysis removes feature dependence]]