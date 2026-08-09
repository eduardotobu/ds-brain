---
id: "202606291150"
type: moc
created: 2026-06-29
modified: 2026-08-09
tags:
  - moc
  - linear-algebra
  - machine-learning
---
# Linear Algebra for Machine Learning — Map of Content

## What is this about?
Linear algebra is the mathematical backbone of machine learning: data becomes vectors and matrices, models become linear transformations, and training becomes solving or approximating systems of equations. This MOC tracks the path from raw building blocks (vectors, matrices, tensors) up through systems of equations, determinants, vector spaces, linear transformations, eigen-theory, orthogonality, and the ML-specific tools (norms, matrix calculus, tensor algebra, factorizations) built on top of them.

The numbered headings (0, 100, 200 …) form a fixed curriculum spine borrowed from a structured LA course — headings without a linked note are deliberate placeholders for topics not yet written up.

## Start here
1. [[Tensors are multi-dimensional arrays that generalize vectors and matrices]] — the umbrella concept: scalars, vectors, and matrices are all tensors of different order.
2. [[Vectors are the foundational data structure of Machine Learning]] — how data becomes math.
3. [[Matrices are collections of vectors representing tabular data or weights]] — how datasets and weights are stored.
4. [[Systems of linear equations]] — why linear algebra exists in the first place.

## 0. Building Blocks

### Tensors, vectors, matrices
1. [[Tensors are multi-dimensional arrays that generalize vectors and matrices]]
	1. [[Vectors are the foundational data structure of Machine Learning]]
	2. [[Matrices are collections of vectors representing tabular data or weights]]

### Vector operations
- [[Vector Addition and Subtraction define the fundamental arithmetic of linear spaces]]
- [[Scalar multiplication scales a vector's magnitude without altering its orientation along the line of action]]
- [[The Hadamard product for vectors computes their element-wise multiplication to scale individual components independently]]
- [[Unit vectors and angles isolate a vector's direction from its magnitude]]
- [[The dot product computes a scalar representing the projection of one vector onto another and their degree of alignment]]

### Matrix operations
- [[The matrix transpose operation reflects a matrix over its main diagonal to exchange row and column indices]]
- [[Matrix multiplication combines two linear transformations into a single transformation]]
- [[Matrix-vector multiplication represents a linear transformation mapping a vector from one space to another]]
- [[The identity matrix acts as the multiplicative identity, leaving any vector or matrix unchanged]]

## 1. Matrix Mechanics & Systems of Equations

1. [[Systems of linear equations]]
	1. [[Geometric interpretation of linear systems]]
	2. [[Matrix form of linear equations]]
	3. [[Matrix singularity determines if a linear system is complete, redundant, or contradictory]]
	4. [[Linear independence ensures vectors provide non-redundant information in a vector space]]
	5. [[Gaussian elimination and row reduction]]
	6. [[Matrix rank quantifies the number of independent dimensions in a data space]]
	7. [[Row Echelon Form and RREF expose the structural foundation of a matrix]]

### 110 Matrix Arithmetic Operations
- [[Matrix multiplication combines two linear transformations into a single transformation]]
### 111 Matrix Multiplication Properties
- [[The matrix transpose operation reflects a matrix over its main diagonal to exchange row and column indices]]
### 120 Matrix Inverses — Definition and Properties
- [[Matrix Inversion requires non-zero determinants and full rank]]
### 121 Algorithms for Computing Matrix Inverses
### 122 Elementary Matrices

## 200 Determinants

1. [[The determinant measures how a matrix scales space and determines invertibility]]
### 201 Geometric Meaning of Determinants — Area and Volume
### 202 Determinant Calculation Rules and Row Operations
### 203 Cofactor Expansion Method
### 204 Cramer's Rule
### 205 The Adjoint Matrix

## 300 Vector Spaces & Subspaces

### 300 Vector Spaces Overview
### 301 Vector Space and Subspace Axioms
### 310 The Fundamental Subspaces of a Matrix
### 311 Null Space — Definition and Calculation
### 312 Column Space and Row Space
### 320 Linear Combinations and Span
### 330 Linear Independence and Dependence
### 340 Basis Vectors and Dimensionality
### 350 Coordinate Systems and Basis Representation

## 400 Linear Transformations

1. [[Linear transformations map vectors between coordinate spaces by redefining basis vectors]]
### 401 Geometric Transformations in Vector Spaces
### 410 Matrix Representation of Linear Transformations
- [[Matrix multiplication combines two linear transformations into a single transformation]] — composing two transformations into one.
### 420 Kernel and Range of a Transformation
### 430 The Rank-Nullity Theorem
### 440 Change of Basis Transformations
### 450 Vector Space Isomorphisms

## 500 Eigen-Theory & Dynamics

### 500 Eigenvalues and Eigenvectors Overview
### 501 Geometric Intuition of Eigen Theory
### 510 The Characteristic Equation
### 520 Matrix Diagonalization
### 521 Complex Eigenvalues Handling
### 530 Eigen Theory Applications — Dynamical Systems and Markov

## 600 Orthogonality & Projections

### 600 Orthogonality Overview
### 601 Inner Products and Vector Distance
### 610 Orthogonal and Orthonormal Sets
### 620 Orthogonal Projections onto Subspaces
### 630 The Gram-Schmidt Orthogonalization Process
### 640 Least Squares Approximation for Inconsistent Systems

## 700 ML Specifics: Norms & Vector Metrics

### 700 Vector and Matrix Norms Overview
1. [[The L1 Norm (Manhattan Distance) calculates absolute distance strictly along grid axes]]
	1. [[L1 Regularization (Lasso) forces feature weights to exactly zero, creating sparse models]]
2. [[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points]]
	1. [[L2 Regularization (Ridge) shrinks feature weights evenly without forcing them to zero]]
### Matrix norm

## 800 ML Specifics: Matrix Calculus

### 800 Matrix Calculus Overview
### 810 The Gradient of a Vector
### 820 The Jacobian Matrix
### 830 Derivatives of Matrix Equations
### 840 Chain Rule for Vectors — Backpropagation Math

## 900 ML Specifics: Tensor Algebra

### 900 Tensor Algebra Overview
### 901 Tensors vs Matrices — Dimensionality
### 910 Tensor Broadcasting Rules
### 920 Tensor Contraction Operations
### 930 Einstein Summation Convention (einsum)

## 1000 Advanced Factorizations & Numerical Methods

### 1000 Matrix Factorizations Overview
### 1010 Symmetric Matrices and the Spectral Theorem
### 1020 Quadratic Forms
### 1030 Singular Value Decomposition (SVD)
### 1031 Principal Component Analysis (PCA) — Math via SVD
### 1040 Cholesky Decomposition
### 1050 QR Decomposition
### 1060 Matrix Condition Numbers and Numerical Stability
### 1070 Iterative Solvers — Conjugate Gradient Method

## Open questions / rabbit holes
- Why is the dot product of two unit vectors the same as $\cos(\theta)$ between them? (geometric proof, see [[Unit vectors and angles isolate a vector's direction from its magnitude]])
- More properties and geometric intuition of the dot product's sign/magnitude, see [[The dot product computes a scalar representing the projection of one vector onto another and their degree of alignment]]

## External resources
- Course:
- Book:
- Docs:

---
*MOC — update as your understanding grows. Let it emerge bottom-up.*
