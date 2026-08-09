---
id: "20260726174600"
type: permanent
subtype: concept
created: 2026-07-26
aliases:
  - matrix multiplication
tags:
  - linear-algebra
Up: "[[Matrix-vector multiplication represents a linear transformation mapping a vector from one space to another]]"
---
# Matrix multiplication combines two linear transformations into a single transformation

Matrix multiplication corresponds to combining two linear transformations into a third one. 

Let's assume we have a basis that goes through 2 linear transformations. If you ignore the middle one, theres a linear transformation between the first and the third, the question is what matrix corresponds to that linear transformation?

Visually it goes like this.
![[file-20260726174641826.jpg]]


 To multiply the two matrices to obtaining the third one, first we put the second matrix and then the first one. As we do in [[Linear transformations map vectors between coordinate spaces by redefining basis vectors]] we can see the first transformation as the first basis group of vectors so they go on the right (like the vector to transform) and the transformator we always put it on the left:
 ![[file-20260726175159350.jpg]]
So it will always go like this:
$$\text{transformator}*\text{thing to transform}=\text{thing transformed}$$

And to multiply matrices is almost the same as [[Matrix-vector multiplication represents a linear transformation mapping a vector from one space to another|matrix-vector multiplication]]: a lot of dot products.
![[file-20260726175509070.jpg]]

So here, the resulting matrix is:
$$\begin{bmatrix} 5 & 0 \\ 2 & 4 \end{bmatrix}$$

We can also multiply matrices of different dimensions:
![[file-20260726175918629.jpg]]

## What Does Non-Square Matrix Multiplication Mean?

To interpret this geometrically, you have to remember one golden rule of linear algebra: **every matrix represents a linear transformation (a mapping) from one vector space to another.**

When you multiply two matrices of different dimensions, you are simply **chaining together two transformations that shift between different spatial dimensions**.

### The Dimensional Journey

Let's break down the exact dimensions from your image:

| **Matrix**              | **Dimensions** | **Geometric Meaning**                | **What Happens to Space?**                                  |
| ----------------------- | -------------- | ------------------------------------ | ----------------------------------------------------------- |
| **Right Matrix ($B$)**  | $3 \times 4$   | Maps $\mathbb{R}^4 \to \mathbb{R}^3$ | Takes 4D vectors and projects them into 3D space.           |
| **Left Matrix ($A$)**   | $2 \times 3$   | Maps $\mathbb{R}^3 \to \mathbb{R}^2$ | Takes those new 3D vectors and squishes them into 2D space. |
| **Result Matrix ($C$)** | $2 \times 4$   | Maps $\mathbb{R}^4 \to \mathbb{R}^2$ | The **direct shortcut** mapping directly from 4D to 2D!     |

When you compute $A \cdot B$, you are asking: _"If I apply transformation $B$ first, and then immediately apply transformation $A$ to the result, what is the single, combined transformation?"_

## The Meaning in Terms of Basis Vectors

The most intuitive way to read any matrix is by looking at its **columns**.

> **The Secret Ring:** The columns of any matrix tell you _exactly where the standard basis vectors of the input space land in the output space._

### 1. What Matrix $B$ Does to 4D Basis Vectors

In 4D space ($\mathbb{R}^4$), you have four standard basis vectors: $\hat{e}_1, \hat{e}_2, \hat{e}_3, \hat{e}_4$.

Matrix $B$ tells you where those four vectors land in 3D space ($\mathbb{R}^3$):
- $\hat{e}_1$ lands at column 1: $\begin{bmatrix} 3 \\ 1 \\ -2 \end{bmatrix}$
- $\hat{e}_2$ lands at column 2: $\begin{bmatrix} 0 \\ 5 \\ 1 \end{bmatrix}$
- And so on for columns 3 and 4.

### 2. What Matrix $A$ Does to Those New Vectors

Now, transformation $A$ comes along. Its own columns tell us where the 3D basis vectors ($\hat{i}, \hat{j}, \hat{k}$) land in 2D space ($\mathbb{R}^2$):

- $\hat{i}$ lands at $\begin{bmatrix} 3 \\ 2 \end{bmatrix}$, $\hat{j}$ lands at $\begin{bmatrix} 1 \\ -1 \end{bmatrix}$, and $\hat{k}$ lands at $\begin{bmatrix} 4 \\ 2 \end{bmatrix}$.

When $A$ multiplies $B$, **it transforms the columns of $B$**. Each column of the new matrix $C$ is simply a linear combination of the columns of $A$, weighted by the numbers in $B$!

### 3. The Grand Finale: What the Result Matrix $C$ Means

Let's look at the very first column of your final blue matrix: $\begin{bmatrix} 2 \\ 1 \end{bmatrix}$.

What does this vector actually mean?

1. We started with the first standard basis vector in 4D space: $[1, 0, 0, 0]^T$.
2. Transformation $B$ moved it to $[3, 1, -2]^T$ in 3D space.
3. Transformation $A$ then grabbed that 3D vector and mapped it into 2D space by doing:
$$3\begin{bmatrix} 3 \\ 2 \end{bmatrix} + 1\begin{bmatrix} 1 \\ -1 \end{bmatrix} - 2\begin{bmatrix} 4 \\ 2 \end{bmatrix} = \begin{bmatrix} 9 + 1 - 8 \\ 6 - 1 - 4 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

**In summary:** The columns of your new $2 \times 4$ matrix represent the final resting places in 2D space of the four original standard basis vectors from 4D space, after undergoing both dimensional shifts!

Since mapping from 4D down to 2D inevitably squishes higher-dimensional space into a flatter plane, would you like to explore how to calculate **how much information is lost** during this transformation (using concepts like _rank_ and the _null space_)?