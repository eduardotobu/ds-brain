---
id: "20260708233112"
type: permanent
subtype: concept
created: 2026-07-08 23:31:12
aliases:
  - Vector Direction
tags:
  - machine-learning
  - linear-algebra
Up: "[[Vectors are the foundational data structure of Machine Learning]]"
---
# Unit vectors and angles isolate a vector's direction from its magnitude
Every vector inherently contains two pieces of information: its magnitude (length) and its **direction** (where it points in space). In machine learning, we often need to isolate the direction and discard the magnitude—for example, when we care about _what_ an embedding represents rather than _how strongly_ it represents it.

We determine and represent a vector's direction in two main ways: geometrically (using angles) or algebraically (by normalizing it into a **unit vector**).

### Mathematical Representation

**1. The Unit Vector (n-Dimensional)**

The most common way to represent direction in ML is to create a unit vector ($\hat{\mathbf{v}}$). A unit vector points in the exact same direction as the original vector but has a guaranteed [[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points|L2 Norm]] (magnitude) of exactly 1. You find it by dividing the vector by its own magnitude:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}$$

The components of this unit vector are highly special: they are the **direction cosines**, meaning each component equals the cosine of the angle the vector forms with that specific coordinate axis.

**2. Angles (2D and 3D)**

In lower dimensions, you can calculate the explicit angle ($\theta$) the vector forms with the standard axes using trigonometry. For a 2D vector $\mathbf{v} = \begin{bmatrix} x \\ y \end{bmatrix}$, the angle from the x-axis is found using the arctangent:

$$\theta = \arctan\left(\frac{y}{x}\right)$$

_(Note: In computation, we use a special function called `atan2(y, x)` to ensure the angle is placed in the correct geometric quadrant)._

### Python Representation

In NumPy, we usually extract direction by normalizing the vector. For explicit 2D angles, we use `np.arctan2`.

``` python
import numpy as np

# Define a 2D vector
v = np.array([3, 4])

# --- 1. Algebraic Direction (The Unit Vector) ---
magnitude = np.linalg.norm(v)
unit_vector = v / magnitude

print(f"Original Vector: {v}")
print(f"Unit Vector (Pure Direction): {unit_vector}") 
# Output: [0.6, 0.8]. Notice how 0.6^2 + 0.8^2 = 1.0

# --- 2. Geometric Direction (Angle in 2D) ---
# arctan2 takes (y, x) to handle quadrant signs correctly
theta_radians = np.arctan2(v[1], v[0])
theta_degrees = np.degrees(theta_radians)

print(f"Angle from X-axis: {theta_degrees:.2f} degrees")
# Output: 53.13 degrees
```

### Semantic Meanings in ML

Isolating vector direction drives several core optimization and similarity mechanics:

- **Gradient Descent Optimization:** During neural network training, the calculus-derived "gradient" is simply a vector. The model ignores the absolute coordinates of this vector and focuses entirely on its _direction_, taking a step in the exact opposite direction to reach the lowest point of the loss function.
- **Hypersphere Projection (Embeddings):** When dealing with Large Language Models (LLMs), word embeddings are often mathematically normalized into unit vectors before being compared. This projects all data points onto the surface of an $n$-dimensional sphere, ensuring that clustering algorithms group words based strictly on semantic direction, ignoring how frequently the words appeared (magnitude).
- **Cosine Similarity:** Calculating [[Cosine Similarity measures the angle between vectors instead of magnitude|Cosine Similarity]] is mathematically identical to taking the dot product of two _unit vectors_. By dividing vectors by their magnitudes, the formula strips away size and leaves only the directional overlap.

**Related notes to create/link:**

- The core size metric: [[The L2 Norm (Euclidean Distance) calculates the shortest straight-line path between points]]
- Moving against the gradient: [[Gradient Descent takes steps in the opposite direction of the derivative]]
- Comparing directional overlap: [[Cosine Similarity measures the angle between vectors instead of magnitude]]