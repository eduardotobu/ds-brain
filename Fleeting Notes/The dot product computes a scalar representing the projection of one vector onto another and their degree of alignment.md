---
id: "20260719184736"
type: permanent
subtype: concept
created: 2026-07-19 18:47:36
aliases: []
tags:
  - linear-algebra
Up: "[[Vectors are the foundational data structure of Machine Learning]]"
---
# The dot product computes a scalar representing the projection of one vector onto another and their degree of alignment

The dot product (or scalar product) is a fundamental algebraic operation that takes two equal-length sequences of numbers and returns a single scalar. Geometrically, it captures the interaction between two vectors $\mathbf{u}$ and $\mathbf{v}$ by quantifying how much they point in the same direction. When the dot product is zero, the vectors are orthogonal (perpendicular); a positive value indicates they point in the same general direction, while a negative value indicates they point in opposing directions.

### Mathematical Representation

Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, the algebraic dot product is the sum of the products of their corresponding components:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = u_1 v_1 + u_2 v_2 + \dots + u_n v_n$$

The geometric definition links the dot product to the lengths (norms) of the vectors and the cosine of the angle $\theta$ between them:

$$\mathbf{u} \cdot \mathbf{v} = \Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{} \cos(\theta)$$

Where $\Vert{}\mathbf{u}\Vert{} = \sqrt{\sum u_i^2}$ is the Euclidean norm of $\mathbf{u}$. This relationship allows for the calculation of the angle between vectors: $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{}}$.

- **Positive Value ($\mathbf{u} \cdot \mathbf{v} > 0$):** Occurs when $\cos(\theta) > 0$, meaning the angle $\theta$ is acute ($0 \le \theta < \pi/2$). The vectors point in the same general direction. The value is maximized when $\theta = 0$ (perfectly aligned).
- **Zero Value ($\mathbf{u} \cdot \mathbf{v} = 0$):** Occurs when $\cos(\theta) = 0$, meaning the angle $\theta = \pi/2$ ($90^\circ$). The vectors are exactly orthogonal (perpendicular). There is zero projection of $\mathbf{u}$ onto $\mathbf{v}$.
- **Negative Value ($\mathbf{u} \cdot \mathbf{v} < 0$):** Occurs when $\cos(\theta) < 0$, meaning the angle $\theta$ is obtuse ($\pi/2 < \theta \le \pi$). The vectors point in opposing general directions. The value is minimized (most negative) when $\theta = \pi$ (perfectly opposed).

![[file-20260719185331952.jpg]]

![[file-20260719185807087.jpg|475]]

### Python Representation


``` python
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# The dot product can be calculated using np.dot() or the @ operator
dot_result = np.dot(u, v)
dot_operator = u @ v

print(f"Dot product: {dot_result}") # Output: 32 (1*4 + 2*5 + 3*6 = 4+10+18 = 32)
```

### Semantic Meanings in ML

- **Cosine Similarity:** In Natural Language Processing and Recommendation Systems, the dot product of normalized vectors represents cosine similarity, a core metric for determining semantic closeness between document embeddings or user preferences.
- **Scaled Dot-Product Attention:** In Transformer architectures, the dot product between Query and Key vectors determines the "attention score," allowing the model to focus on relevant segments of input sequences.
- **Linear Classifiers:** A linear neuron computes $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$. The dot product determines the projection of the input $\mathbf{x}$ onto the weight vector $\mathbf{w}$, with the result deciding which side of the decision boundary the input falls on.
- **Projection and Dimensionality Reduction:** Principal Component Analysis (PCA) relies on projecting data onto principal components, which is mathematically implemented through the dot product between data vectors and orthonormal basis vectors.