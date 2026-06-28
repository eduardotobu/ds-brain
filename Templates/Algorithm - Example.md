---
id: 20260627-1002
type: permanent
subtype: algorithm
created: 2026-06-27
modified: 2026-06-27
aliases: [k-means, Lloyd's algorithm, k-means++]
tags: [ml, unsupervised, clustering, algorithm]
family: clustering
---
# K-Means partitions data into k clusters by minimizing within-cluster variance through iterative centroid reassignment

## TL;DR
Assigns n points to k clusters by repeatedly (1) assigning each point to its
nearest centroid and (2) recomputing centroids as cluster means. Minimizes
total inertia (within-cluster sum of squared distances). Converges to a local,
not global, minimum.

## Intuition
Drop k magnets on a scatter plot. Each point sticks to the nearest magnet.
Move each magnet to the average position of all points stuck to it. Repeat.
The magnets settle when no reassignment would reduce total distance.

## How it works

### Core mechanism — Lloyd's algorithm
1. **Initialize** k centroids (randomly or via K-Means++)
2. **Assign** each point to its nearest centroid (Euclidean distance)
3. **Update** each centroid to the mean of its assigned points
4. **Repeat** 2–3 until centroids stop moving (or `max_iter` is reached)

### Objective function

$$
\underset{C}{\text{minimize}} \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

where $\mu_i$ is the centroid of cluster $C_i$. This is NP-hard in general;
Lloyd's finds a local minimum efficiently.

## Key hyperparameters

| Parameter      | What it controls          | Practical advice                              |
|----------------|---------------------------|-----------------------------------------------|
| `n_clusters` k | Number of clusters        | Elbow method or silhouette score              |
| `init`         | Centroid initialization   | Always use `k-means++` (default in sklearn)   |
| `n_init`       | Number of random restarts | 10 is usually fine; raise if inertia varies   |
| `max_iter`     | Max steps per run         | 300 rarely limits convergence                 |

## Choosing k
- **Elbow method**: plot inertia vs k; pick the point of diminishing returns.
- **Silhouette score**: measures intra-cluster cohesion vs inter-cluster
  separation. Maximize it.
- Domain knowledge often beats both.

## Assumptions
- Clusters are **spherical** (isotropic) and **convex**
- Clusters have **similar sizes** and **similar densities**
- Features are on the **same scale** (K-Means is NOT scale-invariant)

## When to use ✅ / avoid ❌
✅ Fast exploratory clustering on low-to-medium dimensional tabular data  
✅ When you have a rough prior on number of clusters  
✅ Customer segmentation, image compression, document clustering (TF-IDF)  
✅ As a preprocessing step — cluster labels as engineered features  
❌ Non-convex clusters (rings, crescents) → use DBSCAN or HDBSCAN  
❌ Very different cluster sizes or densities → use GMM  
❌ High-dimensional raw features → distance degrades (Curse of Dimensionality)  
❌ Data with many outliers → use K-Medoids  

## Complexity

|           | Time          | Space   |
|-----------|---------------|---------|
| Training  | O(n · k · d · i) | O(n · k) |
| Inference | O(n · k · d)  | O(k · d) |

n = samples, k = clusters, d = dimensions, i = iterations

## Minimal implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale first — always
X_scaled = StandardScaler().fit_transform(X)

km = KMeans(
    n_clusters=5,
    init="k-means++",   # smarter init, far fewer bad local minima
    n_init=10,
    random_state=42,
)
labels    = km.fit_predict(X_scaled)
centroids = km.cluster_centers_   # shape (k, d)
inertia   = km.inertia_

# Choosing k — silhouette score
from sklearn.metrics import silhouette_score

scores = {
    k: silhouette_score(
        X_scaled,
        KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    )
    for k in range(2, 11)
}
best_k = max(scores, key=scores.get)
```

## Variants & extensions
- **K-Means++** — smarter initialization that spreads seeds proportionally to
  squared distance from existing centroids. Default in sklearn, always prefer.
- **Mini-batch K-Means** — uses random subsamples each iteration; much faster
  for large n, marginally worse quality.
- **K-Medoids (PAM)** — centroids must be actual data points; robust to outliers.
- **Bisecting K-Means** — hierarchical variant; splits one cluster per step.
- **Gaussian Mixture Models (GMM)** — probabilistic generalization that handles
  elliptical clusters and gives soft (probabilistic) assignments.

## Connected concepts
- Builds on:    [[The Curse of Dimensionality makes distance metrics unreliable at scale]]
- Related to:   [[DBSCAN finds density-based clusters and is naturally robust to outliers]]
- Related to:   [[Gaussian Mixture Models generalize K-Means with soft probabilistic assignments]]
- Alternative:  [[Hierarchical Clustering builds a dendrogram of nested cluster merges]]
- Precede with: [[PCA projects data onto orthogonal axes of maximum variance]]
- See MOC:      [[MOC - Example]]

## Key papers / sources
- MacQueen (1967). "Some methods for classification and analysis of multivariate
  observations." — original formulation
- Arthur & Vassilvitskii (2007). "k-means++: The advantages of careful seeding."
  — the initialization improvement that actually matters
- [[ESL — Elements of Statistical Learning — Hastie et al.]] Ch. 14
