# K Means Clustering

K-means partitions data into $K$ spherical clusters by minimizing
within-cluster variance. It is fast, scalable, and the default first
thing to try when you want to discover groups in tabular numeric data.

## Mathematical Explanation

K-means minimizes the **inertia** (within-cluster sum of squares):

$$\min_{\{C_k\}, \{\mu_k\}}\; \sum_{k=1}^{K}\;\sum_{x \in C_k}\, \lVert x - \mu_k \rVert_2^2$$

**Lloyd's algorithm** alternates two simple steps:

1. **Assignment.** Each point is assigned to its closest centroid.
2. **Update.** Recompute centroids as cluster means.

Iterate until centroids stop moving (or move less than `tol`).

`rice_ml` initializes with **k-means++** by default — picking the first
centroid uniformly at random and each subsequent one with probability
proportional to $D(x)^2$ (the squared distance to the nearest existing
centroid). This dramatically reduces sensitivity to the random seed.

## When to Use

- Roughly spherical, equal-variance clusters in low to moderate
  dimensions.
- When you have a rough estimate of $K$ (refined via the elbow /
  silhouette method).
- When speed matters — k-means is one of the fastest clustering
  algorithms.

## Notebook

[`k_means.ipynb`](k_means.ipynb) — applied to the Wholesale Customers
Data Set (log-scaled spending channels). The same notebook also lives under
[`supervised_ml/K Means Clustering`](../../supervised_ml/K%20Means%20Clustering/).