# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This package implements the **DBSCAN** algorithm, a method for **unsupervised clustering** that identifies groups based on the density of data points. Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand and can discover arbitrarily shaped clusters while explicitly identifying outliers (noise).

## Algorithm Overview

DBSCAN classifies every point in the dataset into one of three roles:

1. **Core Point:** A point that has at least `min_samples` points within its $\epsilon$ (epsilon) radius.
2. **Border Point:** A point that lies inside the $\epsilon$-neighborhood of a Core Point but does not itself satisfy the `min_samples` criterion.
3. **Noise Point (Outlier):** A point that is neither a Core Point nor a Border Point. It is assigned label `-1`.

### Clustering Mechanism

A cluster is formed by starting at a random, unvisited Core Point and recursively adding all points that are **density-reachable** from it.

* **Directly Density-Reachable:** Point $p$ is directly density-reachable from $q$ if $p \in N_\epsilon(q)$ and $q$ is a Core Point.
* **Density-Reachable:** A chain of directly density-reachable points exists between the two endpoints.

The algorithm sweeps through every unvisited point: if the point is a core point, it seeds a new cluster and grows it by exploring its $\epsilon$-neighbors transitively; otherwise the point is left as noise (it may later be reclaimed as a border point if a neighboring core point sweeps it up).

## Key Hyperparameters

DBSCAN's performance is highly sensitive to the correct tuning of its two core parameters:

| Parameter | Type | Description | Effect on Clustering |
| :--- | :--- | :--- | :--- |
| `eps` ($\epsilon$) | `float` | **Neighborhood Radius.** The maximum distance to look for neighboring samples. | Determines the reach of the local density measure. Too small → most points become noise; too large → distinct clusters merge. |
| `min_samples` | `int` | **Density Threshold.** The minimum number of points required to form a dense region (i.e., to define a Core Point). | Controls the sensitivity to noise and the minimum size of a cluster. Higher → only clearly dense regions become clusters. |
| `metric` | `str` | Distance function (`'euclidean'`, `'manhattan'`). | Controls the geometry of the neighborhood. |

A common heuristic for choosing `eps` is to plot the sorted `k`-distance graph (distance to each point's `k`-th nearest neighbor, where `k = min_samples`) and pick the "elbow" — the value at which the curve turns sharply upward.

---

## Data Requirements

DBSCAN is a distance-based algorithm, making it sensitive to the scale of the input features.

* **Features ($\mathbf{X}$):** Must be a 2D numeric array of shape $(N_{samples}, N_{features})$.
* **Scaling:** **Feature scaling (standardization or normalization)** is highly recommended to ensure that all dimensions contribute equally to the distance calculation.
* **Labels:** None — DBSCAN is unsupervised. After `fit`, the resulting cluster IDs are stored in `labels_` (with `-1` marking noise points).

---

## Notebook & Dataset

* **Notebook:** [`dbscan.ipynb`](dbscan.ipynb) — outlier filtering, feature engineering (eight numeric features for age, income, total spend, total purchases, recency, web visits, kids/teens at home), DBSCAN on the standardized matrix, and a 2-D PCA visualization of the resulting clusters and noise points.
* **Dataset:** [`marketing_campaign.csv`](../../../data/marketing_campaign.csv) — 2240 grocery customers described by 29 demographic, behavioral, and spending features (Customer Personality Analysis dataset, tab-separated).
