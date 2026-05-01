# DBSCAN — Customer Personality Analysis

DBSCAN groups points by **density** rather than distance to a
centroid. It finds arbitrarily-shaped clusters and explicitly labels
low-density points as **noise** (label `-1`). You don't have to
specify the number of clusters in advance.

## Mathematical Explanation

Two hyperparameters:

- $\epsilon$ — the **neighborhood radius**.
- $\mathrm{minPts}$ — the minimum number of points required for a
  region to be "dense".

A point $p$ is a **core point** if its $\epsilon$-neighborhood
contains at least $\mathrm{minPts}$ points (including itself):

$$\big|N_\epsilon(p)\big| \ge \mathrm{minPts}, \quad N_\epsilon(p) = \{\,q : \lVert p - q \rVert \le \epsilon\,\}$$

Two points are **density-reachable** if there is a chain of core
points connecting them. Clusters are maximal sets of mutually
density-reachable points; everything else is noise.

The algorithm scans each unvisited point, and if it is a core point,
it grows the cluster by exploring its $\epsilon$-neighbors.

## Dataset

[`marketing_campaign.csv`](../../../data/marketing_campaign.csv) —
2240 grocery customers described by 29 demographic, behavioral, and
spending features. We engineer eight numeric features (age, income,
total spend, total purchases, recency, web visits, kids/teens at home)
and run DBSCAN on the standardized matrix.

## Notebook

[`dbscan.ipynb`](dbscan.ipynb) — outlier filtering, DBSCAN on
standardized features, and a 2-D PCA visualization of the resulting
clusters and noise points.
