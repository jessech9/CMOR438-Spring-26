# Unsupervised Learning

This folder collects four notebooks demonstrating the unsupervised
algorithms implemented in `rice_ml`. Each algorithm has its own
folder with a brief `README.md` and a notebook.

## Index

| Algorithm           | Dataset                                | Folder                                                  |
| ------------------- | -------------------------------------- | ------------------------------------------------------- |
| DBSCAN              | Customer Personality Analysis          | [`DBSCAN/`](DBSCAN/)                                    |
| K Means Clustering  | Wholesale Customers Data Set           | [`K Means Clustering/`](K%20Means%20Clustering/)        |
| PCA                 | Fetal Health Classification            | [`PCA/`](PCA/)                                          |
| SVD                 | Wholesale Customers Data Set           | [`SVD/`](SVD/)                                          |

Each notebook reads its CSV from the repository [`data/`](../../data/) folder via `find_data_file`.

## How unsupervised learning works

Unsupervised learning finds structure in data **without labels**.
Common applications:

- group similar points (clustering: k-means, DBSCAN), or
- reduce the dimensionality of features for visualization or
  downstream models (decomposition: PCA, SVD).

Without ground-truth labels we use indirect metrics — inertia,
silhouette score, explained variance — and visual inspection to judge
whether the result is meaningful.