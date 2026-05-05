# PCA - Fetal Health Classification

Principal Component Analysis (PCA) projects high-dimensional data onto
orthogonal directions that preserve as much variance as possible. In
this notebook, PCA compresses the fetal-health feature matrix into two
dimensions for visualization.

## Mathematical Explanation

After centering the data matrix $X$, PCA finds unit vectors $v_1, v_2,
\dots$ that maximize projected variance:

$$v_1 = \arg\max_{\lVert v \rVert_2 = 1} \mathrm{Var}(Xv).$$

Subsequent components are constrained to be orthogonal to the previous
ones. Equivalently, the components are the eigenvectors of the sample
covariance matrix:

$$\Sigma = \frac{1}{n - 1} X^\top X.$$

The explained variance ratio reports how much of the total variance is
captured by each component:

$$r_j = \frac{\lambda_j}{\sum_k \lambda_k}.$$

## Dataset

[`fetal_health.csv`](../../../data/fetal_health.csv) - fetal
cardiotocography measurements with 21 numeric features and an expert
`fetal_health` label. PCA is fit without using the label; the notebook
only colors the 2-D projection by label afterward to interpret the
visualization.

## Notebook

[`pca.ipynb`](pca.ipynb) - load and standardize the fetal-health
features, project them into two principal components, visualize the
classes in PCA space, and inspect cumulative explained variance.
