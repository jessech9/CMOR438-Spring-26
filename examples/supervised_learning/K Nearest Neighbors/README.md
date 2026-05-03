# K Nearest Neighbors — Crop Recommendation

k-NN is a non-parametric algorithm: the training data **is** the
model. To predict for a new point, find its $k$ closest training
points and aggregate their labels.

## Mathematical Explanation

Given a query $x$ and a distance metric $d(\cdot, \cdot)$, find the
indices of the $k$ smallest distances:

$$\mathcal{N}_k(x) = \arg\min_{|S|=k} \sum_{i \in S} d(x, x_i)$$

For **classification**, predict the majority class:

$$\hat{y}(x) = \mathrm{mode}\,\{ y_i : i \in \mathcal{N}_k(x) \}$$

For **regression**, predict the (optionally weighted) mean:

$$\hat{y}(x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i\,y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}, \quad w_i = 1 \text{ or } 1/d(x, x_i)$$

The Euclidean distance is the default; Manhattan and Chebyshev are
also supported in `rice_ml`.

## Dataset

[`Crop_recommendation.csv`](../../../data/Crop_recommendation.csv) —
the same 22-class farming dataset as the Decision Tree notebook, used
here to highlight the importance of feature scaling for distance-based
methods (rainfall in millimetres would otherwise dwarf pH and
temperature).

## Notebook

[`knn.ipynb`](knn.ipynb) — structured like the reference course notebook:
load and quality checks, EDA (class counts and feature spans), stratified
split with `StandardScaler`, raw vs scaled baseline, `GridSearchCV` over
`k` / `weights` / `metric`, test accuracy and confusion-matrix heatmap,
and a short discussion of results and limitations.
