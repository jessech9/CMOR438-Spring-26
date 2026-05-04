# K-Nearest Neighbors

The K-Nearest Neighbors (KNN) algorithm is a non-parametric, lazy learning method used for both classification and regression. It is one of the simplest supervised learning algorithms — the training data *is* the model, and prediction happens entirely at query time by comparing a new point against the stored training set.

## Algorithm Overview

The core idea of KNN is to classify or predict the value of a new data point based on the labels or values of its *k* closest neighbors in the feature space.

### Classification

1. **Objective:** Predict the class label of a new data point.
2. **Process:**
    * Compute the distance (default Euclidean) between the new point and every point in the training set.
    * Identify the *k* training samples with the smallest distances (the "nearest neighbors").
    * The predicted class is the **mode** (most frequent class) among those *k* neighbors. With distance-based weighting, votes are weighted by `1 / d(x, x_i)` instead of being uniform.

### Regression

1. **Objective:** Predict a continuous target value for a new data point.
2. **Process:**
    * Steps are identical to classification up to finding the *k* nearest neighbors.
    * The predicted value is the **mean** (or distance-weighted mean) of the target values of those *k* neighbors.

## Key Hyperparameters

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `n_neighbors` | **The number of neighbors** to consider when making a prediction. The most crucial parameter — small `k` overfits, large `k` over-smooths. | Odd numbers like 3, 5, 7. |
| `metric` | The function used to measure distance between points (`'euclidean'`, `'manhattan'`, `'chebyshev'`). | `'euclidean'` |
| `weights` | How neighbors contribute to the vote — `'uniform'` (every neighbor counts equally) or `'distance'` (closer neighbors count more, weighted by `1/d`). | `'uniform'` |

---

## Data Requirements

### Input Features ($\mathbf{X}$)

* **Format:** Requires a 2D array of shape $(N_{samples}, N_{features})$.
* **Type:** Features must be entirely **numeric**. KNN is highly sensitive to the scale of features, so **standardization (scaling)** of the input data is strongly recommended before training.

### Labels ($\mathbf{Y}$)

* **Classification:** Discrete integers or strings.
* **Regression:** Continuous floating-point numbers.

---

## Notebook & Dataset

* **Notebook:** [`knn.ipynb`](knn.ipynb) — load + quality checks, EDA on per-feature spans (showing why scaling matters), stratified train/test split with `StandardScaler`, raw vs scaled baseline, grid search over `k` / `weights` / `metric`, and a confusion-matrix heatmap.
* **Dataset:** [`Crop_recommendation.csv`](../../../data/Crop_recommendation.csv) — 2200 rows, 7 numeric soil/climate features (N, P, K, temperature, humidity, pH, rainfall), and one of 22 crop labels. Balanced (100 rows per class).
