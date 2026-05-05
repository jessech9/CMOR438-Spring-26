# Random Forest

Random Forest is a supervised ensemble method used for both classification and regression. It averages the predictions of many de-correlated decision trees, each grown on a different bootstrap sample of the data and a different random subset of features at each split. The resulting ensemble has dramatically lower variance than any single tree without a corresponding increase in bias.

## Algorithm Overview

A random forest's two key sources of randomness are **bootstrap sampling** (each tree sees a different sample of roughly 63% of unique training rows, drawn with replacement) and **feature subsampling** (each split inside a tree picks the best feature from a random subset rather than from all features). Together these decorrelate the trees so that averaging their predictions cancels out individual mistakes.

Why a tree at all? A single deep CART tree is high-variance — it can memorize the training set, so two trees fit on slightly different bootstrap samples can disagree wildly on test points. That high variance is exactly what bagging exploits: the variance of the average of `M` identically-distributed estimators with pairwise correlation `ρ` shrinks as `M` grows, but only down to a floor of `ρ · σ²`. Feature subsampling is what makes `ρ` small enough for that floor to be useful.

### Classification

1. **Objective:** Predict the class label of a new data point.
2. **Process:**
    * Train `n_estimators` decision-tree classifiers, each on its own bootstrap sample with split candidates restricted to a random subset of `max_features`.
    * For a new point, every tree returns a class-probability vector.
    * The forest predicts the class with the **highest average probability** across all trees (soft voting). With hard voting, the forest takes the **mode** of the trees' top predictions instead.

### Regression

1. **Objective:** Predict a continuous target value for a new data point.
2. **Process:**
    * Train `n_estimators` decision-tree regressors, each on its own bootstrap sample with `max_features` split candidates.
    * For a new point, every tree returns a numeric prediction.
    * The forest predicts the **mean** of those predictions.

## Key Hyperparameters

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `n_estimators` | **Number of trees** in the forest. More trees reduce variance up to a plateau; doubling beyond the plateau buys very little. | 100 (50 is often enough on small data). |
| `max_features` | **Random subset size** considered at each split. Lower values give more decorrelation between trees but also weaker individual trees. | `'sqrt'` for classification, ~`n_features / 3` for regression. |
| `max_depth` | Maximum depth of each tree. `None` lets trees grow until pure / `min_samples_leaf` is hit, which is what RF usually wants (let trees overfit, average them out). | `None`, or a small int for regularization. |
| `min_samples_leaf` | Minimum samples allowed in a leaf. Useful for noisy data. | 1 for classification, 5 for regression. |
| `random_state` | Seed for bootstrap sampling and the split-time feature subsets. Set for reproducibility. | `0` or any fixed int. |

---

## Data Requirements

### Input Features ($\mathbf{X}$)

* **Format:** Requires a 2D array of shape $(N_{samples}, N_{features})$.
* **Type:** Random forests handle numeric features natively; categorical features should be integer-encoded (or one-hot-encoded) before fitting.
* **Scaling:** Random forests are tree-based and **invariant to monotonic feature transformations**, so feature scaling / standardization is *not* required. (Contrast with KNN, where scaling is critical.)

### Labels ($\mathbf{Y}$)

* **Classification:** Discrete integers or strings.
* **Regression:** Continuous floating-point numbers.

---

## Notebook & Dataset

* **Notebook:** [`random_forest.ipynb`](random_forest.ipynb) — full walk-through: imports, EDA (per-feature KDEs split by class), stratified train/test split, single decision tree baseline, 25-tree random forest, confusion-matrix heatmap, per-class precision/recall/F1, an `n_estimators` learning curve, permutation-based feature importance, and a discussion of where the forest still confuses classes. All plots and tables are pre-rendered inline.
* **Dataset:** [`Crop_recommendation.csv`](../../../data/Crop_recommendation.csv) — 2200 rows, 7 numeric soil/climate features (N, P, K, temperature, humidity, pH, rainfall), and one of 22 crop labels. Balanced (100 rows per class). Same dataset as the KNN notebook.
