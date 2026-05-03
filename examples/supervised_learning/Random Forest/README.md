# Random Forest — Crop Recommendation

A random forest averages many de-correlated decision trees. Each tree
is grown on a **bootstrap sample** of the data and considers only a
random subset of features at each split. The combination of
bootstrapping + feature subsampling decorrelates the trees so that
averaging dramatically reduces variance.

## Mathematical Explanation

For an ensemble of $M$ trees $h_1, \dots, h_M$, the forest predicts:

$$\hat{y}(x) = \frac{1}{M} \sum_{m=1}^{M} h_m(x) \quad \text{(regression)}$$

$$\hat{y}(x) = \mathrm{mode}\big\{ h_1(x), \dots, h_M(x) \big\} \quad \text{(classification)}$$

The variance of the average of $M$ identically-distributed estimators
with variance $\sigma^2$ and pairwise correlation $\rho$ is:

$$\rho\,\sigma^2 + \frac{1 - \rho}{M}\,\sigma^2.$$

That is exactly why we need *de-correlated* trees: feature
subsampling makes $\rho$ small.

## Dataset

[`Crop_recommendation.csv`](../../../data/Crop_recommendation.csv) —
the 22-class farming dataset reused once more so we can compare three
different model families (decision tree, k-NN, random forest) on
identical data.

## Notebook

[`random_forest.ipynb`](random_forest.ipynb) — single tree vs. forest,
and a learning curve over `n_estimators`.
