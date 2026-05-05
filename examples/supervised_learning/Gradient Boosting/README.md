# Gradient Boosting

Gradient Boosting is a supervised ensemble method that builds a strong
predictor by adding many small, weak models one stage at a time. In this
project, `GradientBoostingClassifier` handles **binary classification**
by fitting shallow regression trees to the current pseudo-residuals of a
log-odds model.

## Algorithm Overview

The core idea is to start with a simple initial prediction, then improve
it gradually. Each new tree tries to correct the mistakes left by the
trees that came before it.

### Binary Classification

1. **Objective:** Predict one of two class labels.
2. **Process:**
    * Initialize the model with the log-odds of the positive class.
    * Convert the current raw score to a probability with the sigmoid
      function.
    * Compute pseudo-residuals, `y - sigmoid(F)`.
    * Fit a shallow regression tree to those residuals.
    * Add the tree's prediction to the ensemble, scaled by the learning
      rate.
    * Repeat for `n_estimators` boosting stages.

The final prediction is based on the boosted probability
`P(class 1) = sigmoid(F(x))`.

## Key Hyperparameters

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `n_estimators` | Number of boosting stages, or shallow trees, to add. More trees can improve fit but increase runtime and overfitting risk. | 50-200 |
| `learning_rate` | Shrinks each tree's contribution. Smaller values make learning more cautious and usually need more trees. | 0.05-0.2 |
| `max_depth` | Maximum depth of each regression tree. Small depths keep each learner weak and interpretable. | 1-3 |
| `min_samples_split` | Minimum samples required to split an internal tree node. | 2 |
| `min_samples_leaf` | Minimum samples required at each leaf node. | 1 |
| `random_state` | Seed for reproducible tree construction. | Any fixed integer |

---

## Data Requirements

### Input Features ($\mathbf{X}$)

* **Format:** Requires a 2D array of shape
  $(N_{samples}, N_{features})$.
* **Type:** Features must be numeric.
* **Scaling:** Tree models do not require scaling in the same way that
  distance-based methods do, but the notebook uses `StandardScaler` to
  stay consistent with the other supervised examples.

### Labels ($\mathbf{Y}$)

* **Classification:** This implementation supports exactly two target
  classes.
* **Encoding:** Labels can be any two discrete values; internally, they
  are mapped to class indices.

---

## Notebook & Dataset

* **Notebook:** [`gradient_boosting.ipynb`](gradient_boosting.ipynb) —
  load + quality checks, exploratory plots, stratified train/test split,
  `StandardScaler`, a `rice_ml` gradient boosting classifier, test-set
  metrics, confusion matrix, probability histogram, ROC curve, feature
  importances, and a written discussion.
* **Dataset:** [`BankNote_Authentication.csv`](../../../data/BankNote_Authentication.csv)
  — 1372 rows, 4 numeric wavelet-transform features (`variance`,
  `skewness`, `curtosis`, `entropy`), and a binary `class` label for
  banknote authentication.

## Implementation

The notebook uses the source implementation here:

```python
from rice_ml.supervised_learning.gradient_boosting import GradientBoostingClassifier
```

The estimator is implemented in
[`src/rice_ml/supervised_learning/gradient_boosting.py`](../../../src/rice_ml/supervised_learning/gradient_boosting.py).
