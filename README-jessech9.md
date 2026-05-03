# rice_ml — Supervised & Unsupervised Learning Models

This branch implements linear models and k-nearest neighbors from scratch,
with example notebooks and a test suite.

---

## Source Code

### `supervised_learning/linear.py`

Implements two sklearn-style estimators solved without external ML libraries.

**`LinearRegression`**
- Ordinary Least Squares solved via `numpy.linalg.lstsq` for numerical
  stability on rank-deficient inputs
- Optional intercept term (`fit_intercept=True` by default)
- Attributes: `coef_`, `intercept_`

**`LogisticRegression`**
- Binary classifier trained by full-batch gradient descent
- Numerically stable sigmoid
- Optional L2 regularization (`alpha`), configurable learning rate,
  max iterations, and convergence tolerance
- Intercept excluded from regularization penalty
- Attributes: `coef_`, `intercept_`, `classes_`, `loss_history_`, `n_iter_`

---

## Notebooks

### Linear Regression — Steel Industry Energy Consumption
**File:** `linear_regression.ipynb`

Predicts energy usage (`Usage_kWh`) at a steel plant from operational sensor
readings using Ordinary Least Squares.

- Extracted `hour` and `month` from the timestamp
- One-hot encoded `WeekStatus`, `Day_of_week`, `Load_Type` with `drop_first=True`
- Fit/transform split: scaler fit on train set only, applied to both splits
- Evaluated with R², RMSE, and MAE
- Coefficient table ranked by absolute value on standardized inputs
- Residual vs. fitted plot to diagnose non-linearity

**Key findings:** Lagging reactive power and CO₂ emissions dominate. Model
achieves R² ≳ 0.9, but residual fanning suggests load type behaves
non-linearly — a tree-based model would close the gap.

---

### Logistic Regression — Banknote Authentication
**File:** `logistic_regression.ipynb`

Binary classification (genuine vs. forged banknotes) using four
wavelet-transform features: `variance`, `skewness`, `curtosis`, `entropy`.

- Stratified 75/25 train-test split
- StandardScaler fit on train set only
- L2-regularized logistic regression trained with gradient descent
- Evaluated with accuracy, ROC-AUC, precision, recall, and confusion matrix
- Training loss curve plotted to verify convergence

**Results:** Accuracy 98.25%, ROC-AUC 99.92%, Precision 98.67%, Recall 97.37%.

---

### K-Means Clustering — Wholesale Customers
**File:** `k_means.ipynb`

Groups Portuguese wholesale clients by annual spending patterns across six
product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen).

- Applied `log1p` scaling to handle heavy-tailed spend distributions
- Fit K-Means (k=3) and visualized clusters in 2D (Fresh vs. Grocery)
- Elbow (inertia) and silhouette diagnostics for k=2 through k=10

**Key findings:** Log scaling is essential when spend features span very
different magnitudes. Elbow and silhouette plots don't always agree — use
both alongside domain knowledge. Cluster membership can be compared against
`Channel` (HoReCa vs. Retail) post-hoc.

---

## Tests

### `test_linear.py`

Covers `LinearRegression` and `LogisticRegression`:

| Test | What it checks |
|------|----------------|
| `test_linear_regression_recovers_known_coefficients` | Coefficients and intercept match ground truth within tolerance; R² > 0.95 |
| `test_linear_regression_no_intercept` | `fit_intercept=False` sets intercept to 0 and recovers exact coefficients |
| `test_logistic_regression_separates_well` | Accuracy > 0.95 on separable data; probabilities sum to 1 |
| `test_logistic_regression_loss_is_monotonically_decreasing` | Loss is non-increasing across > 95% of iterations |
| `test_logistic_regression_rejects_non_binary_labels` | Raises `ValueError` for multi-class labels |
| `test_logistic_regression_predict_threshold_changes_predictions` | Lower threshold yields at least as many positive predictions |
| `test_logistic_regression_alpha_must_be_nonnegative` | Raises `ValueError` for negative alpha |

### `test_knn.py`

Covers `KNeighborsClassifier` and `KNeighborsRegressor`:

| Test | What it checks |
|------|----------------|
| `test_knn_classifier_perfect_on_separated_blobs` | Accuracy > 0.95 on well-separated blobs |
| `test_knn_classifier_predict_proba_sums_to_one` | Probability rows sum to 1 |
| `test_knn_classifier_distance_weights_change_predictions` | Both weight modes achieve ≥ 0.85 accuracy |
| `test_knn_regressor_learns_linear_function` | Mean absolute error < 0.4 on a clean linear function |
| `test_knn_supports_alternative_metrics` | Euclidean, Manhattan, and Chebyshev all score > 0.85 |
| `test_knn_rejects_invalid_arguments` | Raises `ValueError` for bad `n_neighbors`, `weights`, or `metric` |
| `test_knn_predict_rejects_feature_mismatch` | Raises `ValueError` when predict input has wrong number of features |
| `test_knn_n_neighbors_too_large` | Raises `ValueError` when `n_neighbors` exceeds training set size |