# rice_ml — Perceptron, Gradient Boosting, and SVD

This branch implements three from-scratch algorithms — Perceptron,
Gradient Boosting, and Singular Value Decomposition (SVD) — with example
notebooks on real datasets and tests covering each estimator.

---

## Source Code

### `supervised_learning/perceptron.py`

Rosenblatt's binary perceptron implemented as a sklearn-style classifier.

**`Perceptron`**
- Learns a single linear decision boundary with the update rule
  `w <- w + eta * (y - y_hat) * x`
- Supports `learning_rate`, `max_iter`, `shuffle`, `fit_intercept`, and
  `random_state`
- Stops early when an epoch has zero mistakes, or after `max_iter`
- Exposes `decision_function()` for raw signed distance-style scores
- Attributes: `coef_`, `intercept_`, `n_iter_`, `classes_`
- Validates binary `{0, 1}` targets and rejects invalid hyperparameters

---

### `supervised_learning/gradient_boosting.py`

Binary gradient boosting classifier built from shallow regression trees.

**`GradientBoostingClassifier`**
- Initializes the ensemble with the log-odds of the positive class
- Fits each shallow `DecisionTreeRegressor` to Bernoulli pseudo-residuals
  `y - sigmoid(F)`
- Updates the raw log-odds score stage by stage with `learning_rate`
- Supports `n_estimators`, `learning_rate`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, and `random_state`
- Provides `decision_function()`, `predict_proba()`, `predict()`, and
  `score()`
- Tracks `feature_importances_` by counting tree split usage
- Validates binary targets and rejects invalid boosting parameters

Compatibility import shims were also added for older notebook paths:

- `supervised_learning/decision_trees.py`
- `supervised_learning/ensemble_methods.py`

---

### `unsupervised_learning/svd.py`

Truncated Singular Value Decomposition transformer wrapping
`numpy.linalg.svd` with deterministic sign handling.

**`SVD`**
- Factorizes an input matrix as `X approx U_k Sigma_k V_k.T`
- Supports configurable `n_components`
- Provides `fit()`, `transform()`, `fit_transform()`, and
  `inverse_transform()`
- Exposes `components_`, `singular_values_`,
  `explained_variance_`, and `explained_variance_ratio_`
- Uses a deterministic sign flip so repeated runs have stable component
  orientation
- Rejects non-positive `n_components`

---

## Notebooks

### Perceptron — Fetal Health Classification
**File:** `examples/supervised_learning/Perceptron/perceptron.ipynb`

Binary classifier on `fetal_health.csv`, merging the original labels into
normal fetal health vs. suspect/pathological fetal health.

- Loads the fetal health dataset with `find_data_file`
- Converts the original 3-class target into a binary target
- Uses stratified train/test splitting and `StandardScaler`
- Trains the from-scratch `Perceptron`
- Reports accuracy, precision, recall, F1, and a confusion matrix
- Adds Gradient-Boosting-style visuals: class distribution, scatter matrix,
  feature scatter, boxplots, decision-score histogram, and coefficient plot
- Preserves the XOR failure case to demonstrate the perceptron's linear
  limitation

**Key findings:** The perceptron reaches solid overall accuracy on the
binary fetal-health task, but recall for the adverse class is lower than
accuracy suggests. The decision-score histogram and XOR example highlight
the core limitation: one linear boundary cannot capture all nonlinear
structure.

---

### Gradient Boosting — Banknote Authentication
**File:** `examples/supervised_learning/Gradient Boosting/gradient_boosting.ipynb`

Binary classifier on `BankNote_Authentication.csv`, predicting whether a
banknote is authentic or forged from four wavelet-transform features.

- Loads and checks the banknote dataset
- Uses stratified train/test splitting and `StandardScaler`
- Trains the from-scratch `GradientBoostingClassifier`
- Reports accuracy, ROC-AUC, precision, recall, F1, and a confusion matrix
- Includes visual diagnostics: class distribution, scatter matrix,
  two-feature scatter, boxplots, probability histogram, ROC curve, and
  feature-importance bar chart
- Uses KNN-style markdown structure with numbered sections and an
  auto-generated results summary

**Key findings:** A small boosted ensemble performs strongly on the
banknote data, with high ROC-AUC and clean probability separation. Split
frequency importance ranks the wavelet statistics used most often by the
boosted trees.

---

### Singular Value Decomposition — Wholesale Customers
**File:** `examples/unsupervised_learning/SVD/svd.ipynb`

Low-rank matrix factorization on `Wholesale_customers_data.csv`, where each
row is a customer and each column is annual spending in a product category.

- Loads and checks channel / region counts
- Explores skewed spending with `log1p` histograms, scatter matrices,
  channel-colored scatter plots, and boxplots
- Scales the six spending columns before factorization
- Fits the from-scratch `SVD` transformer
- Plots explained variance, cumulative explained variance, and
  reconstruction error by component count
- Visualizes the scaled matrix, rank-2 reconstruction, and residuals
- Plots customers in rank-2 latent space and displays component loadings

**Key findings:** The first two SVD components capture most of the scaled
matrix energy and reveal clear low-dimensional spending structure. The
latent customer map can be compared with `Channel` post-hoc, while the
loading plot shows which spending categories define the leading directions.

---

## README Writeups

- `examples/supervised_learning/Perceptron/README.md` — perceptron math,
  update rule, convergence theorem, and XOR limitation.
- `examples/supervised_learning/Gradient Boosting/README.md` — binary
  gradient boosting overview, hyperparameters, data requirements, notebook
  link, and source implementation link.
- `examples/unsupervised_learning/SVD/README.md` — SVD factorization,
  Eckart-Young theorem, use cases, and notebook summary.

---

## Tests

### `test_perceptron.py`

Covers `Perceptron`:

| Test | What it checks |
|------|----------------|
| `test_perceptron_learns_linearly_separable` | Fits cleanly separable data with high accuracy |
| `test_perceptron_converges_early_on_separable_data` | Stops before `max_iter` on separable data |
| `test_perceptron_rejects_multiclass_labels` | Raises `ValueError` for non-binary targets |
| `test_perceptron_rejects_invalid_learning_rate` | Raises `ValueError` for non-positive learning rate |
| `test_perceptron_rejects_invalid_max_iter` | Raises `ValueError` for `max_iter < 1` |
| `test_perceptron_cannot_solve_xor` | Demonstrates failure on nonlinear XOR |
| `test_perceptron_no_intercept` | `fit_intercept=False` sets intercept to zero |
| `test_perceptron_coef_shape` | One coefficient per feature |
| `test_perceptron_predict_values` | Predictions are only 0 or 1 |

### `test_gradient_boosting.py`

Covers `GradientBoostingClassifier`:

| Test | What it checks |
|------|----------------|
| `test_gradient_boosting_classifier_fits_separable_binary_data` | Fits separable binary data with high accuracy |
| `test_gradient_boosting_predict_proba_shape_and_sum` | Probability output has shape `(n, 2)` and rows sum to 1 |
| `test_gradient_boosting_feature_importances_sum_to_one` | Feature importances have one value per feature and sum to 1 |
| `test_gradient_boosting_rejects_multiclass_targets` | Raises `ValueError` for non-binary targets |
| `test_gradient_boosting_rejects_invalid_parameters` | Raises `ValueError` for invalid `n_estimators` or `learning_rate` |

### `test_svd.py`

Covers `SVD`:

| Test | What it checks |
|------|----------------|
| `test_svd_factorization_reconstructs_data` | Full-rank transform and inverse transform reconstruct data |
| `test_svd_top_components_are_orthonormal` | Learned components are orthonormal |
| `test_svd_singular_values_match_numpy` | Singular values match `numpy.linalg.svd` |
| `test_svd_rejects_invalid_n_components` | Raises `ValueError` for non-positive `n_components` |
