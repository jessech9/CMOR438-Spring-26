# rice_ml — k-Nearest Neighbors, Random Forest, and DBSCAN

This branch implements three from-scratch algorithms — k-Nearest Neighbors,
Random Forest (with the underlying decision-tree backbone it builds on), and
DBSCAN — with example notebooks on real datasets and a test suite covering
every estimator.

---

## Source Code

### `supervised_learning/knn.py`

Distance-based classifier and regressor that share a private
`_KNeighborsBase` storing the training set on `fit()` and computing pairwise
distances at predict time.

**`KNeighborsClassifier`**
- Predicts class via majority vote among the `n_neighbors` closest training
  points
- `predict_proba` returns class probability rows that sum to 1
- Uniform or distance-based weighting (`weights={'uniform','distance'}`) with
  an epsilon so query=training-point ties are well-defined
- Attributes: `classes_`, `n_features_in_`

**`KNeighborsRegressor`**
- Predicts numeric target as the (optionally distance-weighted) mean of the
  `n_neighbors` neighbors' targets
- Same metric / weights options as the classifier
- Attributes: `n_features_in_`

Both estimators support `metric ∈ {'euclidean', 'manhattan', 'chebyshev'}`,
validate `n_neighbors > 0` and `≤ n_train`, and reject feature-count
mismatches at predict time.

---

### `supervised_learning/decision_tree.py`

CART trees implemented from scratch as the backbone Random Forest builds on.

**`DecisionTreeClassifier`**
- Greedy split selection minimising **Gini** or **entropy** (`criterion`)
- Supports `max_depth`, `min_samples_leaf`, and `max_features` (`int`,
  `float`, `'sqrt'`, or `'log2'`) — the random-subspace split selection that
  random forests rely on
- `predict_proba` rows are normalised class frequencies at the leaf
- Attributes: `classes_`

**`DecisionTreeRegressor`**
- Greedy splits minimising MSE
- Same `max_depth` / `min_samples_leaf` / `max_features` controls

The standalone Decision Tree algorithm slot in the team README (notebook +
write-up) is a different teammate's assignment — `decision_tree.py` itself
is here purely as RF's backbone.

---

### `supervised_learning/random_forest.py`

Bagged ensemble of decision trees with random feature subsets per split.

**`RandomForestClassifier`**
- Trains `n_estimators` `DecisionTreeClassifier` instances on bootstrap
  samples of the data, with split candidates restricted to a random subset
  of `max_features` at every node
- Soft-voting `predict_proba`: averages per-tree class probabilities,
  handling per-tree class subsets correctly
- Attributes: `estimators_`, `classes_`

**`RandomForestRegressor`**
- Same bagging + feature-subsampling recipe, averaging numeric tree
  predictions
- Attributes: `estimators_`

---

### `unsupervised_learning/dbscan.py`

Density-based clustering with the standard region-query / cluster-expansion
algorithm.

**`DBSCAN`**
- Sweeps every unvisited point: if its `eps`-ball contains at least
  `min_samples` neighbours it seeds a cluster and expands transitively
- `metric ∈ {'euclidean', 'manhattan'}`
- Validates `eps > 0`, `min_samples ≥ 1`, and known metric at `__init__`
- Attributes: `labels_` (with `-1` for noise points), `core_sample_indices_`

---

### Shared infrastructure

Also pushed to `main` from this contribution: `_base.py` (BaseEstimator,
Classifier/Regressor/ClusterMixin, `check_array` / `check_X_y`),
`processing/datasets.py` (`find_data_file`), `processing/pre_processing.py`
(StandardScaler, MinMaxScaler, LabelEncoder, train_test_split, one_hot_encode),
and `processing/post_processing.py` (accuracy, confusion matrix,
precision/recall/F1, ROC-AUC, MSE/RMSE/MAE, R², silhouette).

---

## Notebooks

### k-Nearest Neighbors — Crop Recommendation
**File:** `examples/supervised_learning/K Nearest Neighbors/knn.ipynb`

Multi-class classifier that predicts which of 22 crops fits a given soil
and climate profile from 7 numeric features (N, P, K, temperature, humidity,
pH, rainfall).

- Stratified 80/20 split keeps every class represented in both halves
- Quick EDA on per-feature spans illustrating why scaling is essential for
  distance-based methods (rainfall in millimetres would otherwise dwarf pH)
- Raw vs. `StandardScaler`-scaled baselines on the same `n_neighbors`
- `GridSearchCV` over `n_neighbors`, `weights`, and `metric`
- Test accuracy and a confusion-matrix heatmap

**Key findings:** Scaling alone closes most of the accuracy gap; tuning
`n_neighbors` past ~5 yields diminishing returns. Misclassifications
concentrate on crops with overlapping climate envelopes.

---

### Random Forest — Crop Recommendation
**File:** `examples/supervised_learning/Random Forest/random_forest.ipynb`

Forest of 25 from-scratch decision trees on the same Crop Recommendation
dataset, used to compare an ensemble against a single tree.

- KDE plots of `humidity` and `rainfall` split by class as visual EDA
- Single decision tree (`max_depth=8`) baseline vs. 25-tree forest
- Confusion-matrix heatmap and an sklearn classification report for per-class
  precision / recall / F1
- `n_estimators` learning curve over `[1, 5, 10, 25, 50]`
- Permutation feature importance (3 repeats per feature) since the
  from-scratch RF doesn't expose `feature_importances_`

**Key findings:** Test accuracy ≳ 0.99 — the forest beats the single tree
on stability and accuracy. The learning curve plateaus before 25 trees, so
a small forest is plenty here. Permutation importance ranks `humidity`,
`rainfall`, and the macro-nutrients (`K`, `N`) at the top; `ph` contributes
the least.

---

### DBSCAN — Customer Personality Analysis
**File:** `examples/unsupervised_learning/DBSCAN/dbscan.ipynb`

Density-based clustering on a 2240-customer marketing dataset
(`marketing_campaign.csv`).

- Outlier filtering on the raw dataset (a few obviously bogus rows)
- Engineered eight numeric features (age, income, total spend, total
  purchases, recency, web visits, kids/teens at home), then standardized
- DBSCAN on the standardized matrix; tuned `eps` via the k-distance plot
- Silhouette score on non-noise points
- 2-D PCA projection (sklearn `PCA`, since `rice_ml`'s PCA is a teammate's
  assignment) coloured by cluster, with noise points highlighted

**Key findings:** A small handful of well-separated clusters emerge once
spend features are log-friendly and standardized; ~5–10% of points fall to
noise depending on `eps`. The clusters interpret as low / mid / high
spenders with a small group of catalog-heavy outliers.

---

## Tests

### `test_knn.py`

Covers `KNeighborsClassifier` and `KNeighborsRegressor`:

| Test | What it checks |
|------|----------------|
| `test_knn_classifier_perfect_on_separated_blobs` | Accuracy = 1.0 on well-separated 2-D Gaussian blobs |
| `test_knn_classifier_predict_proba_sums_to_one` | Probability rows sum to 1 |
| `test_knn_classifier_distance_weights_change_predictions` | `weights='distance'` produces different predictions than `'uniform'` |
| `test_knn_regressor_learns_linear_function` | Recovers a clean linear function on noise-free synthetic data |
| `test_knn_supports_alternative_metrics` | Euclidean, Manhattan, and Chebyshev all classify the blobs correctly |
| `test_knn_rejects_invalid_arguments` | Raises `ValueError` for bad `n_neighbors`, `weights`, or `metric` |
| `test_knn_predict_rejects_feature_mismatch` | Raises `ValueError` when predict input has the wrong number of features |
| `test_knn_n_neighbors_too_large` | Raises `ValueError` when `n_neighbors` exceeds the training set size |

### `test_decision_tree.py`

Covers `DecisionTreeClassifier` and `DecisionTreeRegressor`:

| Test | What it checks |
|------|----------------|
| `test_decision_tree_classifier_can_overfit_training_data` | A deep tree reaches 100% training accuracy on a separable multi-class problem |
| `test_decision_tree_classifier_max_depth_limits_complexity` | A shallower tree's training accuracy is bounded below the deep tree's |
| `test_decision_tree_classifier_predict_proba_normalized` | `predict_proba` rows sum to 1 |
| `test_decision_tree_classifier_rejects_invalid_criterion` | Raises `ValueError` for unknown split criteria |
| `test_decision_tree_regressor_fits_step_function` | Recovers a piecewise-constant target with low MSE |
| `test_decision_tree_max_features_changes_chosen_feature` | Setting `max_features` actually restricts the search and can change the chosen split feature |

### `test_random_forest.py`

Covers `RandomForestClassifier` and `RandomForestRegressor`:

| Test | What it checks |
|------|----------------|
| `test_random_forest_classifier_beats_single_stump` | A 25-tree forest scores at least as well as a depth-1 stump on multi-class blobs |
| `test_random_forest_regressor_outperforms_single_tree` | A 20-tree forest matches or beats a single deep tree on a noisy sin curve |

### `test_dbscan.py`

Covers `DBSCAN`:

| Test | What it checks |
|------|----------------|
| `test_dbscan_separates_three_blobs` | Discovers exactly 3 clusters on the shared `cluster_data` fixture, with cluster purity > 0.9 |
| `test_dbscan_marks_isolated_points_as_noise` | An obvious outlier point is labeled `-1` |
| `test_dbscan_validates_inputs` | Raises `ValueError` for `eps ≤ 0` and `min_samples < 1` |
