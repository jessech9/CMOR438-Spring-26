# Arhan Sankhla — contributions to `rice_ml`

This is my personal working branch. From this point forward, all of my work
goes here first and gets merged into `main` through a pull request, instead of
pushed directly.

## What's already on `main` from me

### `0044386` — Add 3 assigned algorithms (KNN / Random Forest / DBSCAN)

My three assigned algorithms (k-Nearest Neighbors, Random Forest, DBSCAN), plus
the shared package infrastructure they build on.

**Algorithms (`src/rice_ml/`)**

| Module                                 | Classes                                                         |
| -------------------------------------- | --------------------------------------------------------------- |
| `supervised_learning/knn.py`           | `KNeighborsClassifier`, `KNeighborsRegressor`                   |
| `supervised_learning/decision_tree.py` | `DecisionTreeClassifier`, `DecisionTreeRegressor` (RF backbone) |
| `supervised_learning/random_forest.py` | `RandomForestClassifier`, `RandomForestRegressor`               |
| `unsupervised_learning/dbscan.py`      | `DBSCAN`                                                        |

The standalone Decision Tree algorithm slot in the team README (notebook +
write-up) is a different teammate's assignment — `decision_tree.py` is here
purely as the backbone Random Forest builds on.

**Shared infrastructure (`src/rice_ml/`)**

- `_base.py` — `BaseEstimator` (param introspection / `repr` / fit-check),
  `ClassifierMixin` / `RegressorMixin` / `ClusterMixin` providing `.score()`,
  `check_array` and `check_X_y` validation helpers.
- `processing/datasets.py` — `find_data_file` walks upward from the cwd to
  locate CSVs in `data/`, so notebooks run from any subdirectory.
- `processing/pre_processing.py` — `StandardScaler`, `MinMaxScaler`,
  `LabelEncoder`, `train_test_split` (with optional stratify),
  `one_hot_encode`.
- `processing/post_processing.py` — `accuracy_score`, `confusion_matrix`,
  `precision_recall_f1`, `roc_auc_score`, `mean_squared_error`,
  `root_mean_squared_error`, `mean_absolute_error`, `r2_score`,
  `silhouette_score`.

**Tests (`tests/unit/`)**

`test_knn.py`, `test_decision_tree.py`, `test_random_forest.py`,
`test_dbscan.py`, plus shared fixtures in `conftest.py` (`rng`,
`binary_classification_data`, `multiclass_data`, `cluster_data`).

**Notebooks (`examples/`)**

- `supervised_ml/K Nearest Neighbors/knn.ipynb` — Crop Recommendation,
  with a section on why feature scaling matters for distance-based methods.
- `supervised_ml/Random Forest/random_forest.ipynb` — Crop Recommendation,
  with feature-importance bars and an `n_estimators` sweep.
- `unsupervised_ml/DBSCAN/dbscan.ipynb` — Customer Personality
  (`marketing_campaign.csv`), with `eps` tuning via the k-distance plot,
  silhouette scoring on non-noise points, and a 2D PCA scatter (PCA from
  `sklearn.decomposition` since `rice_ml`'s PCA is a teammate's assignment).

**Datasets (`data/`)**

- `Crop_recommendation.csv` (used by both KNN and Random Forest notebooks).
- `marketing_campaign.csv` (used by the DBSCAN notebook).

**Project plumbing**

- `pyproject.toml` (setuptools src layout, pytest discovery).
- `requirements.txt`.
- `.github/workflows/tests.yml` — pytest matrix on Python 3.10 / 3.11 / 3.12.
- `.gitignore`.

## What's pending on this branch (not yet on `main`)

### `fixing repo folder structure`

Cleaning up the layout so it matches the team-standard naming:

- Moved `examples/supervised_ml/K Nearest Neighbors/` and
  `examples/supervised_ml/Random Forest/` into the team-standard
  `examples/supervised_learning/`. Removed the now-empty `supervised_ml/`.
- Removed the duplicate `examples/unsupervised_ml/DBSCAN/` (Jesse had
  already copied DBSCAN over to `examples/unsupervised_learning/`).
- Pulled `tests/unit/test_*.py` up into `tests/` and deleted `tests/unit/`,
  so Jesse's `tests/test_linear.py` and the rest of the suite live side by
  side.
- Merged `tests/unit/conftest.py` into `tests/conftest.py`, adding a
  `regression_data` fixture so `test_linear.py` has the shared fixture it
  needs.
- Updated `pyproject.toml` so `testpaths = ["tests"]` (was `tests/unit`).

This will go in via the PR that merges this branch.

## Going forward

All future changes from me land here on `arhansa` first; PR into `main` for
review.
