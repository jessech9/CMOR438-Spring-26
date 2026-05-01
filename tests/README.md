# Tests

The `tests/` folder is a `pytest` suite that exercises every module in
`rice_ml`. Each `test_<module>.py` file pairs with the module of the
same name in `src/rice_ml/`.

## Running

From the repository root:

```bash
# install the dev extras, which include pytest + pytest-cov
pip install -e .[dev]

pytest                       # run everything (fast)
pytest -k logistic           # run a single test file by keyword
pytest --cov=rice_ml         # with coverage
```

The CI workflow at `.github/workflows/tests.yml` runs
`pytest --cov=rice_ml` on Python 3.10, 3.11, and 3.12 for every push
and pull request.

## Structure

Shared fixtures (random seed, regression data, blobs, multi-class
data, etc.) live in [`conftest.py`](conftest.py) so individual tests
stay short and focused.

| File                          | Covers                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| `test_base.py`                | `BaseEstimator`, validation helpers                             |
| `test_metrics.py`             | classification, regression, and cluster metrics                 |
| `test_preprocessing.py`       | scalers, label encoders, train/test split                       |
| `test_linear.py`              | LinearRegression, LogisticRegression                            |
| `test_neighbors.py`           | KNeighbors classifier and regressor                             |
| `test_trees.py`               | DecisionTreeClassifier, DecisionTreeRegressor                   |
| `test_ensembles.py`           | RandomForest classifier/regressor, GradientBoostingRegressor    |
| `test_neural.py`              | Perceptron and MLPClassifier                                    |
| `test_clustering.py`          | KMeans (with k-means++), DBSCAN                                 |
| `test_decomposition.py`       | PCA, SVD                                                        |
| `test_model_selection.py`     | KFold, cross_val_score, GridSearchCV                            |

Each file mixes happy-path correctness tests, shape/validation tests,
and algorithmic-property tests (loss decreases, PCA recovers the
principal axis, …).