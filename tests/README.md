# Tests

The `tests/` folder is a `pytest` suite that exercises every algorithm
in `rice_ml`. Each `test_<algorithm>.py` file pairs with the module of
the same name in `src/rice_ml/`.

## Running

From the repository root:

```bash
# install the dev extras, which include pytest + pytest-cov
pip install -e ".[dev]"

pytest                       # run everything (fast)
pytest -k knn                # run a single test file by keyword
pytest --cov=rice_ml         # with coverage
```

The CI workflow at `.github/workflows/tests.yml` runs
`pytest --cov=rice_ml` on Python 3.10, 3.11, and 3.12 for every push
and pull request.

## Structure

Shared fixtures (random seed, regression data, classification blobs,
multi-class blobs, cluster blobs) live in [`conftest.py`](conftest.py)
so individual tests stay short and focused.

| File                               | Covers                                                  |
| ---------------------------------- | ------------------------------------------------------- |
| `test_linear.py`                   | `LinearRegression`, `LogisticRegression`                |
| `test_knn.py`                      | `KNeighborsClassifier`, `KNeighborsRegressor`           |
| `test_decision_tree.py`            | `DecisionTreeClassifier`, `DecisionTreeRegressor`       |
| `test_random_forest.py`            | `RandomForestClassifier`, `RandomForestRegressor`       |
| `test_gradient_boosting.py`        | `GradientBoostingClassifier`                            |
| `test_perceptron.py`               | `Perceptron`                                            |
| `test_multilayer_perceptron.py`    | `MLPClassifier`                                         |
| `test_dbscan.py`                   | `DBSCAN`                                                |
| `test_pca.py`                      | `PCA`                                                   |
| `test_svd.py`                      | `SVD`                                                   |

Each file mixes happy-path correctness tests, shape/validation tests,
and algorithmic-property tests (loss decreases, RF beats single tree,
PCA recovers the principal axis, …).
