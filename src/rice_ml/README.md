# `rice_ml` — Package Reference

`rice_ml` is the from-scratch machine-learning package for the
CMOR 438 final project. Everything in this directory is implemented
with NumPy only (plus pandas / matplotlib / seaborn for utilities and
plotting in the example notebooks).

For dataset demos, see the [`examples/`](../../examples) folder.

## Subpackages

```
rice_ml/
├── supervised_learning/
│   ├── linear.py                — LinearRegression, LogisticRegression
│   ├── knn.py                   — KNeighborsClassifier, KNeighborsRegressor
│   ├── decision_tree.py         — DecisionTreeClassifier, DecisionTreeRegressor
│   ├── random_forest.py         — RandomForestClassifier, RandomForestRegressor
│   ├── gradient_boosting.py     — GradientBoostingClassifier
│   ├── ensemble_methods.py      — back-compat re-exports of RF + GB
│   ├── perceptron.py            — Perceptron
│   └── multilayer_perceptron.py — MLPClassifier
├── unsupervised_learning/
│   ├── dbscan.py                — DBSCAN
│   ├── pca.py                   — PCA
│   ├── svd.py                   — SVD
│   └── k_means_clustering.py    — KMeans (placeholder)
├── processing/
│   ├── pre_processing.py        — StandardScaler, MinMaxScaler, LabelEncoder, train_test_split, one_hot_encode
│   ├── post_processing.py       — accuracy, R^2, ROC-AUC, silhouette, confusion_matrix, ...
│   └── datasets.py              — find_data_file()
└── _base.py                     — BaseEstimator + mixins + check_array / check_X_y
```

## Public API

```python
# Supervised learning
from rice_ml.supervised_learning.linear import LinearRegression, LogisticRegression
from rice_ml.supervised_learning.knn import (
    KNeighborsClassifier, KNeighborsRegressor,
)
from rice_ml.supervised_learning.decision_tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
)
from rice_ml.supervised_learning.random_forest import (
    RandomForestClassifier, RandomForestRegressor,
)
from rice_ml.supervised_learning.gradient_boosting import GradientBoostingClassifier
from rice_ml.supervised_learning.perceptron import Perceptron
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier

# Unsupervised learning
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.unsupervised_learning.svd import SVD

# Processing
from rice_ml.processing.pre_processing import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    train_test_split, one_hot_encode,
)
from rice_ml.processing.post_processing import (
    accuracy_score, confusion_matrix, precision_recall_f1, roc_auc_score,
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    r2_score, silhouette_score,
)
from rice_ml.processing.datasets import find_data_file
```

The most common supervised classes are also re-exported from
`rice_ml.supervised_learning` itself, and DBSCAN / SVD from
`rice_ml.unsupervised_learning`, so shorter import paths work too.

## Estimator contract

Every estimator implements `fit(X, y) -> self`. Classifiers expose
`predict(X)` and (where probabilistic) `predict_proba(X)`; regressors
expose `predict(X)`; clusterers store fitted labels in `labels_` and
expose `fit_predict(X)`.

The base classes in [`_base.py`](_base.py) provide:

| Symbol              | Purpose                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `BaseEstimator`     | parameter introspection (`get_params` / `set_params`) and `__repr__`     |
| `ClassifierMixin`   | provides `score` = accuracy                                              |
| `RegressorMixin`    | provides `score` = R²                                                    |
| `ClusterMixin`      | provides `fit_predict`                                                   |
| `NotFittedError`    | raised when methods are called before `fit`                              |
| `check_array`       | validate / coerce 2-D input (NaN / inf / shape checks)                   |
| `check_X_y`         | validate consistent `(X, y)` pair                                        |

## Testing

Run the test suite from the repository root:

```bash
pytest --cov=rice_ml
```
