# `rice_ml` — Package Reference

`rice_ml` is the from-scratch machine-learning package for the
CMOR 438 final project. Everything in this directory is implemented
with NumPy only (plus pandas / matplotlib for utilities and plotting).

For dataset demos, see the [`examples/`](../../examples) folder.

## Subpackages

```
rice_ml/
├── supervised_learning/
│   ├── linear        — LinearRegression, LogisticRegression
│   ├── neighbors     — KNeighborsClassifier, KNeighborsRegressor
│   ├── trees         — DecisionTreeClassifier, DecisionTreeRegressor
│   ├── ensembles     — RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor
│   └── neural        — Perceptron, MLPClassifier
├── unsupervised_learning/
│   ├── clustering    — KMeans (k-means++), DBSCAN
│   └── decomposition — PCA, SVD
├── processing/
│   ├── preprocessing  — StandardScaler, MinMaxScaler, LabelEncoder, train_test_split, one_hot_encode
│   ├── metrics        — accuracy, R^2, ROC-AUC, silhouette, confusion_matrix, ...
│   └── model_selection — KFold, cross_val_score, GridSearchCV
└── _base.py            — BaseEstimator + mixins + check_array / check_X_y
```

## Public API

```python
# Supervised learning
from rice_ml.supervised_learning.linear import LinearRegression, LogisticRegression
from rice_ml.supervised_learning.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
)
from rice_ml.supervised_learning.trees import (
    DecisionTreeClassifier, DecisionTreeRegressor,
)
from rice_ml.supervised_learning.ensembles import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from rice_ml.supervised_learning.neural import Perceptron, MLPClassifier

# Unsupervised learning
from rice_ml.unsupervised_learning.clustering import KMeans, DBSCAN
from rice_ml.unsupervised_learning.decomposition import PCA, SVD

# Processing
from rice_ml.processing.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    train_test_split, one_hot_encode,
)
from rice_ml.processing.metrics import (
    accuracy_score, confusion_matrix, precision_recall_f1, roc_auc_score,
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    r2_score, silhouette_score,
)
from rice_ml.processing.model_selection import (
    KFold, cross_val_score, GridSearchCV,
)
```

For convenience, the most common helpers are also re-exported at the
top level so that `from rice_ml.metrics import ...`,
`from rice_ml.preprocessing import ...`, and
`from rice_ml.model_selection import ...` keep working.

Every estimator implements `fit(X, y) -> self`. Classifiers expose
`predict(X)` and (where probabilistic) `predict_proba(X)`; regressors
expose `predict(X)`; clusterers store fitted labels in `labels_`.

The base classes in [`_base.py`](_base.py) provide:

| Symbol              | Purpose                                                           |
| ------------------- | ----------------------------------------------------------------- |
| `BaseEstimator`     | parameter introspection (`get_params` / `set_params`) and `__repr__` |
| `ClassifierMixin`   | provides `score` = accuracy                                       |
| `RegressorMixin`    | provides `score` = R²                                             |
| `ClusterMixin`      | provides `fit_predict`                                            |
| `NotFittedError`    | raised when methods are called before `fit`                       |
| `check_array`       | validate / coerce 2-D input                                       |
| `check_X_y`         | validate consistent `(X, y)` pair                                 |

## Testing

Run the test suite from the repository root:

```bash
pytest --cov=rice_ml
```