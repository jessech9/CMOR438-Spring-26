# CMOR 438 — Spring 2026

**Authors:** Jesse Chen, Rahul Santhanam, Arhan Sankhla, Jason Swann
**Course:** CMOR 438 — Data Science and Machine Learning
**Term:** Spring 2026
**Institution:** Rice University

This repository is our final project for CMOR 438. It pairs:

1. **`rice_ml`** — a small, NumPy-only Python package implementing the
   classical machine-learning algorithms covered in the course from
   scratch.
2. **`examples/`** — one Jupyter notebook per algorithm, applied to a
   real dataset, alongside a README per algorithm explaining the
   intuition and the math.

Tests are run with `pytest`; CI runs the suite on every push and pull
request.

Each contributor also has a personal README at the repo root
(`README-jessech9.md`, `README-arhansa.md`, `README-rahulsanthanam0.md`)
documenting the modules / notebooks / tests they own.

---

## Repository layout

```
.
├── src/rice_ml/                                # the from-scratch ML package
│   ├── supervised_learning/
│   │   ├── linear.py                           # LinearRegression, LogisticRegression
│   │   ├── knn.py                              # KNeighborsClassifier, KNeighborsRegressor
│   │   ├── decision_tree.py                    # DecisionTreeClassifier, DecisionTreeRegressor
│   │   ├── random_forest.py                    # RandomForestClassifier, RandomForestRegressor
│   │   ├── gradient_boosting.py                # GradientBoostingClassifier
│   │   ├── ensemble_methods.py                 # back-compat re-exports of RF + GB
│   │   ├── perceptron.py                       # Perceptron
│   │   └── multilayer_perceptron.py            # MLPClassifier
│   ├── unsupervised_learning/
│   │   ├── dbscan.py                           # DBSCAN
│   │   ├── pca.py                              # PCA
│   │   ├── svd.py                              # SVD
│   │   └── k_means_clustering.py               # KMeans (placeholder)
│   ├── processing/
│   │   ├── pre_processing.py                   # StandardScaler, MinMaxScaler, train_test_split, ...
│   │   ├── post_processing.py                  # accuracy, R^2, ROC-AUC, silhouette, ...
│   │   └── datasets.py                         # find_data_file()
│   └── _base.py                                # BaseEstimator + mixins + check_array / check_X_y
│
├── data/                                       # CSVs read by the example notebooks
├── examples/
│   ├── supervised_learning/
│   │   ├── Decision Tree/                      ← Crop Recommendation
│   │   ├── Gradient Boosting/                  ← Banknote Authentication
│   │   ├── K Nearest Neighbors/                ← Crop Recommendation
│   │   ├── Linear Regression/                  ← Steel Industry Energy Consumption
│   │   ├── Logistic Regression/                ← Banknote Authentication
│   │   ├── Neural Network/                     ← Fetal Health
│   │   ├── Perceptron/                         ← Fetal Health
│   │   └── Random Forest/                      ← Crop Recommendation
│   └── unsupervised_learning/
│       ├── DBSCAN/                             ← Customer Personality Analysis
│       ├── K Means Clustering/                 ← Wholesale Customers
│       ├── PCA/                                ← Fetal Health
│       └── SVD/                                ← Wholesale Customers
│
├── tests/                                      # pytest suite, one file per algorithm
├── .github/workflows/tests.yml                 # CI: pytest on Python 3.10 / 3.11 / 3.12
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Algorithms

| Family                       | Module                                                       | Classes (exported via package `__init__`)                            |
| ---------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------- |
| Linear models                | `supervised_learning.linear`                                 | `LinearRegression`, `LogisticRegression`                             |
| Distance-based               | `supervised_learning.knn`                                    | `KNeighborsClassifier`, `KNeighborsRegressor`                        |
| Trees                        | `supervised_learning.decision_tree`                          | `DecisionTreeClassifier`, `DecisionTreeRegressor`                    |
| Ensembles                    | `supervised_learning.random_forest`, `…gradient_boosting`    | `RandomForest{Classifier,Regressor}`, `GradientBoostingClassifier`   |
| Neural networks              | `supervised_learning.perceptron`, `…multilayer_perceptron`   | `Perceptron`, `MLPClassifier` (importable directly from each module) |
| Clustering                   | `unsupervised_learning.dbscan`                               | `DBSCAN`                                                             |
| Dimensionality reduction     | `unsupervised_learning.pca`, `unsupervised_learning.svd`     | `PCA` (importable directly), `SVD`                                   |
| Pre-processing               | `processing.pre_processing`                                  | `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `train_test_split`, `one_hot_encode` |
| Post-processing (metrics)    | `processing.post_processing`                                 | `accuracy_score`, `confusion_matrix`, `precision_recall_f1`, `roc_auc_score`, `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`, `silhouette_score` |
| Dataset locator              | `processing.datasets`                                        | `find_data_file`                                                     |

Every estimator follows a consistent `(fit / predict / score)` contract
via `BaseEstimator`, `ClassifierMixin`, `RegressorMixin`, and
`ClusterMixin` in `rice_ml._base`.

---

## Quickstart

```bash
git clone https://github.com/jessech9/CMOR438-Spring-26.git
cd CMOR438-Spring-26
python -m pip install -e ".[dev,notebooks]"
```

Example use:

```python
from sklearn.datasets import load_breast_cancer
from rice_ml.supervised_learning.linear import LogisticRegression
from rice_ml.processing.pre_processing import StandardScaler, train_test_split
from rice_ml.processing.post_processing import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(alpha=0.01, max_iter=2000).fit(scaler.transform(X_tr), y_tr)
print(f"AUC = {roc_auc_score(y_te, clf.predict_proba(scaler.transform(X_te))[:, 1]):.3f}")
```

### Run the tests

```bash
pytest --cov=rice_ml
```

### Open the notebooks

```bash
jupyter lab examples/
```

All notebooks are checked in with executed outputs so figures and tables
render directly on GitHub.

---

## License

Released under the [MIT License](LICENSE).
