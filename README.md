# CMOR 438 — Spring 2026

**Author:** Jesse Chen, Rahul Santhanam, Arhan Sankhla, Jason Swann  
**Course:** CMOR 438 — Data Science and Machine Learning  
**Term:** Spring 2026  
**Institution:** Rice University

This repository is my final project for CMOR 438. It pairs:

1. **`rice_ml`** — a small, NumPy-only Python package implementing the
   classical machine-learning algorithms covered in the course from
   scratch.
2. **`examples/`** — one Jupyter notebook per algorithm, applied to a
   real dataset, alongside a README per algorithm explaining the
   intuition and the math.

Tests are run with `pytest`; CI runs the suite on every push and pull
request.

---

## Repository layout

```
.
├── src/rice_ml/                           # the from-scratch ML package
│   ├── supervised_learning/
│   │   ├── linear.py                      # LinearRegression, LogisticRegression
│   │   ├── neighbors.py                   # KNeighbors{Classifier,Regressor}
│   │   ├── trees.py                       # DecisionTree{Classifier,Regressor}
│   │   ├── ensembles.py                   # RandomForest, GradientBoostingRegressor
│   │   └── neural.py                      # Perceptron, MLPClassifier
│   ├── unsupervised_learning/
│   │   ├── clustering.py                  # KMeans (k-means++), DBSCAN
│   │   └── decomposition.py               # PCA, SVD
│   ├── processing/
│   │   ├── preprocessing.py               # StandardScaler, MinMaxScaler, train_test_split, ...
│   │   ├── metrics.py                     # accuracy, R^2, ROC-AUC, silhouette, ...
│   │   └── model_selection.py             # KFold, cross_val_score, GridSearchCV
│   └── _base.py                           # BaseEstimator + mixins + check_array / check_X_y
│
├── data/                                  # CSVs read by the example notebooks
├── examples/
│   ├── supervised_ml/
│   │   ├── Decision Tree/                 ← Crop Recommendation
│   │   ├── Gradient Boosting/             ← Credit Card Fraud (pending)
│   │   ├── K Means Clustering/            ← Spotify Tracks (pending)
│   │   ├── K Nearest Neighbors/           ← Crop Recommendation
│   │   ├── Linear Regression/             ← Steel Industry Energy Consumption
│   │   ├── Logistic Regression/           ← Credit Card Fraud (pending)
│   │   ├── Neural Network/                ← Fashion MNIST (pending)
│   │   ├── Perceptron/                    ← Fashion MNIST (pending)
│   │   └── Random Forest/                 ← Crop Recommendation
│   └── unsupervised_ml/
│       ├── DBSCAN/                        ← Customer Personality Analysis
│       ├── K Means Clustering/            ← Spotify Tracks (pending)
│       ├── PCA/                           ← Fashion MNIST (pending)
│       └── SVD/                           ← Spotify Tracks (pending)
│
├── tests/                                 # pytest suite, one test file per module
├── scripts/build_notebooks.py             # source-of-truth for every notebook
├── .github/workflows/tests.yml            # CI: pytest on Python 3.10 / 3.11 / 3.12
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Algorithms

| Family                       | Module                                       | Classes                                                                                |
| ---------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| Linear models                | `supervised_learning.linear`                 | `LinearRegression`, `LogisticRegression`                                               |
| Distance-based               | `supervised_learning.neighbors`              | `KNeighborsClassifier`, `KNeighborsRegressor`                                          |
| Trees                        | `supervised_learning.trees`                  | `DecisionTreeClassifier`, `DecisionTreeRegressor`                                      |
| Ensembles                    | `supervised_learning.ensembles`              | `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingRegressor`         |
| Neural networks              | `supervised_learning.neural`                 | `Perceptron`, `MLPClassifier`                                                          |
| Clustering                   | `unsupervised_learning.clustering`           | `KMeans` (with k-means++), `DBSCAN`                                                    |
| Dimensionality reduction     | `unsupervised_learning.decomposition`        | `PCA`, `SVD`                                                                           |
| Preprocessing                | `processing.preprocessing`                   | `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `train_test_split`, `one_hot_encode` |
| Metrics                      | `processing.metrics`                         | `accuracy_score`, `confusion_matrix`, `precision_recall_f1`, `roc_auc_score`, `mean_squared_error`, `r2_score`, `silhouette_score`, ... |
| Model selection              | `processing.model_selection`                 | `KFold`, `cross_val_score`, `GridSearchCV`                                             |

Every estimator follows a consistent `(fit / predict / score)` contract
via `BaseEstimator`, `ClassifierMixin`, and `RegressorMixin` in
`rice_ml._base`.

---

## Quickstart

```bash
git clone https://github.com/jessech9/CMOR438-Spring-2026.git
cd CMOR438-Spring-2026
python -m pip install -e .[dev,notebooks]
```

Example use:

```python
from sklearn.datasets import load_breast_cancer
from rice_ml.supervised_learning.linear import LogisticRegression
from rice_ml.processing.preprocessing import StandardScaler, train_test_split
from rice_ml.processing.metrics import roc_auc_score

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

All notebooks are pre-executed so figures and outputs render directly
on GitHub.

---

## License

Released under the [MIT License](LICENSE).