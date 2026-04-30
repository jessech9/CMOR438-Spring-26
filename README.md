# CMOR 438 — Spring 2026 Final Project

**Author:** Jesse Chen, Rahul Santhanam, Arhan Sankhla, Jason Swann
**Course:** CMOR 438 — Data Science and Machine Learning  
**Term:** Spring 2026  
**Institution:** Rice University

This final project for CMOR 438 – Data Science & Machine Learning applies a variety of machine learning techniques to real-world datasets. Through hands‑on Jupyter notebooks, we demonstrate data exploration, preprocessing, modeling, and evaluation workflows.

1. **`learnlab`** — a small, NumPy-only Python package that implements
   the classical machine-learning algorithms covered in the course from
   scratch.
2. **`examples/`** — a Jupyter notebook for each algorithm, applied to a
   real dataset, plus a README per algorithm explaining the intuition
   and the math.

Everything is version-controlled, unit-tested, and continuously
integrated via GitHub Actions.

---

## Repository layout

```
.
├── src/learnlab/                         # the from-scratch ML package
│   ├── supervised_learning/
│   │   ├── linear.py                     # OLS, Ridge, Lasso, Logistic Regression
│   │   ├── neighbors.py                  # KNeighbors{Classifier,Regressor}
│   │   ├── bayes.py                      # GaussianNB
│   │   ├── trees.py                      # DecisionTree{Classifier,Regressor}
│   │   ├── ensembles.py                  # Bagging, RandomForest, AdaBoost, GradientBoosting
│   │   └── neural.py                     # Perceptron, MLPClassifier
│   ├── unsupervised_learning/
│   │   ├── clustering.py                 # KMeans (++), DBSCAN, AgglomerativeClustering
│   │   ├── decomposition.py              # PCA (SVD-based)
│   │   └── graph.py                      # LabelPropagation
│   ├── processing/
│   │   ├── preprocessing.py              # StandardScaler, MinMaxScaler, train_test_split, ...
│   │   ├── metrics.py                    # accuracy, R^2, ROC-AUC, silhouette, ...
│   │   └── model_selection.py            # KFold, cross_val_score, GridSearchCV
│   ├── _base.py                          # BaseEstimator + mixins + validation helpers
│   └── README.md                         # full package API reference
│
├── examples/
│   ├── Supervised_Learning/              # ten algorithm folders, each with README + notebook
│   └── Unsupervised_Learning/            # five algorithm folders, each with README + notebook
│
├── tests/                                # pytest suite, one test_<module>.py per module
├── scripts/build_notebooks.py            # source-of-truth for every notebook
├── .github/workflows/tests.yml           # CI: pytest on Python 3.10 / 3.11 / 3.12
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Algorithms implemented

| Family                       | Module                                       | Classes                                                                                |
| ---------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| Linear models                | `supervised_learning.linear`                 | `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`                             |
| Distance-based               | `supervised_learning.neighbors`              | `KNeighborsClassifier`, `KNeighborsRegressor`                                          |
| Probabilistic                | `supervised_learning.bayes`                  | `GaussianNB`                                                                           |
| Trees                        | `supervised_learning.trees`                  | `DecisionTreeClassifier`, `DecisionTreeRegressor`                                      |
| Ensembles                    | `supervised_learning.ensembles`              | `BaggingClassifier`, `RandomForestClassifier/Regressor`, `AdaBoostClassifier`, `GradientBoostingRegressor` |
| Neural networks              | `supervised_learning.neural`                 | `Perceptron`, `MLPClassifier` (multi-class softmax + cross-entropy)                    |
| Clustering                   | `unsupervised_learning.clustering`           | `KMeans` (k-means++), `DBSCAN`, `AgglomerativeClustering`                              |
| Dimensionality reduction     | `unsupervised_learning.decomposition`        | `PCA` (SVD-based)                                                                      |
| Graph / community detection  | `unsupervised_learning.graph`                | `LabelPropagation`                                                                     |
| Preprocessing                | `processing.preprocessing`                   | `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `train_test_split`, `one_hot_encode` |
| Metrics                      | `processing.metrics`                         | `accuracy_score`, `confusion_matrix`, `precision_recall_f1`, `roc_auc_score`, `mean_squared_error`, `r2_score`, `silhouette_score`, ... |
| Model selection              | `processing.model_selection`                 | `KFold`, `cross_val_score`, `GridSearchCV`                                             |

Every estimator follows a consistent `(fit / predict / score)` contract
via `BaseEstimator`, `ClassifierMixin`, and `RegressorMixin` in
`learnlab._base`.

---

## Notebooks at a glance

| Algorithm                     | Folder                                                                                              | Dataset                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------- |
| Linear Regression             | [`Supervised_Learning/Linear_Regression/`](examples/Supervised_Learning/Linear_Regression/)         | California housing            |
| Ridge & Lasso                 | [`Supervised_Learning/Ridge_and_Lasso/`](examples/Supervised_Learning/Ridge_and_Lasso/)             | Diabetes                      |
| Logistic Regression           | [`Supervised_Learning/Logistic_Regression/`](examples/Supervised_Learning/Logistic_Regression/)     | Wisconsin Breast Cancer       |
| k-Nearest Neighbors           | [`Supervised_Learning/KNN/`](examples/Supervised_Learning/KNN/)                                     | Wine recognition              |
| Naive Bayes                   | [`Supervised_Learning/Naive_Bayes/`](examples/Supervised_Learning/Naive_Bayes/)                     | Wine recognition              |
| Decision Tree                 | [`Supervised_Learning/Decision_Trees/`](examples/Supervised_Learning/Decision_Trees/)               | Palmer Penguins               |
| Regression Tree               | [`Supervised_Learning/Regression_Trees/`](examples/Supervised_Learning/Regression_Trees/)           | Synthetic non-linear curve    |
| Ensemble Methods              | [`Supervised_Learning/Ensemble_Methods/`](examples/Supervised_Learning/Ensemble_Methods/)           | Synthetic + Friedman #1       |
| Perceptron                    | [`Supervised_Learning/Perceptron/`](examples/Supervised_Learning/Perceptron/)                       | Blobs + XOR                   |
| Neural Network (MLP)          | [`Supervised_Learning/Neural_Networks/`](examples/Supervised_Learning/Neural_Networks/)             | scikit-learn 8×8 digits       |
| K-Means                       | [`Unsupervised_Learning/K_Means_Clustering/`](examples/Unsupervised_Learning/K_Means_Clustering/)   | Synthetic mall-customer blobs |
| DBSCAN                        | [`Unsupervised_Learning/DBSCAN/`](examples/Unsupervised_Learning/DBSCAN/)                           | Half-moons                    |
| Agglomerative Clustering      | [`Unsupervised_Learning/Agglomerative_Clustering/`](examples/Unsupervised_Learning/Agglomerative_Clustering/) | Hand-crafted "zoo" features |
| PCA                           | [`Unsupervised_Learning/PCA/`](examples/Unsupervised_Learning/PCA/)                                 | 8×8 digits                    |
| Community Detection           | [`Unsupervised_Learning/Community_Detection/`](examples/Unsupervised_Learning/Community_Detection/) | Zachary's karate club         |

All notebooks are pre-executed so the figures and outputs render
directly on GitHub.

---

## Quickstart

```bash
git clone https://github.com/jessech9/CMOR438-Spring-2026.git
cd CMOR438-Spring-2026
python -m pip install -e .[dev,notebooks]
```

```python
from sklearn.datasets import load_breast_cancer
from learnlab.supervised_learning.linear import LogisticRegression
from learnlab.processing.preprocessing import StandardScaler, train_test_split
from learnlab.processing.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(alpha=0.01, max_iter=2000).fit(scaler.transform(X_tr), y_tr)
print(f"AUC = {roc_auc_score(y_te, clf.predict_proba(scaler.transform(X_te))[:, 1]):.3f}")
```

### Run the tests

```bash
pytest --cov=learnlab
```

The CI workflow runs the same command on Python 3.10, 3.11, and 3.12 on
every push and pull request.

### Open the notebooks

```bash
jupyter lab examples/
```

Or browse the rendered versions directly on GitHub.

---

## License

Released under the [MIT License](LICENSE).
