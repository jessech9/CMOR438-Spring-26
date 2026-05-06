# Supervised Learning

This folder collects eight notebooks demonstrating the supervised
algorithms implemented in `rice_ml`. Each algorithm has its own
folder with a brief `README.md` and a notebook.

## Index

| Algorithm           | Dataset                                | Folder                                                    |
| ------------------- | -------------------------------------- | --------------------------------------------------------- |
| Linear Regression   | Steel Industry Energy Consumption      | [`Linear Regression/`](Linear%20Regression/)              |
| Logistic Regression | Banknote Authentication                | [`Logistic Regression/`](Logistic%20Regression/)          |
| K Nearest Neighbors | Crop Recommendation                    | [`K Nearest Neighbors/`](K%20Nearest%20Neighbors/)        |
| Decision Tree       | Crop Recommendation                    | [`Decision Tree/`](Decision%20Tree/)                      |
| Random Forest       | Crop Recommendation                    | [`Random Forest/`](Random%20Forest/)                      |
| Gradient Boosting   | Banknote Authentication                | [`Gradient Boosting/`](Gradient%20Boosting/)              |
| Perceptron          | Fetal Health                           | [`Perceptron/`](Perceptron/)                              |
| Neural Network (MLP)| Fetal Health                           | [`Neural Network/`](Neural%20Network/)                    |

Each notebook reads its CSV from the repository [`data/`](../../data/)
folder via `find_data_file`.

## How supervised learning works

Supervised learning fits a function from a feature matrix `X` to a
target `y` using paired training examples. Two flavours:

- **Regression** — the target is continuous. Common metrics: MSE,
  RMSE, R².
- **Classification** — the target is discrete. Common metrics:
  accuracy, precision / recall / F1, ROC-AUC.

The algorithms in this folder span linear models (closed-form OLS,
gradient-descent logistic regression), distance-based methods (KNN),
tree-based methods (CART, random forest, gradient boosting), and
neural networks (single-layer perceptron, multi-layer perceptron).
