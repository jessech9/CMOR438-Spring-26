# `data/`

Notebooks in `examples/` read their input data from this folder. Files
here are not tracked individually in the package — they are dropped in
locally and committed when small enough.

## Contents

| File                          | Used by                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Crop_recommendation.csv`     | Decision Tree, K Nearest Neighbors, Random Forest                                                                      |
| `Steel_industry_data.csv`     | Linear Regression                                                                                                      |
| `marketing_campaign.csv`      | DBSCAN (Customer Personality Analysis)                                                                                 |
| `BankNote_Authentication.csv`  | Gradient Boosting, Logistic Regression                                                                                 |
| `fetal_health.csv`             | Neural Network, Perceptron, PCA (unsupervised)                                                                         |
| `Wholesale_customers_data.csv` | Supervised K Means Clustering, Unsupervised K Means Clustering, SVD                                                    |

## How notebooks find these files

Each notebook calls
[`rice_ml.processing.datasets.find_data_file`](../src/rice_ml/processing/datasets.py),
which walks up from the notebook's working directory looking for a
`data/<filename>` folder. That means notebooks work the same whether
you run them from the repository root, from JupyterLab, or from CI.