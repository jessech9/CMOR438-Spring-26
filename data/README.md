# `data/`

Notebooks in `examples/` read their input data from this folder.

## Contents

| File                          | Used by                                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| `Crop_recommendation.csv`     | Decision Tree, K Nearest Neighbors, Random Forest                                               |
| `Steel_industry_data.csv`     | Linear Regression                                                                               |
| `BankNote_Authentication.csv` | Logistic Regression, Gradient Boosting                                                          |
| `fetal_health.csv`            | Neural Network (MLP), Perceptron, PCA                                                           |
| `Wholesale_customers_data.csv`| K Means Clustering, SVD                                                                         |
| `marketing_campaign.csv`      | DBSCAN (Customer Personality Analysis)                                                          |

## How notebooks find these files

Each notebook calls
[`rice_ml.processing.datasets.find_data_file`](../src/rice_ml/processing/datasets.py),
which walks up from the notebook's working directory looking for a
`data/<filename>` folder. That means the same notebook works whether
you run it from the repository root, from JupyterLab in the algorithm
subfolder, or from CI — there are no hard-coded paths.
