# Examples

This folder contains one Jupyter notebook per algorithm in `rice_ml`,
applied to a real dataset. Each algorithm folder also has a short
`README.md` with a brief description, the relevant equations, and any
notes on the dataset / hyperparameters used.

## Layout

```
examples/
├── supervised_learning/
│   ├── Decision Tree/                   ← Crop Recommendation
│   ├── Gradient Boosting/               ← Banknote Authentication
│   ├── K Nearest Neighbors/             ← Crop Recommendation
│   ├── Linear Regression/               ← Steel Industry Energy Consumption
│   ├── Logistic Regression/             ← Banknote Authentication
│   ├── Neural Network/                  ← Fetal Health
│   ├── Perceptron/                      ← Fetal Health
│   ├── Random Forest/                   ← Crop Recommendation
│   └── README.md
└── unsupervised_learning/
    ├── DBSCAN/                          ← Customer Personality Analysis
    ├── K Means Clustering/              ← Wholesale Customers
    ├── PCA/                             ← Fetal Health
    ├── SVD/                             ← Wholesale Customers
    └── README.md
```

Notebooks read their CSVs from the repository's [`data/`](../data/)
folder via `rice_ml.processing.datasets.find_data_file`, so they work
regardless of which directory you run them from.

## Running the notebooks

```bash
pip install -e ".[notebooks]"
jupyter lab examples/
```

Notebooks are **checked in with executed outputs** so figures and
tables render on GitHub. To re-execute a notebook in place:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  "examples/supervised_learning/Random Forest/random_forest.ipynb"
```

(Replace the path with whichever notebook you want to re-run.)
