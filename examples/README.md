# Examples

This folder contains one Jupyter notebook per algorithm in `rice_ml`,
applied to a dataset. Each algorithm folder also has a short
`README.md` with a brief description and the relevant equations.

## Layout

```
examples/
├── supervised_learning/
│   ├── Decision Tree/                  ← Crop Recommendation
│   ├── Gradient Boosting/              ← Banknote Authentication
│   ├── K Means Clustering/             ← Wholesale Customers Data Set
│   ├── K Nearest Neighbors/            ← Crop Recommendation
│   ├── Linear Regression/              ← Steel Industry Energy Consumption
│   ├── Logistic Regression/            ← Banknote Authentication
│   ├── Neural Network/                 ← Fetal Health Classification
│   ├── Perceptron/                     ← Fetal Health Classification
│   ├── Random Forest/                  ← Crop Recommendation
│   └── README.md
└── unsupervised_learning/
    ├── DBSCAN/                         ← Customer Personality Analysis
    ├── K Means Clustering/             ← Wholesale Customers Data Set
    ├── PCA/                            ← Fetal Health Classification
    ├── SVD/                            ← Wholesale Customers Data Set
    └── README.md
```

Notebooks read their CSVs from the repository's [`data/`](../data/)
folder via `rice_ml.processing.datasets.find_data_file`, so they work
regardless of which directory you run them from.

`python scripts/build_notebooks.py` regenerates the Gradient Boosting,
K Means (supervised and unsupervised), Logistic Regression, Neural
Network, Perceptron, PCA, SVD, and **K Nearest Neighbors** notebooks.
When a target file already exists, the script **re-attaches stored cell
outputs** (so GitHub previews keep figures after a rebuild). Decision Tree,
Linear Regression, Random Forest, and DBSCAN are maintained as hand-edited
notebooks and are not overwritten by the script.

To **re-run every** `examples/**/*.ipynb` so GitHub shows fresh plots and
tables after you change code cells, install `jupyter` / `nbconvert` and run:

```bash
pip install -e .[notebooks]
python scripts/execute_example_notebooks.py
```

## Running the notebooks

```bash
pip install -e .[notebooks]
jupyter lab examples/
```

Example notebooks are **checked in with executed outputs** so figures and
tables render on GitHub. Regenerate notebook *sources* from
`build_notebooks.py`, then refresh outputs with the execute script above.
To regenerate notebook *structure* only:

```bash
python scripts/build_notebooks.py
```