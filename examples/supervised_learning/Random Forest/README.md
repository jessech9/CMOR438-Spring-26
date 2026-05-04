# Random Forest — Crop Recommendation

This folder contains a Jupyter notebook demonstrating how to predict the most
suitable crop for a given soil and climate profile using a from-scratch
**Random Forest classifier** (`rice_ml.supervised_learning.random_forest`)
applied to the Crop Recommendation dataset. The workflow covers data loading,
exploratory data analysis, train/test splitting with stratification, training,
evaluation against a single decision tree baseline, an `n_estimators` learning
curve, permutation-based feature importance, and a confusion matrix.

## What is a random forest?

A random forest averages many de-correlated decision trees. Each tree is grown
on a **bootstrap sample** of the data — the same number of rows as the
training set, drawn with replacement, so roughly two-thirds of unique training
rows appear in any given tree and the rest form an "out-of-bag" sample. At
each split inside a tree, only a random subset of features is considered for
the best-split search (`max_features="sqrt"` by default). The combination of
bootstrapping and feature subsampling decorrelates the trees so that averaging
their predictions dramatically reduces variance without a corresponding
increase in bias.

For classification, the forest predicts the class with the highest average
class probability across all trees; for regression, it averages the trees'
numeric outputs.

## Mathematical Explanation

For an ensemble of `M` trees `h_1, …, h_M`, the forest predicts:

$$\hat{y}(x) = \frac{1}{M} \sum_{m=1}^{M} h_m(x) \quad \text{(regression)}$$

$$\hat{y}(x) = \arg\max_c \frac{1}{M} \sum_{m=1}^{M} P_m(c \mid x) \quad \text{(classification, soft-vote)}$$

The variance of the average of `M` identically-distributed estimators with
variance `σ²` and pairwise correlation `ρ` is:

$$\rho \sigma^2 + \frac{1 - \rho}{M}\,\sigma^2$$

The first term is the floor that adding more trees *can't* reduce, and the
second term shrinks toward zero as `M` grows. This is exactly why the trees
need to be **decorrelated** — feature subsampling makes `ρ` small, so the
floor is low and the ensemble keeps benefiting from extra trees.

Why a tree at all? A single deep CART tree has very high variance — it can
memorize the training set, so two trees fit on slightly different bootstrap
samples can disagree wildly on test points. That high variance is exactly
what bagging exploits.

## Project Structure

```
CMOR438-Spring-26/
├── data/
│   └── Crop_recommendation.csv                # raw dataset (also used by KNN)
├── src/rice_ml/
│   ├── _base.py                               # BaseEstimator, mixins, validators
│   ├── supervised_learning/
│   │   ├── decision_tree.py                   # CART classifier/regressor (RF backbone)
│   │   └── random_forest.py                   # RandomForest{Classifier,Regressor}
│   └── processing/
│       ├── pre_processing.py                  # train_test_split, scalers, encoders
│       ├── post_processing.py                 # accuracy, precision/recall/F1, ROC-AUC, ...
│       └── datasets.py                        # find_data_file()
├── tests/
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   └── conftest.py                            # rng + classification/cluster fixtures
└── examples/supervised_learning/Random Forest/
    ├── README.md                              # this file
    └── random_forest.ipynb                    # the walk-through
```

## Getting Started

### 1. Clone

```bash
git clone https://github.com/jessech9/CMOR438-Spring-26.git
cd CMOR438-Spring-26
```

### 2. Install

```bash
python -m pip install --upgrade pip
pip install -e .[dev,notebooks]
```

`[dev]` brings in `pytest`/`pytest-cov`; `[notebooks]` adds `jupyter`,
`seaborn`, and `scikit-learn` (the notebook uses sklearn for the confusion
matrix display only — the model itself is from scratch).

### 3. Open the notebook

```bash
jupyter lab "examples/supervised_learning/Random Forest/random_forest.ipynb"
```

The Crop Recommendation CSV is located by
`rice_ml.processing.datasets.find_data_file`, which walks upward from the
notebook's directory until it finds `data/Crop_recommendation.csv`, so the
notebook runs from any working directory.

## Notebook Workflow

The notebook is organized as the following sequential sections:

1. **Imports & setup.** Load `numpy`, `pandas`, `matplotlib`, `seaborn`, and
   pull `RandomForestClassifier`, `DecisionTreeClassifier`, and the
   processing utilities from `rice_ml`.
2. **Data loading & EDA.** Load the CSV, peek at the schema (7 numeric
   features + 1 string label across 22 crop classes), and plot per-feature
   distributions split by class to show the dataset isn't trivially linearly
   separable.
3. **Train/test split with stratification.** Use
   `rice_ml.processing.pre_processing.train_test_split` with `stratify=y` so
   every crop class is represented proportionally in both train and test
   splits.
4. **Single decision tree baseline.** Fit a depth-limited
   `DecisionTreeClassifier` to establish what a single de-correlated estimator
   can do — this is the baseline the forest needs to beat.
5. **Random forest training.** Fit a `RandomForestClassifier` with 100 trees
   and `max_features="sqrt"`; report train and test accuracy.
6. **Evaluation.** Compute confusion matrix and per-class precision / recall /
   F1; visualize the confusion matrix as a heatmap.
7. **`n_estimators` learning curve.** Sweep `n_estimators ∈ {1, 5, 10, 25, 50, 100, 200}`,
   plot test accuracy vs. number of trees to show the variance-reduction
   benefit and where it plateaus.
8. **Permutation feature importance.** Shuffle each input column on the test
   set and measure the resulting accuracy drop — the bigger the drop, the
   more important the feature. The from-scratch RF doesn't expose
   `feature_importances_` directly, so this is the model-agnostic version.
9. **Discussion.** Wrap up with takeaways on which features dominate, why
   ~50 trees suffices for this dataset, and where the forest still confuses
   classes.

## Example Results

A typical run with `n_estimators=100`, `max_depth=None`, `max_features="sqrt"`,
`random_state=0` yields, on a 80/20 stratified split:

- **Test accuracy:** ~0.99
- **Macro F1:** ~0.99
- **Top features by permutation importance:** `humidity`, `rainfall`, `K`,
  `N`, `temperature` (the macro-nutrient and weather columns dominate; `ph`
  contributes least).
- **Trees needed for the plateau:** test accuracy stops improving around
  `n_estimators=50`; doubling that to 100 gains ~0.001 accuracy at the cost
  of 2× training time.

## Next Steps

- Try a regression task with `RandomForestRegressor` on a different dataset
  (e.g. predicting yield or a continuous soil quality index).
- Experiment with `max_depth` and `min_samples_leaf` to see if shallower
  trees can still reach the plateau.
- Compare against `GradientBoostingClassifier` (a teammate's assignment) on
  the same dataset to see whether boosting beats bagging here.

## License

Released under the [MIT License](../../../LICENSE).
