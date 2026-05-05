# Decision Tree - Crop Recommendation

A decision tree makes predictions by recursively splitting the feature
space into regions that are increasingly pure. In this notebook, the
model recommends crops from soil chemistry and weather measurements.

## Mathematical Explanation

For classification, each candidate split is scored by how much it
reduces node impurity. With class proportions $p_1, \dots, p_K$, the
Gini impurity is:

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

The tree chooses the feature and threshold that maximize impurity
reduction:

$$\Delta G = G(parent) - \frac{n_L}{n}G(left) - \frac{n_R}{n}G(right)$$

Splitting continues until a stopping rule is reached, such as
`max_depth`, `min_samples_split`, or a pure leaf.

## Dataset

[`Crop_recommendation.csv`](../../../data/Crop_recommendation.csv) -
a 22-class farming dataset with nitrogen, phosphorus, potassium,
temperature, humidity, pH, and rainfall features. The target is the
recommended crop label.

## Notebook

[`decision_tree.ipynb`](decision_tree.ipynb) - data loading, exploratory
checks, train/test evaluation, a max-depth complexity curve, feature
importance review, and a short interpretation of a shallow tree as a
crop recommendation flow chart.
