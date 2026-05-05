# Neural Network - Fetal Health Classification

A neural network composes affine transformations with nonlinear
activations so it can learn decision boundaries that are not linearly
separable. In this example, a multilayer perceptron classifies fetal
health measurements into the dataset's three diagnostic classes.

## Mathematical Explanation

For layer $\ell$, the network computes:

$$z^{(\ell)} = a^{(\ell-1)} W^{(\ell)} + b^{(\ell)}$$

and applies ReLU hidden activations:

$$a^{(\ell)} = \max(0, z^{(\ell)}).$$

The output layer uses softmax to convert logits into class
probabilities:

$$\hat{p}_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}.$$

Training minimizes cross-entropy between the one-hot labels and the
predicted class probabilities using mini-batch gradient descent.

## Dataset

[`fetal_health.csv`](../../../data/fetal_health.csv) - fetal
cardiotocography measurements labeled as normal, suspect, or
pathological. The notebook standardizes the numeric features before
training because neural-network optimization is sensitive to feature
scale.

## Notebook

[`neural_network.ipynb`](neural_network.ipynb) - load and prepare the
fetal-health dataset, train `MLPClassifier`, inspect training loss, and
evaluate multiclass predictions with accuracy and class-level results.
