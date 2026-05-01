# Logistic Regression

Logistic regression is a linear classifier that outputs the
**probability** that a sample belongs to the positive class. It is
fast, interpretable, and naturally calibrated for binary problems.

## Mathematical Explanation

Linear score combined with the sigmoid:

$$P(y=1 \mid x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}$$

Coefficients are estimated by minimizing the **negative
log-likelihood** (binary cross-entropy):

$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i}\!\Big[y_i \log \sigma(w^\top x_i) + (1 - y_i)\log\!\big(1 - \sigma(w^\top x_i)\big)\Big]$$

There is no closed form; we use full-batch gradient descent with
optional $L_2$ regularization. The gradient of the unregularized loss
is:

$$\nabla_w \mathcal{L} = \frac{1}{n}\,X^\top\big(\sigma(Xw) - y\big)$$

## When to Use

- Binary classification with mostly linear class boundaries.
- When you need calibrated probabilities, not just a label.
- As an interpretable baseline.

## Notebook

[`logistic_regression.ipynb`](logistic_regression.ipynb) — applied to
the Banknote Authentication dataset.