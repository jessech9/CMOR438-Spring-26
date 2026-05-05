# Perceptron

The perceptron (Rosenblatt, 1958) is the simplest neural network: a
**single linear unit** with a step activation. It learns online via a
mistake-driven update rule and converges in finitely many steps if and
only if the data is linearly separable.

## Mathematical Explanation

Decision rule:

$$\hat{y}(x) = \begin{cases} 1 & \text{if } w^\top x + b \ge 0 \\ 0 & \text{otherwise} \end{cases}$$

Learning rule (per training example $(x, y)$):

$$w \leftarrow w + \eta\,(y - \hat{y})\,x, \qquad b \leftarrow b + \eta\,(y - \hat{y})$$

That is, **only update on mistakes**, and move the boundary toward
the misclassified point.

The *Perceptron Convergence Theorem* states: if there exists a
hyperplane that perfectly separates the classes, the perceptron will
find one in a finite number of updates. Conversely, on
**non-separable** data (XOR being the canonical example) the
algorithm oscillates forever.

## When to Use

- Almost never in practice — but it is essential to understand
  because it motivates everything that follows.
- As the historical and conceptual gateway to multi-layer networks.

## Notebook

[`perceptron.ipynb`](perceptron.ipynb) — Normal vs adverse fetal health
(binary merge of the multi-class labels), plus the XOR failure case.