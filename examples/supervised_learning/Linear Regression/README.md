# Linear Regression — Steel Industry Energy Consumption

A linear regression model fits a hyperplane through the data so that
the sum of squared residuals is minimized. We use it to predict the
energy consumption (`Usage_kWh`) of a steel plant from its operational
sensor readings.

## Mathematical Explanation

Predicted value:

$$\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p = w^\top x$$

Loss (mean squared error):

$$\mathcal{L}(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Closed-form solution (normal equations):

$$w^* = (X^\top X)^{-1} X^\top y$$

In `rice_ml` we solve them with `np.linalg.lstsq`, which is more
stable when $X$ is rank-deficient.

## Dataset

[`Steel_industry_data.csv`](../../../data/Steel_industry_data.csv) —
35,040 fifteen-minute readings from a small steel plant, with reactive
power, CO₂ emissions, power factor, time of day, and load type as
features and `Usage_kWh` as the regression target.

## Notebook

[`linear_regression.ipynb`](linear_regression.ipynb) — feature
engineering (one-hot encoded categorical columns, hour-of-day from the
timestamp), train/test split, residual diagnostic, and a ranked
coefficient table.