# Singular Value Decomposition (SVD)

SVD factorizes any real matrix $X$ as $U \Sigma V^\top$ where $U$ and
$V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries
(the **singular values**). Truncating to the top-$k$ singular values
gives the **best rank-$k$ approximation** in the Frobenius norm
(Eckart–Young theorem).

## Mathematical Explanation

For an $n \times p$ matrix $X$ with rank $r \le \min(n, p)$:

$$X = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i\,u_i\,v_i^\top$$

where $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ are the
singular values and $u_i$, $v_i$ are the corresponding left and right
singular vectors.

The **best rank-$k$ approximation** drops everything past the $k$th
singular value:

$$X_k = \sum_{i=1}^{k} \sigma_i\,u_i\,v_i^\top$$

This minimizes $\lVert X - X_k \rVert_F^2 = \sum_{i=k+1}^{r}
\sigma_i^2$ over all rank-$k$ matrices.

`rice_ml.SVD` is a transformer wrapping `np.linalg.svd` with a
deterministic sign convention. Unlike `PCA`, SVD does **not** subtract
the mean — useful when the mean is meaningful (e.g. raw pixel
intensities) or when working with sparse matrices.

## When to Use

- Lossy compression of matrices (images, document-term matrices,
  recommender-system rating matrices).
- Latent Semantic Analysis (LSA) for text.
- Initialization for low-rank optimization problems.

## Notebook

[`svd.ipynb`](svd.ipynb) — truncated SVD and rank-2 reconstruction on
the centered Wholesale Customers spending matrix.