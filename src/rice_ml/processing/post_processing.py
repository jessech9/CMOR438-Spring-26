"""Evaluation metrics for classification, regression, and clustering."""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "precision_recall_f1",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "roc_auc_score",
    "silhouette_score",
]


def _as_pair(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"length mismatch: y_true={y_true.shape[0]} y_pred={y_pred.shape[0]}"
        )
    return y_true, y_pred


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of correctly classified samples."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Compute a confusion matrix."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, *, positive_label=1) -> dict:
    """Precision, recall, and F1 for a binary classification problem."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    tp = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    fn = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc_score(y_true, y_score) -> float:
    """Area under the ROC curve via the Mann-Whitney U formula."""
    y_true, y_score = _as_pair(y_true, y_score)
    y_true = y_true.astype(int)
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("roc_auc_score requires binary labels in {0, 1}.")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both classes must be present in y_true.")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def mean_squared_error(y_true, y_pred) -> float:
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination R^2."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def silhouette_score(X, labels) -> float:
    """Mean silhouette coefficient over samples (Euclidean distances)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    mask = labels != -1
    X_in = X[mask]
    labels_in = labels[mask]

    unique = np.unique(labels_in)
    if len(unique) < 2:
        raise ValueError("silhouette_score needs at least 2 clusters.")

    n = X_in.shape[0]
    diff = X_in[:, None, :] - X_in[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    s = np.zeros(n)
    for i in range(n):
        own = labels_in == labels_in[i]
        own[i] = False
        if own.sum() == 0:
            s[i] = 0.0
            continue
        a_i = dist[i, own].mean()
        b_i = np.inf
        for c in unique:
            if c == labels_in[i]:
                continue
            others = labels_in == c
            if others.any():
                b_i = min(b_i, dist[i, others].mean())
        s[i] = (b_i - a_i) / max(a_i, b_i)
    return float(s.mean())
