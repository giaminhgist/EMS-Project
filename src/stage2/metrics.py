"""Subject-level metrics (guide 07 §10, contracts §24).

Computed once from the complete validation subject set; never assumed to
contain both classes. HC=0, SZ=1. Undefined values (single-class AUROC,
zero-denominator sensitivity/specificity/F1) are stored as ``None``, never
substituted with zero. The best-epoch rule is a deterministic ordered tuple.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


class MetricsError(ValueError):
    pass


def _check_inputs(labels: Any, probabilities: Any) -> None:
    labels = np.asarray(labels)
    if labels.size == 0:
        raise MetricsError("metrics require at least one subject")
    if not set(np.unique(labels)) <= {0, 1}:
        raise MetricsError(f"subject labels must be binary, got {sorted(set(np.unique(labels)))}")


def auroc(labels: Any, scores: Any) -> float | None:
    """Area under the ROC curve via average positive ranks (Mann-Whitney U).

    Ties are handled by mid-ranks. ``None`` when either class is absent.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Mid-ranks for ties.
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = mid
        i = j + 1
    rank_sum_pos = ranks[pos].sum()
    return float(
        (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


def subject_metrics(
    labels: Any,
    logits: Any,
    probabilities: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """All contract §24 metrics from complete subject predictions.

    ``labels`` ints (0/1), ``logits`` raw floats, ``probabilities`` in [0,1].
    Undefined metrics are ``None`` (never 0).
    """
    _check_inputs(labels, probabilities)
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape:
        raise MetricsError("probabilities and labels must have identical shapes")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise MetricsError("probabilities must be in [0, 1]")

    n = y.size
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    accuracy = float((tp + tn) / n)
    sensitivity = (tp / n_pos) if n_pos > 0 else None
    specificity = (tn / n_neg) if n_neg > 0 else None
    balanced_accuracy = None
    if sensitivity is not None and specificity is not None:
        balanced_accuracy = float((sensitivity + specificity) / 2.0)
    f1 = None
    if n_pos > 0 and (tp + fp + fn) > 0 and (tp + fp) > 0:
        precision = tp / (tp + fp)
        recall = tp / n_pos
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else None
    brier = float(np.mean((p - y) ** 2))
    auc = auroc(y, p)
    if auc is None:
        warnings.warn("AUROC undefined: the validation set contains a single class", stacklevel=2)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": brier,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n_subjects": n,
        "n_hc": n_neg,
        "n_sz": n_pos,
        "threshold": threshold,
    }


def best_epoch_rule(
    val_balanced_accuracy: float | None,
    val_auroc: float | None,
    val_loss: float,
    epoch: int,
) -> tuple:
    """Ordered best-epoch key: maximize bal-acc, then AUROC, then lower loss,
    then the earlier epoch. ``None`` metrics sort after finite ones for
    maximization (they are stored as -inf)."""
    bal = val_balanced_accuracy if val_balanced_accuracy is not None else float("-inf")
    auc = val_auroc if val_auroc is not None else float("-inf")
    return (bal, auc, -float(val_loss), -epoch)


def is_better_candidate(candidate: tuple, current: tuple) -> bool:
    """``candidate`` beats ``current`` under the ordered best-epoch rule."""
    return tuple(candidate) > tuple(current)
