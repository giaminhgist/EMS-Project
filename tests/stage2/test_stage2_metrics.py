"""Metrics tests (guide 07 §10, §17.2, §17.6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from stage2.metrics import (
    auroc,
    best_epoch_rule,
    is_better_candidate,
    subject_metrics,
)


def test_metrics_on_known_fixture():
    labels = [0, 0, 1, 1]
    logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    probs = torch.sigmoid(logits)
    m = subject_metrics(labels, logits, probs)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m["n_hc"] == 2 and m["n_sz"] == 2
    assert m["brier"] < 0.2


def test_metrics_with_errors():
    labels = [0, 0, 1, 1]
    probs = torch.tensor([0.9, 0.6, 0.4, 0.1])
    m = subject_metrics(labels, torch.zeros(4), probs)
    # Predictions 1,1,0,0 against labels 0,0,1,1: everything wrong.
    assert m["accuracy"] == 0.0
    cm = m["confusion_matrix"]
    assert cm == [[0, 2], [2, 0]]
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_single_class_auroc_is_none_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = subject_metrics([0, 0, 0], torch.zeros(3), torch.tensor([0.2, 0.3, 0.1]))
    assert m["auroc"] is None
    assert any("AUROC" in str(w.message) for w in caught)
    assert m["balanced_accuracy"] is None  # sensitivity undefined
    assert m["sensitivity"] is None
    assert m["specificity"] == 1.0


def test_zero_denominator_metrics_are_none_not_zero():
    m = subject_metrics([1, 1, 1], torch.zeros(3), torch.tensor([0.5, 0.5, 0.5]))
    assert m["specificity"] is None
    assert m["balanced_accuracy"] is None
    assert m["f1"] is not None  # positives exist


def test_auroc_with_ties():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert auroc(labels, scores) == pytest.approx(0.5)


def test_best_epoch_tie_breakers():
    # Primary: higher balanced accuracy wins.
    a = best_epoch_rule(0.8, 0.9, 1.0, 5)
    b = best_epoch_rule(0.7, 0.99, 0.1, 1)
    assert is_better_candidate(a, b)
    # Tie on bal-acc: higher AUROC wins.
    c = best_epoch_rule(0.8, 0.95, 1.0, 5)
    assert is_better_candidate(c, a)
    # Tie on bal-acc + AUROC: lower val_loss wins.
    d = best_epoch_rule(0.8, 0.95, 0.5, 5)
    assert is_better_candidate(d, c)
    # Tie on all metrics: earlier epoch wins.
    e = best_epoch_rule(0.8, 0.95, 0.5, 2)
    assert is_better_candidate(e, d)
    # None metrics sort after finite ones.
    f = best_epoch_rule(None, None, 0.1, 0)
    assert is_better_candidate(d, f)


def test_metrics_reject_empty_and_bad_inputs():
    with pytest.raises(ValueError, match="at least one subject"):
        subject_metrics([], [], [])
    with pytest.raises(ValueError, match="binary"):
        subject_metrics([0, 2], torch.zeros(2), torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="identical shapes"):
        subject_metrics([0, 1], torch.zeros(2), torch.tensor([0.5]))
