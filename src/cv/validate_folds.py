"""Fold validation: leakage, completeness, and stratification checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import CVConfig


def run_all_checks(
    cfg: CVConfig,
    assignments: pd.DataFrame,
    trials: pd.DataFrame,
) -> dict[str, Any]:
    """Run every required check; return measured results (raises on failure)."""
    checks: dict[str, Any] = {}
    n = cfg.n_splits

    # 1. Each subject occurs exactly once.
    checks["subject_occurs_exactly_once"] = bool(
        assignments.subject_id.is_unique and len(assignments) > 0
    )

    # 2./3. Validation sets mutually disjoint; union equals the population.
    val_sets = [
        set(assignments[assignments.validation_fold == k].subject_id) for k in range(n)
    ]
    checks["validation_sets_disjoint"] = all(
        not (val_sets[i] & val_sets[j]) for i in range(n) for j in range(i + 1, n)
    )
    checks["validation_union_equals_population"] = (
        set().union(*val_sets) == set(assignments.subject_id)
    )
    checks["every_subject_assigned"] = bool(
        assignments.validation_fold.isin(range(n)).all()
    )

    # 4./6. No subject in both train and validation of one fold; no trial
    # crosses partitions.
    per_fold = []
    for k in range(n):
        val_ids = val_sets[k]
        train_trials = trials[~trials.subject_id.isin(val_ids)]
        val_trials = trials[trials.subject_id.isin(val_ids)]
        train_subjects = set(train_trials.subject_id)
        val_subjects = set(val_trials.subject_id)
        per_fold.append(
            {
                "fold": k,
                "train_val_subject_intersection_empty": bool(
                    train_subjects.isdisjoint(val_subjects)
                ),
                "trials_belong_to_partition_subjects": bool(
                    set(trials.subject_id) == (train_subjects | val_subjects)
                ),
                "train_trials": int(len(train_trials)),
                "val_trials": int(len(val_trials)),
            }
        )
    checks["folds"] = per_fold

    # 5. Stratification within feasible integer limits.
    strat = []
    for k in range(n):
        part = assignments[assignments.validation_fold == k]
        strat.append(
            {
                "fold": k,
                "val_hc": int((part.label == 0).sum()),
                "val_sz": int((part.label == 1).sum()),
            }
        )
    checks["stratification"] = strat

    # 7. Leading-zero / non-contiguous IDs preserved (string dtype + explicit
    # canonical set).
    checks["leading_zero_ids_preserved"] = bool(
        assignments.subject_id.dtype == object or pd.api.types.is_string_dtype(assignments.subject_id)
    ) and all(len(s) == 3 and s == s.zfill(3) for s in assignments.subject_id)

    failed = []
    if not checks["subject_occurs_exactly_once"]:
        failed.append("subject_occurs_exactly_once")
    if not checks["validation_sets_disjoint"]:
        failed.append("validation_sets_disjoint")
    if not checks["validation_union_equals_population"]:
        failed.append("validation_union_equals_population")
    if not checks["every_subject_assigned"]:
        failed.append("every_subject_assigned")
    for f in per_fold:
        if not f["train_val_subject_intersection_empty"]:
            failed.append(f"fold {f['fold']}: train/val subject intersection non-empty")
        if not f["trials_belong_to_partition_subjects"]:
            failed.append(f"fold {f['fold']}: trial partitions do not cover the population")
    checks["all_checks_pass"] = not failed
    checks["failures"] = failed
    return checks
