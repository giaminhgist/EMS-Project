"""Subject-level five-fold assignment and population reconciliation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CVConfig


class CVError(ValueError):
    pass


@dataclass
class Population:
    """The audited All_Data population reconciled from its inventory."""

    subjects: pd.DataFrame  # subject_id (str, leading zeros), subject_numeric_id, group, label, n_trials
    trials: pd.DataFrame  # full trial manifest rows

    @classmethod
    def load(cls, cfg: CVConfig) -> "Population":
        subject_manifest = pd.read_csv(
            cfg.subject_manifest,
            dtype={"subject_id": str, "group": str, "source_workbook": str},
        )
        if not subject_manifest.subject_id.is_unique:
            raise CVError("subject manifest contains duplicate subject_ids")
        # Reconcile with the audited population inventory (the All_Data
        # workbook inventory; no All_Data.xlsx exists in this release).
        inventory = json.loads(cfg.source_inventory.read_text(encoding="utf-8"))
        inventory_subjects = [s["subject_id"] for s in inventory["subjects"]]
        if len(set(inventory_subjects)) != len(inventory_subjects):
            raise CVError("source inventory contains duplicate subject ids")
        manifest_ids = set(subject_manifest.subject_id)
        inventory_ids = set(inventory_subjects)
        if manifest_ids != inventory_ids:
            raise CVError(
                f"subject manifest and audited inventory disagree: "
                f"manifest-only {sorted(manifest_ids - inventory_ids)}, "
                f"inventory-only {sorted(inventory_ids - manifest_ids)}"
            )
        # No unapproved external/held-out test subject may enter CV.
        test_like = [s for s in manifest_ids if s.startswith("Test_")]
        if test_like:
            raise CVError(f"test-set subjects leaked into the CV population: {test_like}")
        # Exactly one explicit label per subject; no duplicate IDs after
        # preserving leading zeros.
        labels_per_subject = subject_manifest.groupby("subject_id").label.nunique()
        if (labels_per_subject > 1).any():
            raise CVError("at least one subject has multiple labels")
        if not set(subject_manifest.label) <= {0, 1}:
            raise CVError("labels must be 0 (HC) or 1 (SZ)")

        trials = pd.read_parquet(cfg.trial_manifest)
        trials["subject_id"] = trials["subject_id"].astype("string")
        unknown = set(trials.subject_id) - manifest_ids
        if unknown:
            raise CVError(f"trial manifest references subjects outside the population: {sorted(unknown)}")

        subjects = subject_manifest[
            ["subject_id", "subject_numeric_id", "group", "label", "n_trials"]
        ].sort_values("subject_numeric_id").reset_index(drop=True)
        return cls(subjects=subjects, trials=trials)


def build_assignments(cfg: CVConfig, subjects: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stratified subject-level fold assignment.

    ``StratifiedKFold`` is applied to one row per subject; trial completeness
    never influences membership.
    """
    splitter = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state
    )
    y = subjects.label.to_numpy()
    fold_of = np.full(len(subjects), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(splitter.split(subjects, y)):
        if np.any(fold_of[val_idx] != -1):
            raise CVError("a subject was assigned to more than one validation fold")
        fold_of[val_idx] = fold
    if np.any(fold_of < 0):
        raise CVError("some subjects were not assigned to any validation fold")

    out = subjects.copy()
    out["validation_fold"] = fold_of
    out = out.rename(columns={"n_trials": "n_observed_trials"})
    return out[
        ["subject_id", "subject_numeric_id", "group", "label", "validation_fold", "n_observed_trials"]
    ]


def fold_subject_partitions(
    cfg: CVConfig, assignments: pd.DataFrame
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return ``{fold: (train_subjects, val_subjects)}``."""
    out = {}
    for fold in range(cfg.n_splits):
        val = assignments[assignments.validation_fold == fold].reset_index(drop=True)
        train = assignments[assignments.validation_fold != fold].reset_index(drop=True)
        out[fold] = (train, val)
    return out


def fold_trial_partitions(
    assignments: pd.DataFrame, trials: pd.DataFrame, n_splits: int
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Join subject partitions to the trial manifest (observed trials only)."""
    out = {}
    for fold in range(n_splits):
        val_ids = set(assignments[assignments.validation_fold == fold].subject_id)
        val_trials = trials[trials.subject_id.isin(val_ids)].reset_index(drop=True)
        train_trials = trials[~trials.subject_id.isin(val_ids)].reset_index(drop=True)
        out[fold] = (train_trials, val_trials)
    return out


def fold_summaries(
    cfg: CVConfig,
    assignments: pd.DataFrame,
    trial_partitions: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[dict[str, Any]]:
    """Per-fold counts incl. the HC-only Stage-1 views."""
    summaries = []
    for fold in range(cfg.n_splits):
        train, val = fold_subject_partitions(cfg, assignments)[fold]
        train_trials, val_trials = trial_partitions[fold]
        hc_train = train[train.group == "HC"]
        hc_val = val[val.group == "HC"]
        summaries.append(
            {
                "fold": fold,
                "n_train_subjects": int(len(train)),
                "n_val_subjects": int(len(val)),
                "train_hc": int((train.group == "HC").sum()),
                "train_sz": int((train.group == "SZ").sum()),
                "val_hc": int((val.group == "HC").sum()),
                "val_sz": int((val.group == "SZ").sum()),
                "n_train_trials": int(len(train_trials)),
                "n_val_trials": int(len(val_trials)),
                "stage1_train_hc_subjects": int(len(hc_train)),
                "stage1_val_hc_subjects": int(len(hc_val)),
                "stage1_train_hc_trials": int(len(train_trials[train_trials.group == "HC"])),
                "stage1_val_hc_trials": int(len(val_trials[val_trials.group == "HC"])),
            }
        )
    return summaries
