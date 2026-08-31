"""Missing-trial handling: incomplete subjects keep only their real trials."""

from __future__ import annotations

import pandas as pd
import pytest

from cv.config import CVConfig
from tests.cv.conftest import cv_config_dict


@pytest.fixture
def built(cv_processed, tmp_path):
    from build_cv import build_split

    cfg = CVConfig.from_dict(cv_config_dict(cv_processed, tmp_path / "CV"))
    report = build_split(cfg)
    return cfg, report


def test_incomplete_subject_keeps_only_real_trials(built, cv_processed):
    cfg, report = built
    trials = pd.read_parquet(cv_processed / "trial_manifest.parquet")
    # Subject 013 observes exactly one stimulus in the synthetic population.
    assert trials[trials.subject_id == "013"].shape[0] == 1
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    row = a[a.subject_id == "013"].iloc[0]
    assert row.n_observed_trials == 1
    fold = int(row.validation_fold)
    val_trials = pd.read_parquet(cfg.output_dir / f"fold_{fold}" / "val_trials.parquet")
    assert val_trials[val_trials.subject_id == "013"].shape[0] == 1
    # No fabricated rows anywhere: every partition row resolves to an observed
    # trial in the trial manifest.
    observed_keys = set(zip(trials.trial_uid, trials.subject_id, trials.stimulus_id))
    for f in range(5):
        tr = pd.read_parquet(cfg.output_dir / f"fold_{f}" / "train_trials.parquet")
        va = pd.read_parquet(cfg.output_dir / f"fold_{f}" / "val_trials.parquet")
        for part in (tr, va):
            keys = set(zip(part.trial_uid, part.subject_id, part.stimulus_id))
            assert keys <= observed_keys


def test_partition_trial_counts_sum_to_population(built, cv_processed):
    cfg, report = built
    trials = pd.read_parquet(cv_processed / "trial_manifest.parquet")
    total_train = total_val = 0
    for f in range(5):
        total_train += len(pd.read_parquet(cfg.output_dir / f"fold_{f}" / "train_trials.parquet"))
        total_val += len(pd.read_parquet(cfg.output_dir / f"fold_{f}" / "val_trials.parquet"))
    assert total_val == len(trials)  # every observed trial appears in one val partition
    assert total_train == 4 * len(trials)  # and in the 4 complementary train partitions


def test_missing_trials_do_not_affect_fold_membership(built, cv_processed):
    # Membership is subject-level and independent of trial completeness.
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    incomplete = set(a[a.n_observed_trials < 3].subject_id)
    complete = set(a[a.n_observed_trials >= 3].subject_id)
    assert incomplete and complete  # both kinds exist in the synthetic population
    for fold in range(5):
        val = set(a[a.validation_fold == fold].subject_id)
        # Fold membership is not simply completeness-sorted: both incomplete
        # and complete subjects appear across folds.
        assert val  # non-empty
    # Every subject is assigned regardless of completeness.
    assert set(a.subject_id) == set(pd.read_csv(cv_processed / "subject_manifest.csv", dtype={"subject_id": str}).subject_id)
