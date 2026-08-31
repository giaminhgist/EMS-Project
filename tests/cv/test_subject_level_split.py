"""Subject-level split tests on the synthetic 12-subject population."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cv.build_subject_folds import Population, build_assignments, fold_subject_partitions
from cv.config import CVConfig
from tests.cv.conftest import ALL_IDS, HC_IDS, SZ_IDS, cv_config_dict


@pytest.fixture
def built(cv_processed, tmp_path):
    from build_cv import build_split

    cfg = CVConfig.from_dict(cv_config_dict(cv_processed, tmp_path / "CV"))
    report = build_split(cfg)
    return cfg, report


def test_each_subject_assigned_exactly_once(built):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    assert len(a) == 12
    assert a.subject_id.is_unique
    assert set(a.subject_id) == set(ALL_IDS)
    assert a.validation_fold.isin(range(5)).all()


def test_validation_sets_disjoint_and_union_complete(built):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    sets = [set(a[a.validation_fold == k].subject_id) for k in range(5)]
    assert all(not (sets[i] & sets[j]) for i in range(5) for j in range(i + 1, 5))
    assert set().union(*sets) == set(ALL_IDS)
    # Union of validation sets equals the complete input population.
    assert set().union(*sets) == set(ALL_IDS)


def test_train_val_disjoint_per_fold(built):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    for fold in range(5):
        val = set(a[a.validation_fold == fold].subject_id)
        train = set(a[a.validation_fold != fold].subject_id)
        assert val.isdisjoint(train)
        assert len(train) + len(val) == 12


def test_stratification_within_feasible_limits(built):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    total_sz = total_hc = 0
    for fold in range(5):
        part = a[a.validation_fold == fold]
        val_sz = int((part.group == "SZ").sum())
        val_hc = int((part.group == "HC").sum())
        assert val_sz == 1  # 5 SZ across 5 folds
        assert val_hc in (1, 2)  # 7 HC across 5 folds
        total_sz += val_sz
        total_hc += val_hc
    assert total_sz == 5 and total_hc == 7


def test_trials_belong_to_partition_subjects(built, cv_processed):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    trials = pd.read_parquet(cv_processed / "trial_manifest.parquet")
    for fold in range(5):
        val_ids = set(a[a.validation_fold == fold].subject_id)
        train_trials = pd.read_parquet(cfg.output_dir / f"fold_{fold}" / "train_trials.parquet")
        val_trials = pd.read_parquet(cfg.output_dir / f"fold_{fold}" / "val_trials.parquet")
        assert set(train_trials.subject_id).isdisjoint(val_ids)
        assert set(val_trials.subject_id) <= val_ids
        # No subject contributes trials to both partitions of one fold.
        assert set(train_trials.subject_id).isdisjoint(set(val_trials.subject_id))
        # Partitions cover all trials exactly once.
        assert len(train_trials) + len(val_trials) == len(trials)
    # Every trial belongs to its subject's partition (explicit-ID join).
    for fold in range(5):
        part = a[a.validation_fold == fold]
        val_trials = pd.read_parquet(cfg.output_dir / f"fold_{fold}" / "val_trials.parquet")
        for subj in part.subject_id:
            assert val_trials[val_trials.subject_id == subj].shape[0] > 0
            assert set(val_trials[val_trials.subject_id == subj].subject_id) <= set(part.subject_id)


def test_stage1_hc_views_contain_no_sz(built):
    cfg, report = built
    meta = json.loads((cfg.output_dir / "cv_metadata.json").read_text())
    for f in meta["folds"]:
        assert f["stage1_train_hc_subjects"] == f["train_hc"]
        assert f["stage1_val_hc_subjects"] == f["val_hc"]
        # HC trial counts: recompute from the partition files.
        train_trials = pd.read_parquet(cfg.output_dir / f"fold_{f['fold']}" / "train_trials.parquet")
        val_trials = pd.read_parquet(cfg.output_dir / f"fold_{f['fold']}" / "val_trials.parquet")
        assert f["stage1_train_hc_trials"] == int((train_trials.group == "HC").sum())
        assert f["stage1_val_hc_trials"] == int((val_trials.group == "HC").sum())
        # HC-only views contain no SZ trials by construction.
        hc_train = train_trials[train_trials.group == "HC"]
        hc_val = val_trials[val_trials.group == "HC"]
        assert (hc_train.label == 1).sum() == 0
        assert (hc_val.label == 1).sum() == 0


def test_leading_zero_and_noncontiguous_ids_survive(built):
    cfg, report = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    assert "013" in set(a.subject_id)
    assert a.subject_id.dtype == object or pd.api.types.is_string_dtype(a.subject_id)
    for fold in range(5):
        tr = pd.read_csv(cfg.output_dir / f"fold_{fold}" / "train_subjects.csv", dtype={"subject_id": str})
        va = pd.read_csv(cfg.output_dir / f"fold_{fold}" / "val_subjects.csv", dtype={"subject_id": str})
        assert "013" in set(pd.concat([tr, va]).subject_id)
