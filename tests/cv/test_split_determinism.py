"""Split determinism and immutability tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cv.config import CVConfig
from tests.cv.conftest import cv_config_dict


def _build(cv_processed, out_root, **overrides):
    from build_cv import build_split

    cfg = CVConfig.from_dict(cv_config_dict(cv_processed, out_root, **overrides))
    report = build_split(cfg)
    return cfg, report


def test_same_seed_and_inputs_reproduce_identical_assignments(cv_processed, tmp_path):
    cfg1, _ = _build(cv_processed, tmp_path / "CV1")
    cfg2, _ = _build(cv_processed, tmp_path / "CV2")
    a1 = (cfg1.output_dir / "fold_assignments.csv").read_bytes()
    a2 = (cfg2.output_dir / "fold_assignments.csv").read_bytes()
    assert a1 == a2
    m1 = json.loads((cfg1.output_dir / "cv_metadata.json").read_text())
    m2 = json.loads((cfg2.output_dir / "cv_metadata.json").read_text())
    assert m1["fold_assignments_sha256"] == m2["fold_assignments_sha256"]
    assert m1["input_checksums"] == m2["input_checksums"]


def test_different_seed_changes_assignments_but_stays_valid(cv_processed, tmp_path):
    cfg1, _ = _build(cv_processed, tmp_path / "CV1", random_state=2026)
    cfg2, _ = _build(cv_processed, tmp_path / "CV2", random_state=7)
    assert cfg1.versioned_dir_name != cfg2.versioned_dir_name
    a1 = pd.read_csv(cfg1.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    a2 = pd.read_csv(cfg2.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    assert not a1.equals(a2)  # at least one assignment changed
    # Both remain valid: disjoint, complete, stratified.
    for a in (a1, a2):
        sets = [set(a[a.validation_fold == k].subject_id) for k in range(5)]
        assert all(not (sets[i] & sets[j]) for i in range(5) for j in range(i + 1, 5))
        assert set().union(*sets) == set(a.subject_id)
        assert all((a[a.validation_fold == k].group == "SZ").sum() == 1 for k in range(5))


def test_same_directory_refuses_different_config(cv_processed, tmp_path):
    out_root = tmp_path / "CV"
    _build(cv_processed, out_root, random_state=2026)
    from cv.build_subject_folds import CVError

    # A different seed gets its own versioned directory (never overwritten)...
    cfg_other_seed, _ = _build(cv_processed, out_root, random_state=7)
    assert cfg_other_seed.versioned_dir_name == "5fold_seed7"
    # ...but the SAME versioned directory refuses a different config hash.
    tampered = tmp_path / "tampered_manifest.csv"
    import shutil

    shutil.copy(cv_processed / "subject_manifest.csv", tampered)
    with pytest.raises(CVError, match="config hash"):
        _build(cv_processed, out_root, subject_manifest=str(tampered), random_state=2026)


def test_identical_config_rerun_is_unchanged(cv_processed, tmp_path):
    from build_cv import build_split, verify_only

    out_root = tmp_path / "CV"
    cfg, _ = _build(cv_processed, out_root)
    # A second build with identical config rewrites identical artifacts.
    report = build_split(cfg)
    assert verify_only(cfg)["status"] == "ok"
    assert report["population"]["n_subjects"] == 12
