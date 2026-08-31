"""Manifest compatibility, verify-only, and tamper detection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cv.build_subject_folds import CVError
from cv.config import CVConfig
from tests.cv.conftest import cv_config_dict


@pytest.fixture
def built(cv_processed, tmp_path):
    from build_cv import build_split

    cfg = CVConfig.from_dict(cv_config_dict(cv_processed, tmp_path / "CV"))
    report = build_split(cfg)
    return cfg, report, cv_processed


def test_verify_only_passes_on_fresh_split(built):
    from build_cv import verify_only

    cfg, report, _ = built
    result = verify_only(cfg)
    assert result["status"] == "ok"
    assert result["checks"] is True


def test_verify_only_detects_modified_subject_manifest(built):
    from build_cv import verify_only

    cfg, report, cv_processed = built
    manifest = cv_processed / "subject_manifest.csv"
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b"999,999,HC,0,999.xlsx,10,10,90,abc\n")
        with pytest.raises(CVError):
            verify_only(cfg)
    finally:
        manifest.write_bytes(original)


def test_verify_only_detects_modified_trial_manifest(built):
    from build_cv import verify_only

    cfg, report, cv_processed = built
    manifest = cv_processed / "trial_manifest.parquet"
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b"tamper")
        with pytest.raises(CVError):
            verify_only(cfg)
    finally:
        manifest.write_bytes(original)


def test_verify_only_detects_modified_fold_assignments(built):
    from build_cv import verify_only

    cfg, report, _ = built
    path = cfg.output_dir / "fold_assignments.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"999,999,HC,0,0,1\n")
        with pytest.raises(CVError):
            verify_only(cfg)
    finally:
        path.write_bytes(original)


def test_partition_subject_files_match_assignments(built):
    cfg, report, _ = built
    a = pd.read_csv(cfg.output_dir / "fold_assignments.csv", dtype={"subject_id": str})
    for fold in range(5):
        tr = pd.read_csv(cfg.output_dir / f"fold_{fold}" / "train_subjects.csv", dtype={"subject_id": str})
        va = pd.read_csv(cfg.output_dir / f"fold_{fold}" / "val_subjects.csv", dtype={"subject_id": str})
        expected_val = set(a[a.validation_fold == fold].subject_id)
        expected_train = set(a[a.validation_fold != fold].subject_id)
        assert set(va.subject_id) == expected_val
        assert set(tr.subject_id) == expected_train
        assert set(va.label) <= {0, 1}


def test_metadata_records_input_checksums_and_fold_counts(built, cv_processed):
    cfg, report, _ = built
    meta = json.loads((cfg.output_dir / "cv_metadata.json").read_text())
    assert meta["population"] == {"n_subjects": 12, "n_hc": 7, "n_sz": 5, "n_trials": 25}
    assert len(meta["folds"]) == 5
    for key in ("subject_manifest_sha256", "trial_manifest_sha256", "source_inventory_sha256"):
        assert len(meta["input_checksums"][key]) == 64
    assert meta["seed"] == 2026
    assert meta["config_hash"] == cfg.config_hash()
