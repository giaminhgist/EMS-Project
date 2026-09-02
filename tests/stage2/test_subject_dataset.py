"""Subject dataset tests (guide 04 §12): panel assembly, masking, transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import (
    make_cv_metadata,
    make_cv_root,
    make_processed_fixture,
    make_stage2_config,
)
from stage2.config import Stage2Config
from stage2.dataset import Stage2DataError, Stage2SubjectDataset, fixed_heatmap_transform

HC = {"000": 0, "002": 0, "005": 0, "007": 0}
SZ = {"101": 1, "103": 1}
SUBJECTS = {**HC, **SZ}
TRAIN = ["000", "002", "101", "103"]
VAL = ["005", "007"]


def make_fixture(tmp_path, observed: dict[str, list[int]] | None = None):
    if observed is None:
        observed = {sid: list(range(4)) for sid in SUBJECTS}
    processed = make_processed_fixture(tmp_path, SUBJECTS, observed)
    cv_root = make_cv_root(tmp_path, SUBJECTS, TRAIN, VAL)
    make_cv_metadata(cv_root, processed)
    return processed, cv_root


def make_cfg(tmp_path, processed, cv_root) -> Stage2Config:
    # A dataset-only config needs no real bank; use a stub registry path.
    registry = tmp_path / "registry.yaml"
    import yaml

    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"x")
    import hashlib

    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "evaluation_regime": "pilot_existing_stage1",
                "folds": {
                    str(f): {
                        "checkpoint": str(checkpoint),
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    }
                    for f in range(5)
                },
            }
        ),
        encoding="utf-8",
    )
    bank_root = tmp_path / "bank"
    bank_root.mkdir(exist_ok=True)
    return make_stage2_config(
        processed_root=processed,
        cv_root=cv_root,
        bank_root=bank_root,
        registry_path=registry,
    )


class TestPanelAssembly:
    def test_complete_100_trial_subject(self, tmp_path):
        observed = {"000": list(range(100)), "002": list(range(4)), "101": list(range(4)), "103": list(range(4)), "005": list(range(4)), "007": list(range(4))}
        processed, cv_root = make_fixture(tmp_path, observed)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        assert len(ds) == 4
        sample = ds[0]
        assert sample.subject_id == "000"
        assert sample.label == 0
        assert bool(sample.trial_mask.all())
        assert sample.heatmaps.shape == (100, 3, 48, 64)
        assert all(uid is not None for uid in sample.trial_uids)
        assert sample.stimulus_indices.tolist() == list(range(100))

    def test_subject_missing_one_trial(self, tmp_path):
        observed = {sid: list(range(4)) for sid in SUBJECTS}
        observed["000"] = [0, 1, 3]  # stimulus 2 missing
        processed, cv_root = make_fixture(tmp_path, observed)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        sample = ds[0]
        assert not bool(sample.trial_mask[2])
        assert bool(sample.trial_mask[0]) and bool(sample.trial_mask[1]) and bool(sample.trial_mask[3])
        assert sample.trial_uids[2] is None
        assert sample.trial_uids[0] == "000_000"
        assert float(sample.heatmaps[2].abs().sum()) == 0.0  # storage padding stays zero

    def test_severe_missingness(self, tmp_path):
        observed = {sid: list(range(4)) for sid in SUBJECTS}
        observed["000"] = list(range(63))  # 37 missing, mirrors subject-216 style
        processed, cv_root = make_fixture(tmp_path, observed)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        sample = ds[0]
        assert int(sample.trial_mask.sum()) == 63
        assert int((~sample.trial_mask).sum()) == 37
        assert all(sample.trial_uids[s] is None for s in range(63, 100))

    def test_duplicate_stimulus_row_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        tri = pd.read_parquet(processed / "trial_manifest.parquet")
        dup = tri[tri.subject_id == "000"].iloc[[0]].copy()
        dup["stimulus_index"] = 0
        dup["trial_uid"] = "000_000_dup"
        dup["subject_row_index"] = 0
        pd.concat([tri, dup]).to_parquet(processed / "trial_manifest.parquet", index=False)
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="duplicate trial row"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)

    def test_noncanonical_stimulus_index_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        tri = pd.read_parquet(processed / "trial_manifest.parquet")
        tri.loc[tri.subject_id == "000", "stimulus_index"] = 100
        tri.to_parquet(processed / "trial_manifest.parquet", index=False)
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="noncanonical stimulus index"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)

    def test_wrong_heatmap_shape_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        np.save(processed / "subjects" / "000" / "heatmaps.npy", np.zeros((4, 3, 32, 64), np.float32))
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="heatmap shape"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)

    def test_wrong_heatmap_dtype_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        np.save(processed / "subjects" / "000" / "heatmaps.npy", np.zeros((4, 3, 48, 64), np.float64))
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="dtype"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)

    def test_non_finite_heatmap_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        arr = np.zeros((4, 3, 48, 64), np.float32)
        arr[0, 0, 0, 0] = np.inf
        np.save(processed / "subjects" / "000" / "heatmaps.npy", arr)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        with pytest.raises(Stage2DataError, match="non-finite"):
            _ = ds[0]

    def test_labels_from_manifest_not_id_ranges(self, tmp_path):
        # Non-contiguous ids: an id > 200 is HC and an id < 100 is SZ.
        subjects = {"208": 0, "209": 0, "005": 1, "006": 1}
        processed = make_processed_fixture(tmp_path, subjects, {s: list(range(3)) for s in subjects})
        cv_root = make_cv_root(tmp_path, subjects, ["208", "005"], ["209", "006"])
        make_cv_metadata(cv_root, processed)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        assert ds.subject_labels() == [0, 1]  # from manifest, not id thresholds

    def test_partition_label_disagreement_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        train_csv = cv_root / "fold_0" / "train_subjects.csv"
        df = pd.read_csv(train_csv, dtype=str)
        df.loc[df.subject_id == "000", "label"] = "1"
        df.to_csv(train_csv, index=False)
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="disagree"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)

    def test_no_observed_trials_failure(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        tri = pd.read_parquet(processed / "trial_manifest.parquet")
        tri = tri[tri.subject_id != "101"]
        tri.to_parquet(processed / "trial_manifest.parquet", index=False)
        cfg = make_cfg(tmp_path, processed, cv_root)
        with pytest.raises(Stage2DataError, match="no observed"):
            Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)


class TestTransformAndMemory:
    def test_fixed_transform_exact_values(self):
        x = np.zeros((3, 48, 64), np.float32)
        x[0].fill(1.0)
        x[1].fill(-2.0)
        x[2].fill(0.5)
        x[2, 0, 0] = 2.0
        out = fixed_heatmap_transform(x)
        np.testing.assert_allclose(out[0], np.log1p(1.0))
        np.testing.assert_allclose(out[1], 0.0)  # log1p(max(-2, 0)) = 0
        assert out[2, 0, 0] == 1.0  # clipped
        assert out[2, 0, 1] == 0.5
        assert x[0, 0, 0] == 1.0  # input not mutated

    def test_lru_memmap_round_trip(self, tmp_path):
        processed, cv_root = make_fixture(tmp_path)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        a = ds[0].heatmaps
        b = ds[0].heatmaps
        assert torch_equal(a, b)
        # Owned tensors: mutating the returned panel does not touch the store.
        a[0, 0, 0, 0] = 123.0
        c = ds[0].heatmaps
        assert float(c[0, 0, 0, 0]) != 123.0

    def test_missing_trial_counts(self, tmp_path):
        observed = {sid: list(range(4)) for sid in SUBJECTS}
        processed, cv_root = make_fixture(tmp_path, observed)
        cfg = make_cfg(tmp_path, processed, cv_root)
        ds = Stage2SubjectDataset(cfg, 0, "train", verify_checksums=False)
        # 4 train subjects each missing 96 of 100 stimuli
        assert ds.n_missing_trials() == 4 * 96


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))
