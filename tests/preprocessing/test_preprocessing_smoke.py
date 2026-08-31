"""Real-data smoke tests: one HC and one SZ subject into a scratch output.

These tests parse real EMS workbooks (a small subset) but never modify the
raw dataset and never interpret group differences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import PipelineOptions, run_pipeline

REPO = Path(__file__).resolve().parents[2]
RAW_FIX = REPO / "original_dataset" / "EMS" / "All_Data" / "Fixations"
RAW_IMG = REPO / "original_dataset" / "EMS" / "Images"
SMOKE_SUBJECTS = ["000", "203"]  # one HC, one SZ

pytestmark = pytest.mark.smoke


def _real_config_dict(output_root: Path) -> dict:
    return {
        "raw_root": str(REPO / "original_dataset"),
        "fixation_root": str(RAW_FIX),
        "image_root": str(RAW_IMG),
        "output_root": str(output_root),
        "num_workers": 0,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def raw_checksums() -> dict[str, str]:
    return {
        p.name: _sha256(p)
        for p in [RAW_FIX / f"{s}.xlsx" for s in SMOKE_SUBJECTS]
    }


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory) -> tuple[Path, Path]:
    """Run the smoke preprocessing once and share across tests in the module."""
    out = tmp_path_factory.mktemp("smoke_out")
    report = run_pipeline(
        PreprocessingConfig.from_dict(_real_config_dict(out)),
        PipelineOptions(subjects=SMOKE_SUBJECTS, output_root_override=out),
    )
    assert report.n_subjects == 2
    return out, tmp_path_factory.mktemp("smoke_out2")


def test_smoke_parses_real_workbooks(smoke_output, raw_checksums):
    out, _ = smoke_output
    # Raw sources unchanged.
    for name, digest in raw_checksums.items():
        assert _sha256(RAW_FIX / name) == digest
    # Both subjects present with arrays.
    for sid in SMOKE_SUBJECTS:
        assert (out / "subjects" / sid / "heatmaps.npy").is_file()
        assert (out / "subjects" / sid / "stimulus_indices.npy").is_file()
    # Partial run must NOT look like a complete dataset.
    assert not (out / "dataset_metadata.json").exists()


def test_smoke_arrays_and_counts(smoke_output):
    out, _ = smoke_output
    # Derive expectations from the raw workbooks (never hard-code them).
    expected_rows: dict[str, int] = {}
    expected_trials: dict[str, int] = {}
    for sid in SMOKE_SUBJECTS:
        df = pd.read_excel(RAW_FIX / f"{sid}.xlsx", sheet_name="Free_viewing")
        expected_rows[sid] = len(df)
        expected_trials[sid] = df["IMAGE"].nunique()
        qc = pd.read_parquet(out / "subjects" / sid / "trial_qc.parquet")
        assert qc.n_fixations_raw.sum() == expected_rows[sid]
        assert len(qc) == expected_trials[sid]
        heatmaps = np.load(out / "subjects" / sid / "heatmaps.npy")
        stim_idx = np.load(out / "subjects" / sid / "stimulus_indices.npy")
        assert heatmaps.shape[1:] == (3, 48, 64)
        assert heatmaps.dtype == np.float32
        assert np.all(np.isfinite(heatmaps))
        assert np.all(heatmaps[:, :2] >= 0)  # density channels are nonnegative
        assert np.all(np.abs(heatmaps[:, 2]) <= 1.0 + 1e-6)
        assert len(stim_idx) == len(heatmaps)
        assert np.all(np.diff(stim_idx) > 0)
        # Mass sanity: channel 0 mass close to the fixations used.
        assert np.all(heatmaps[:, 0].sum(axis=(1, 2)) > 0)


def test_smoke_qc_counts_reconcile(smoke_output):
    out, _ = smoke_output
    for sid in SMOKE_SUBJECTS:
        qc = pd.read_parquet(out / "subjects" / sid / "trial_qc.parquet")
        heatmaps = np.load(out / "subjects" / sid / "heatmaps.npy")
        ok = qc[qc.qc_status == "ok"]
        assert len(ok) == len(heatmaps)
        # subject_row_index is a dense 0..n-1 assignment in stimulus_index order.
        assert list(ok.subject_row_index.dropna().astype(int)) == list(range(len(ok)))
        # Mass columns match fixation counts within tolerance.
        for row in ok.itertuples():
            assert abs(row.mass_density - row.n_fixations_used) < 1e-2
            assert abs(row.mass_transition - row.n_transitions_used) < 1e-2
        # Per-trial median fixation duration is stored for EDA.
        med = ok.median_fixation_duration_ms.dropna()
        assert len(med) == len(ok)
        assert med.between(1.0, 5001.0).all()


def test_smoke_subjects_have_no_zero_spatial_trials(smoke_output):
    # The two smoke subjects observe all 100 stimuli with on-canvas fixations;
    # the dataset's zero-spatial trials belong to other subjects and are
    # covered by the unit tests.
    out, _ = smoke_output
    for sid in SMOKE_SUBJECTS:
        qc = pd.read_parquet(out / "subjects" / sid / "trial_qc.parquet")
        assert set(qc.qc_status) == {"ok"}
        assert len(qc) == 100


def test_smoke_determinism_across_runs(smoke_output):
    out, out2 = smoke_output
    _ = run_pipeline(
        PreprocessingConfig.from_dict(_real_config_dict(out2)),
        PipelineOptions(subjects=SMOKE_SUBJECTS, output_root_override=out2),
    )
    for sid in SMOKE_SUBJECTS:
        a = (out / "subjects" / sid / "heatmaps.npy").read_bytes()
        b = (out2 / "subjects" / sid / "heatmaps.npy").read_bytes()
        assert a == b


def test_smoke_dry_run_writes_nothing(smoke_output, tmp_path):
    _, _ = smoke_output
    dry_out = tmp_path / "dry_out"
    report = run_pipeline(
        PreprocessingConfig.from_dict(_real_config_dict(dry_out)),
        PipelineOptions(dry_run=True),
    )
    assert report.dry_run
    assert not dry_out.exists()
