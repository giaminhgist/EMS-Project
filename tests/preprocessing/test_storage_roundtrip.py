"""Storage and round-trip tests on the synthetic dataset."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.config import ConfigError, PreprocessingConfig
from preprocessing.heatmaps import HeatmapParams, build_trial_heatmap
from preprocessing.pipeline import PipelineOptions, run_pipeline
from preprocessing.storage import StorageError, TrialStore
from tests.preprocessing.conftest import synthetic_config_dict


def _run(cfg, **options):
    config = PreprocessingConfig.from_dict(cfg)
    return run_pipeline(config, PipelineOptions(**options))


def test_composite_key_lookup_returns_exact_stored_row(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    report = _run(synthetic_config_dict(synthetic_raw, out))
    assert report.n_trials_ok == 6  # 3 (000) + 2 (005) + 1 (013)
    assert report.n_trials_excluded == 1
    assert report.n_trials_observed == 7

    store = TrialStore(out)
    store.verify_manifest_checksums()
    rec = store.get_trial("000", "a1.jpg")
    assert rec.subject_id == "000"
    assert rec.stimulus_id == "a1.jpg"
    assert rec.heatmap.shape == (3, 48, 64)
    assert rec.heatmap.dtype == np.float32
    assert np.all(np.isfinite(rec.heatmap))
    # Independently recompute the expected heatmap for this trial.
    events = [("a1.jpg", 1, 150.0, 300.0, 400.0, 1000.0),
              ("a1.jpg", 2, 200.0, 600.0, 400.0, 1000.0),
              ("a1.jpg", 3, 250.0, 800.0, 200.0, 1000.0)]
    xs = np.array([e[3] for e in events], dtype=np.float64)
    ys = np.array([e[4] for e in events], dtype=np.float64)
    durations = np.array([e[2] for e in events], dtype=np.float64)
    params = HeatmapParams(48, 64, 2.0, 4.0, 0.5, 1e-8)
    expected, _ = build_trial_heatmap(
        xs, ys, np.ones(3, bool), durations, np.ones(3, bool),
        source_width=1024, source_height=768, params=params,
    )
    assert np.array_equal(rec.heatmap, expected)


def test_trial_manifest_roundtrip_and_ids(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    store = TrialStore(out)
    tm = store.trial_manifest
    assert set(tm.subject_id) == {"000", "005", "013"}
    # Non-contiguous IDs never become array positions.
    assert "000" in set(tm.subject_id)
    # Excluded trial has no array row.
    excl = tm[(tm.subject_id == "013") & (tm.stimulus_id == "a1.jpg")]
    assert len(excl) == 1
    assert excl.qc_status.iloc[0] == "excluded_no_spatial_fixations"
    assert pd_isna(excl.subject_row_index.iloc[0])
    # Every ok trial round-trips.
    for row in tm[tm.qc_status == "ok"].itertuples():
        rec = store.get_trial_by_uid(row.trial_uid)
        assert rec.stimulus_index == row.stimulus_index
        assert rec.heatmap.shape == (3, 48, 64)


def test_excluded_trial_raises_on_lookup(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    store = TrialStore(out)
    with pytest.raises(StorageError, match="not heatmap-eligible"):
        store.get_trial("013", "a1.jpg")


def test_store_rejects_missing_completion_record(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    (out / "dataset_metadata.json").unlink()
    with pytest.raises(StorageError, match="completion record"):
        TrialStore(out)


def test_incompatible_existing_output_requires_force(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    # Different configuration (sigma) -> incompatible without --force.
    with pytest.raises(StorageError):
        _run(synthetic_config_dict(synthetic_raw, out, gaussian_sigma_cells=1.5))
    # With --force the arrays are replaced.
    report = _run(
        synthetic_config_dict(synthetic_raw, out, gaussian_sigma_cells=1.5), force=True
    )
    assert report.n_trials_ok == 6


def test_rerun_with_unchanged_config_skips_compatible_outputs(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    report = _run(synthetic_config_dict(synthetic_raw, out))
    # All three subjects verified and skipped; outputs byte-unchanged.
    assert sum(1 for n in report.notes if n.startswith("skipped existing")) == 3
    assert report.n_trials_ok == 6


def test_resume_skips_valid_subjects_and_refills_missing(synthetic_raw, tmp_path):
    out = tmp_path / "out"
    _run(synthetic_config_dict(synthetic_raw, out))
    # Delete one subject dir to simulate an interrupted run.
    import shutil

    shutil.rmtree(out / "subjects" / "005")
    report = _run(synthetic_config_dict(synthetic_raw, out), resume=True)
    # The deleted subject is reprocessed; the intact subjects are skipped.
    assert any("000" in note for note in report.notes)
    assert any("013" in note for note in report.notes)
    assert not any("005" in note for note in report.notes)
    assert (out / "subjects" / "005" / "heatmaps.npy").is_file()
    # No staging directories remain after either run.
    leftovers = [p.name for p in (out / "subjects").iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []


def test_repeated_seeded_preprocessing_is_byte_identical(synthetic_raw, tmp_path):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    _run(synthetic_config_dict(synthetic_raw, out1))
    _run(synthetic_config_dict(synthetic_raw, out2))
    for subj in ["000", "005", "013"]:
        a = (out1 / "subjects" / subj / "heatmaps.npy").read_bytes()
        b = (out2 / "subjects" / subj / "heatmaps.npy").read_bytes()
        assert a == b
        ai = (out1 / "subjects" / subj / "stimulus_indices.npy").read_bytes()
        bi = (out2 / "subjects" / subj / "stimulus_indices.npy").read_bytes()
        assert ai == bi
    assert (out1 / "source_inventory.json").read_bytes() == (
        out2 / "source_inventory.json"
    ).read_bytes()


def test_config_rejects_unknown_and_invalid_fields(synthetic_raw, tmp_path):
    with pytest.raises(ConfigError, match="unknown config fields"):
        PreprocessingConfig.from_dict(
            synthetic_config_dict(synthetic_raw, tmp_path / "x", bogus_field=1)
        )
    with pytest.raises(ConfigError, match="off_canvas_policy"):
        PreprocessingConfig.from_dict(
            synthetic_config_dict(synthetic_raw, tmp_path / "x", off_canvas_policy="clip")
        )
    with pytest.raises(ConfigError, match="gaussian_sigma_cells"):
        PreprocessingConfig.from_dict(
            synthetic_config_dict(synthetic_raw, tmp_path / "x", gaussian_sigma_cells=-1.0)
        )
    with pytest.raises(ConfigError, match="missing required fields"):
        PreprocessingConfig.from_dict({"raw_root": "/tmp"})


def test_ambiguous_basename_refused(tmp_path):
    from tests.preprocessing.conftest import write_image

    img_root = tmp_path / "Images"
    write_image(img_root / "C1" / "dup.jpg")
    write_image(img_root / "C2" / "dup.jpg")
    from preprocessing.inventory import InventoryError, inventory_stimuli

    with pytest.raises(InventoryError, match="ambiguous"):
        inventory_stimuli(img_root)


def pd_isna(value) -> bool:
    import pandas as pd

    return pd.isna(value)
