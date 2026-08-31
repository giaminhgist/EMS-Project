"""Stage-1 dataset and collator tests on the synthetic environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from stage1.config import ConfigError, Stage1Config
from stage1.dataset import Stage1DataError, Stage1Dataset


def _cfg(stage1_cfg_dict):
    return Stage1Config.from_dict(stage1_cfg_dict)


def test_dataset_returns_exact_shapes_and_ids(stage1_cfg_dict):
    cfg = _cfg(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    assert ds.n_rows_filtered_non_hc > 0  # SZ rows were filtered, not served
    assert len(ds) == 5  # 000: 3 ok trials + 005: 2 ok trials
    for i in range(len(ds)):
        s = ds[i]
        assert tuple(s.heatmap.shape) == (3, 48, 64)
        assert s.heatmap.dtype == torch.float32
        assert s.group == "HC"
        assert s.subject_id in {"000", "005"}
        assert torch.all(torch.isfinite(s.heatmap))
        # Fixed transform ranges.
        assert torch.all(s.heatmap[0] >= 0) and torch.all(s.heatmap[1] >= 0)
        assert torch.all(torch.abs(s.heatmap[2]) <= 1.0)


def test_composite_key_matches_stored_row(stage1_env, stage1_cfg_dict):
    cfg = _cfg(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    sample = next(s for s in (ds[i] for i in range(len(ds))) if s.subject_id == "000" and s.stimulus_id == "a1.jpg")
    stored = np.load(stage1_env["processed"] / "subjects" / "000" / "heatmaps.npy", mmap_mode="r")
    tm = pd.read_parquet(stage1_env["processed"] / "trial_manifest.parquet")
    row = tm[(tm.subject_id == "000") & (tm.stimulus_id == "a1.jpg")].iloc[0]
    raw = np.array(stored[int(row.subject_row_index)], dtype=np.float32)
    expected = raw.copy()
    expected[0] = np.log1p(np.maximum(expected[0], 0.0))
    expected[1] = np.log1p(np.maximum(expected[1], 0.0))
    expected[2] = np.clip(expected[2], -1.0, 1.0)
    assert np.array_equal(sample.heatmap.numpy(), expected)


def test_collator_deduplicates_stimuli(stage1_env, stage1_cfg_dict):
    cfg = _cfg(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    indices = []
    for i in range(len(ds)):
        s = ds[i]
        if s.stimulus_id in ("a1.jpg", "b1.jpg"):
            indices.append(i)
    batch = ds.collate_from_indices(indices)  # 4 trials over 2 stimuli
    assert batch.n_trials == 4
    assert batch.n_unique_stimuli == 2
    dino = np.load(stage1_env["dino"] / "patch_tokens.npy", mmap_mode="r")
    feature_manifest = pd.read_csv(stage1_env["dino"] / "feature_manifest.csv", dtype=str)
    for k, sample in enumerate(ds[i] for i in indices):
        slot = int(batch.trial_to_stimulus_slot[k])
        row = int(feature_manifest[feature_manifest.stimulus_id == sample.stimulus_id].feature_row_index.iloc[0])
        expected = torch.from_numpy(np.array(dino[row], dtype=np.float32))
        assert torch.equal(batch.unique_dino_tokens[slot], expected)
    # Metadata preserved per trial.
    assert len(batch.subject_ids) == 4
    assert len(batch.trial_uids) == 4
    assert all(g == "HC" for g in batch.groups)


def test_val_split_excludes_non_ok_rows(stage1_cfg_dict):
    cfg = _cfg(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "val"
    )
    assert ds.n_rows_filtered_non_ok == 1  # 013/a1 excluded_no_spatial
    assert len(ds) == 1  # 013/a2 only
    assert ds[0].stimulus_id == "a2.jpg"
    assert ds[0].subject_id == "013"


def test_dataset_verifies_checksums_and_refuses_tampering(stage1_env, stage1_cfg_dict):
    cfg = _cfg(stage1_cfg_dict)
    manifest = stage1_env["processed"] / "subject_manifest.csv"
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b"tamper\n")
        with pytest.raises(Stage1DataError, match="checksum"):
            Stage1Dataset(
                cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
            )
    finally:
        manifest.write_bytes(original)


def test_config_rejects_invalid_combinations(stage1_cfg_dict):
    with pytest.raises(ConfigError, match="share_cross_attention_weights"):
        Stage1Config.from_dict(
            {
                **stage1_cfg_dict,
                "model": {**stage1_cfg_dict["model"], "share_cross_attention_weights": True},
            }
        )
    with pytest.raises(ConfigError, match="masked-only"):
        Stage1Config.from_dict(
            {
                **stage1_cfg_dict,
                "masking": {**stage1_cfg_dict["masking"], "train_mask_ratio": 0.0},
            }
        )
    with pytest.raises(ConfigError, match="unknown"):
        Stage1Config.from_dict({**stage1_cfg_dict, "bogus": 1})
