"""Norm-bank generation after training (acceptance path)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stage1.config import Stage1Config
from stage1.trainer import TrainLimits, build_norm_bank_from_checkpoint, run_training


def _smoke_cfg(stage1_cfg_dict):
    return Stage1Config.from_dict(
        {
            **stage1_cfg_dict,
            "loss": {**stage1_cfg_dict["loss"], "norm_start_epoch": 0, "norm_ramp_epochs": 0},
            "optimization": {**stage1_cfg_dict["optimization"], "epochs": 2, "amp": False},
        }
    )


def test_bank_built_from_best_checkpoint_excludes_validation(stage1_cfg_dict, tmp_path):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=2, run_name="bank"))
    best = outcome.run_dir / "fold_0" / "checkpoints" / f"best_stage1_fold{cfg.fold}.pt"
    assert best.is_file()
    out_dir = tmp_path / "bank"
    meta = build_norm_bank_from_checkpoint(
        best, cfg=cfg, device="cpu", output_dir=out_dir, min_samples=1
    )
    # Metadata contains only outer-training HC subjects.
    subjects = set(meta["subject_ids"])
    from stage1.dataset import Stage1Dataset

    val_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "val"
    )
    val_ids = set(val_ds.trial_rows.subject_id.astype(str))
    assert not (subjects & val_ids)
    assert subjects == {"000", "005"}
    # Arrays match the contract.
    mu = np.load(out_dir / "mu_trial.npy")
    sigma = np.load(out_dir / "sigma_trial.npy")
    counts = np.load(out_dir / "count_trial.npy")
    assert mu.shape == (3, 128) and mu.dtype == np.float32
    assert sigma.shape == (3, 128) and sigma.dtype == np.float32
    assert counts.dtype == np.int32
    assert counts.tolist() == [2, 1, 2]
    # Checkpoint SHA-256 recorded.
    assert len(meta["checkpoint_sha256"]) == 64
    assert meta["fold"] == cfg.fold
    # feature_manifest.csv lists stimuli explicitly.
    fm = pd.read_csv(out_dir / "feature_manifest.csv", dtype=str)
    assert list(fm.stimulus_id) == ["a1.jpg", "a2.jpg", "b1.jpg"]


def test_bank_rejects_validation_subject_contributions(stage1_cfg_dict):
    from stage1.dataset import Stage1Dataset
    from stage1.model import Stage1Model
    from stage1.normative_bank import NormBankConfig, NormBankError, build_normative_bank

    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    train_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    model = Stage1Model(cfg)
    model.eval()
    with pytest.raises(NormBankError, match="forbidden"):
        build_normative_bank(
            model, train_ds, fold=0, seed=2026, n_stimuli=3,
            checkpoint_sha256="0" * 64, processed_checksums={}, dino_checksum="0" * 64,
            config=NormBankConfig(min_samples=1),
            forbidden_subject_ids={"000"},
        )
