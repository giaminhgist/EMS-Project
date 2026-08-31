"""Synthetic training smoke tests: finite losses, history, checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1.config import Stage1Config
from stage1.trainer import TrainLimits, run_training


def _smoke_cfg(stage1_cfg_dict, **overrides):
    d = {
        **stage1_cfg_dict,
        "sampler": {
            **stage1_cfg_dict["sampler"],
            "stimuli_per_batch": 2,
            "hc_per_stimulus": 2,
            "min_hc_per_stimulus": 2,
        },
        "loss": {
            **stage1_cfg_dict["loss"],
            "norm_start_epoch": 1,
            "norm_ramp_epochs": 1,
            "lambda_norm": 0.1,
        },
        "optimization": {**stage1_cfg_dict["optimization"], "epochs": 2, "amp": False},
    }
    d.update(overrides)
    return Stage1Config.from_dict(d)


def test_two_epoch_synthetic_smoke(stage1_cfg_dict, tmp_path):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(
        cfg,
        device="cpu",
        limits=TrainLimits(max_epochs=2, run_name="smoke"),
        max_epochs=2,
    )
    assert outcome.epochs_completed == 2
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert len(history) == 2
    for col in ["train_loss", "val_loss", "train_recon_loss", "val_norm_loss"]:
        assert pd.api.types.is_float_dtype(history[col])
        assert history[col].notna().all()
    # Epoch 0 is warm-up (lambda_norm 0), epoch 1 is ramp.
    assert history.lambda_norm.tolist() == [0.0, pytest.approx(0.1)]
    # Training phase labels.
    assert history.training_phase.tolist() == ["warmup", "ramp"]
    # Checkpoints exist with the canonical names.
    ckpt_dir = outcome.run_dir / "fold_0" / "checkpoints"
    assert (ckpt_dir / f"last_stage1_fold{cfg.fold}.pt").is_file()
    # With eligibility only after the ramp completes (norm_start=1, ramp=1 ->
    # eligible from epoch 2), no best checkpoint may exist yet.
    assert not (ckpt_dir / f"best_stage1_fold{cfg.fold}.pt").exists()


def test_three_epoch_synthetic_selects_best_after_ramp(stage1_cfg_dict, tmp_path):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="smoke2"))
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert len(history) == 3
    assert history.eligible_for_best.tolist() == [False, False, True]
    ckpt_dir = outcome.run_dir / "fold_0" / "checkpoints"
    assert (ckpt_dir / f"best_stage1_fold{cfg.fold}.pt").is_file()
    assert history.is_best_epoch.any()
    assert history.loc[history.is_best_epoch, "best_epoch_so_far"].iloc[0] == 2


def test_run_directory_contains_metadata(stage1_cfg_dict, tmp_path):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="meta"))
    run_dir = outcome.run_dir
    for fname in ["run_metadata.json", "config_resolved.yaml", "environment.json", "source_checksums.json"]:
        assert (run_dir / fname).is_file(), fname
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["config_hash"] == cfg.config_hash()
    assert meta["ablation"] == "base"
    resolved = (run_dir / "config_resolved.yaml").read_text()
    assert "experiment_name" in resolved
    # Completed runs are never overwritten: a second run gets a new run_id.
    outcome2 = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="meta"))
    assert outcome2.run_id != outcome.run_id


def test_train_and_val_datasets_are_hc_only(stage1_cfg_dict):
    from stage1.dataset import Stage1Dataset

    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    for split in ("train", "val"):
        ds = Stage1Dataset(
            cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, split
        )
        assert ds.group_filter == "HC"
        assert all(s.group == "HC" for s in (ds[i] for i in range(len(ds))))


@pytest.mark.smoke
def test_real_fold_short_smoke():
    """Acceptance: one real fold completes a short smoke run."""
    cfg = Stage1Config.load_base_with_ablation(
        Path("/root/EMS-Project/configs/stage1/base.yaml"), None
    )
    cfg = Stage1Config(**{**cfg.to_dict(), "fold": 0})
    outcome = run_training(
        cfg,
        device="cpu",
        limits=TrainLimits(
            max_epochs=2, max_train_batches=3, max_val_batches=3, run_name="final_smoke"
        ),
    )
    assert outcome.epochs_completed == 2
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert len(history) == 2
    assert history.num_train_trials.tolist() == [192, 192]  # 3 batches x 64
    assert history.train_loss.notna().all()
    assert history.val_loss.notna().all()
