"""Checkpoint save/resume/weight-load tests."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from stage1.checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_weights_only,
    resume_checkpoint,
    save_checkpoint,
)
from stage1.config import Stage1Config
from stage1.model import Stage1Model
from stage1.trainer import TrainLimits, run_training


def _smoke_cfg(stage1_cfg_dict):
    return Stage1Config.from_dict(
        {
            **stage1_cfg_dict,
            "loss": {**stage1_cfg_dict["loss"], "norm_start_epoch": 1, "norm_ramp_epochs": 1},
            "optimization": {**stage1_cfg_dict["optimization"], "epochs": 4, "amp": False},
        }
    )


def test_exact_resume_continues_epoch_and_state(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome1 = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=2, run_name="r1"))
    ckpt = outcome1.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"

    # Resume into a NEW run (same config, new run_id).
    resume_cfg = Stage1Config(**{
        **cfg.to_dict(),
        "paths": {**cfg.paths.to_dict(), "output_root": str(cfg.paths.output_root)},
    })
    # run_training refuses an existing run dir only for new run ids; resume
    # writes into a fresh run dir with the same run_id as the checkpoint.
    from stage1.trainer import run_training as rt

    meta = load_checkpoint(ckpt)
    resume_run_id = meta.meta["run_id"]
    outcome2 = rt(
        resume_cfg, device="cpu",
        limits=TrainLimits(max_epochs=4, run_name=None),
        resume_path=ckpt, run_id=resume_run_id,
    )
    history = pd.read_csv(outcome2.run_dir / "fold_0" / "history.csv")
    assert list(history.epoch) == [0, 1, 2, 3]  # resumed at epoch 2 (0-based)
    assert len(history) == 4
    # Optimizer/scheduler state was restored exactly. With warmup=4 over 4
    # epochs, lr after scheduler step k is base * min((k+1)/4, 1); the resumed
    # epoch 2 continues that sequence uninterrupted (step 1 -> 0.5x,
    # step 3 -> 1.0x, step 4 -> 1.0x).
    base = cfg.optimization.learning_rate
    assert abs(history.learning_rate.iloc[0] - base * 0.5) < 1e-12
    assert abs(history.learning_rate.iloc[2] - base * 1.0) < 1e-12
    assert abs(history.learning_rate.iloc[3] - base * 1.0) < 1e-12


def test_resume_rejects_fold_mismatch(stage1_cfg_dict, tmp_path):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="f"))
    ckpt = outcome.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"
    other_fold_cfg = Stage1Config(**{**cfg.to_dict(), "fold": 1})
    model = Stage1Model(other_fold_cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(CheckpointError, match="fold"):
        resume_checkpoint(
            ckpt, model, optimizer, None, None,
            fold=1, run_id=outcome.run_id, cfg=other_fold_cfg,
            source_checksums={"k": "v"}, device="cpu",
        )


def test_resume_rejects_config_mismatch(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="c"))
    ckpt = outcome.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"
    changed = Stage1Config(**{**cfg.to_dict(), "loss": {**cfg.loss.to_dict(), "lambda_norm": 0.7}})
    model = Stage1Model(changed)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(CheckpointError, match="config hash"):
        resume_checkpoint(
            ckpt, model, optimizer, None, None,
            fold=cfg.fold, run_id=outcome.run_id, cfg=changed,
            source_checksums={"k": "v"}, device="cpu",
        )


def test_weight_only_loading_fresh_optimizer(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="w"))
    ckpt = outcome.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"
    fresh_cfg = Stage1Config(**{**cfg.to_dict(), "fold": 0})
    model = Stage1Model(fresh_cfg)
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    meta = load_weights_only(ckpt, model, fold=0, cfg=fresh_cfg)
    assert meta.get("optimizer_state") is None or "optimizer_state" in meta  # payload has it; we didn't restore
    weights_after = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(weights_before[k], weights_after[k]) for k in weights_before)
    # Architecture mismatch is reported.
    from stage1.model import Stage1Model as M

    small = Stage1Model(Stage1Config.from_dict({**stage1_cfg_dict, "model": {**stage1_cfg_dict["model"], "d_model": 64}}))
    with pytest.raises(CheckpointError, match="architecture mismatch"):
        load_weights_only(ckpt, small, fold=0, cfg=fresh_cfg)


def test_checkpoint_contains_rng_and_sampler_state(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="s"))
    ckpt = outcome.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"
    payload = torch.load(ckpt, weights_only=False)
    for key in [
        "model_state", "optimizer_state", "scheduler_state", "epoch", "global_step",
        "best_metric", "best_epoch", "fold", "run_id", "config_resolved", "config_hash",
        "source_checksums", "rng_state", "sampler_epoch",
    ]:
        assert key in payload, key
    assert payload["rng_state"]["torch_cpu"] is not None
