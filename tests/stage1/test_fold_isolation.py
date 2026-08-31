"""Fold isolation: artifacts never leak across folds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1.checkpoint import CheckpointError, load_weights_only
from stage1.config import Stage1Config
from stage1.model import Stage1Model
from stage1.trainer import TrainLimits, run_training


def _smoke_cfg(stage1_cfg_dict):
    return Stage1Config.from_dict(
        {
            **stage1_cfg_dict,
            "loss": {**stage1_cfg_dict["loss"], "norm_start_epoch": 1, "norm_ramp_epochs": 1},
            "optimization": {**stage1_cfg_dict["optimization"], "epochs": 2, "amp": False},
        }
    )


def test_fold0_checkpoint_rejected_for_fold1(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=1, run_name="iso"))
    ckpt = outcome.run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"

    fold1_cfg = Stage1Config(**{**cfg.to_dict(), "fold": 1})
    model = Stage1Model(fold1_cfg)
    with pytest.raises(CheckpointError, match="fold"):
        load_weights_only(ckpt, model, fold=1, cfg=fold1_cfg)


def test_history_rows_never_mix_folds(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=2, run_name="mix"))
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert history.fold.nunique() == 1
    assert int(history.fold.iloc[0]) == cfg.fold


def test_datasets_filter_hc_before_sampling(stage1_cfg_dict):
    from stage1.dataset import Stage1Dataset
    from stage1.sampler import StimulusGroupedHCBatchSampler

    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    sampler = StimulusGroupedHCBatchSampler(
        ds.stimulus_groups(),
        stimuli_per_batch=2, hc_per_stimulus=2, min_hc_per_stimulus=2,
        seed=cfg.seed, fold=cfg.fold,
    )
    for batch in sampler:
        for i in batch:
            assert ds[i].group == "HC"


def test_validation_embeddings_carry_no_validation_subjects_into_norm_bank(stage1_cfg_dict):
    # The normative bank builder receives the train dataset only; the trainer
    # passes validation subject IDs as forbidden. Verify via a direct run that
    # val subjects (013) are not in the train dataset.
    from stage1.dataset import Stage1Dataset

    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    train_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    train_subjects = set(train_ds.trial_rows.subject_id.astype(str))
    val_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "val"
    )
    val_subjects = set(val_ds.trial_rows.subject_id.astype(str))
    assert not (train_subjects & val_subjects)
