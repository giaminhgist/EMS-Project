"""Best-checkpoint policy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from stage1.config import Stage1Config
from stage1.trainer import TrainLimits, run_training
from stage1.validation import best_checkpoint_eligible, effective_lambda_norm


def _smoke_cfg(stage1_cfg_dict, norm_start=1, ramp=1):
    return Stage1Config.from_dict(
        {
            **stage1_cfg_dict,
            "loss": {
                **stage1_cfg_dict["loss"],
                "norm_start_epoch": norm_start,
                "norm_ramp_epochs": ramp,
            },
            "optimization": {**stage1_cfg_dict["optimization"], "epochs": 5, "amp": False},
        }
    )


def test_effective_lambda_schedule():
    from stage1.config import Stage1Config as C

    cfg = C.from_dict(
        {"loss": {"lambda_norm": 0.2, "norm_start_epoch": 10, "norm_ramp_epochs": 5}}
    )
    assert effective_lambda_norm(cfg, 9) == 0.0
    assert effective_lambda_norm(cfg, 10) == pytest.approx(0.04)  # 1/5 ramp
    assert effective_lambda_norm(cfg, 14) == pytest.approx(0.2)  # 5/5 ramp
    assert effective_lambda_norm(cfg, 20) == pytest.approx(0.2)
    assert best_checkpoint_eligible(cfg, 14) is False
    assert best_checkpoint_eligible(cfg, 15) is True


def test_best_not_selected_before_eligibility(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict, norm_start=3, ramp=2)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=4, run_name="nelig"))
    ckpt_dir = outcome.run_dir / "fold_0" / "checkpoints"
    assert not (ckpt_dir / f"best_stage1_fold{cfg.fold}.pt").exists()
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert history.eligible_for_best.tolist() == [False, False, False, False]
    assert not history.is_best_epoch.any()
    assert history.best_epoch_so_far.isna().all()


def test_is_best_and_best_epoch_consistent(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict, norm_start=0, ramp=0)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="best"))
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert history.eligible_for_best.all()
    best_epochs = history.loc[history.is_best_epoch, "epoch"].tolist()
    assert best_epochs
    # is_best_epoch rows carry best_epoch_so_far == their own epoch.
    for e in best_epochs:
        assert int(history.loc[history.epoch == e, "best_epoch_so_far"].iloc[0]) == e
    # The final best_epoch_so_far equals the max of best epochs.
    final_best = int(history.best_epoch_so_far.iloc[-1])
    assert final_best == max(best_epochs)
    # Best checkpoint exists with the canonical name and is loadable.
    import torch

    ckpt_dir = outcome.run_dir / "fold_0" / "checkpoints"
    payload = torch.load(ckpt_dir / f"best_stage1_fold{cfg.fold}.pt", weights_only=False)
    assert payload["best_epoch"] == final_best
