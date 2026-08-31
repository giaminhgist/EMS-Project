"""History durability tests: per-epoch commits and interruption safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stage1.config import Stage1Config
from stage1.history import HISTORY_COLUMNS, HistoryError, read_history, write_history
from stage1.trainer import TrainLimits, TrainingError, run_training


def _smoke_cfg(stage1_cfg_dict):
    return Stage1Config.from_dict(
        {
            **stage1_cfg_dict,
            "loss": {**stage1_cfg_dict["loss"], "norm_start_epoch": 1, "norm_ramp_epochs": 1},
            "optimization": {**stage1_cfg_dict["optimization"], "epochs": 3, "amp": False},
        }
    )


def test_history_written_after_each_epoch(stage1_cfg_dict, monkeypatch):
    """history.csv exists after epoch 1, before training completes."""
    cfg = _smoke_cfg(stage1_cfg_dict)
    seen: dict[int, pd.DataFrame] = {}

    from stage1 import history as history_mod

    original = history_mod.write_history

    def spy(path, rows):
        original(path, rows)
        if path.name == "history.csv" and len(rows) == 1:
            seen[1] = pd.read_csv(path)

    monkeypatch.setattr(history_mod, "write_history", spy)
    monkeypatch.setattr("stage1.trainer.write_history", spy)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="h"))
    assert 1 in seen  # epoch 1's commit happened while training continued
    row = seen[1].iloc[0]
    assert int(row.epoch) == 0
    assert len(read_history(outcome.run_dir / "fold_0" / "history.csv")) == 3


def test_every_completed_epoch_exactly_one_row(stage1_cfg_dict):
    cfg = _smoke_cfg(stage1_cfg_dict)
    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="rows"))
    rows = read_history(outcome.run_dir / "fold_0" / "history.csv")
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2]
    history_df = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert list(history_df.columns) == HISTORY_COLUMNS


def test_spread_loss_committed_and_val_loss_composition(stage1_cfg_dict):
    """Smoke run: spread columns exist, are finite, and val_loss recomposes."""
    cfg = _smoke_cfg(stage1_cfg_dict)  # norm_start=1, ramp=1 -> spread active from epoch 1
    from stage1.validation import effective_lambda_norm, effective_lambda_spread

    outcome = run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="spread"))
    history = pd.read_csv(outcome.run_dir / "fold_0" / "history.csv")
    assert {"train_spread_loss", "val_spread_loss"} <= set(history.columns)
    for _, row in history.iterrows():
        assert np.isfinite(row.train_spread_loss)
        assert np.isfinite(row.val_spread_loss)
        epoch = int(row.epoch)
        expected_val = (
            row.val_recon_loss
            + effective_lambda_norm(cfg, epoch) * row.val_norm_loss
            + effective_lambda_spread(cfg, epoch) * row.val_spread_loss
        )
        assert row.val_loss == pytest.approx(expected_val, abs=1e-5)


def test_history_validates_uniqueness_and_monotonicity(tmp_path):
    base = {
        "run_id": "r", "fold": 0, "epoch": 0, "training_phase": "warmup",
        "eligible_for_best": False, "is_best_epoch": False, "best_epoch_so_far": None,
        "learning_rate": 1e-3, "learning_rate_min": 1e-3, "learning_rate_max": 1e-3,
        "weight_decay": 0.0, "lambda_norm": 0.0, "train_loss": 0.1,
        "train_recon_loss": 0.1, "train_recon_fixation": 0.1, "train_recon_transition": 0.0,
        "train_recon_temporal": 0.0, "train_norm_loss": 0.0, "val_loss": 0.2,
        "val_recon_loss": 0.2, "val_recon_fixation": 0.2, "val_recon_transition": 0.0,
        "val_recon_temporal": 0.0, "val_norm_loss": 0.0,
        "train_within_stimulus_dispersion": 0.0, "train_between_stimulus_dispersion": 0.0,
        "val_within_stimulus_dispersion": 0.0, "val_between_stimulus_dispersion": 0.0,
        "grad_norm_mean": 1.0, "grad_norm_max": 1.0, "grad_clip_fraction": 0.0,
        "semantic_gamma_attention1": 0.05, "semantic_gamma_attention2": 0.05,
        "spatial_bridge_eta": 0.1, "train_mask_ratio_realized": 0.35,
        "val_mask_ratio_realized": 0.35, "num_train_trials": 4, "num_val_trials": 1,
        "n_train_batches": 1, "n_val_batches": 1, "n_train_stimulus_groups": 2,
        "n_val_stimulus_groups": 1, "n_skipped_norm_groups_train": 0,
        "n_skipped_norm_groups_val": 1, "nonfinite_batch_count": 0,
        "epoch_time_seconds": 1.0, "peak_gpu_memory_mb": 0.0, "seed": 2026,
    }
    row2 = {**base, "epoch": 1}
    write_history(tmp_path / "history.csv", [base, row2])
    # Duplicate epoch is rejected.
    with pytest.raises(HistoryError, match="duplicate"):
        write_history(tmp_path / "history.csv", [base, base])
    # Non-monotonic order is rejected.
    with pytest.raises(HistoryError, match="monotonic"):
        write_history(tmp_path / "history.csv", [row2, base])


def test_interrupted_training_keeps_previous_history_and_checkpoint(stage1_cfg_dict, monkeypatch):
    """A simulated failure mid-training leaves the last completed epoch intact."""
    cfg = _smoke_cfg(stage1_cfg_dict)
    from stage1 import trainer as trainer_mod

    original = trainer_mod.train_one_epoch
    calls = {"n": 0}

    def failing_train(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "train_one_epoch", failing_train)
    with pytest.raises(TrainingError, match="simulated interruption"):
        run_training(cfg, device="cpu", limits=TrainLimits(max_epochs=3, run_name="int"))
    # Find the run dir: the error path still created it (run dir initialized).
    output_root = cfg.paths.output_root
    run_dirs = sorted(output_root.glob("*int*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    history = pd.read_csv(run_dir / "fold_0" / "history.csv")
    assert len(history) == 1  # only epoch 0 committed; no false completed epoch
    assert (run_dir / "fold_0" / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt").is_file()
    assert (run_dir / "fold_0" / "error_report.json").is_file()
