"""Trainer smoke tests on synthetic fixtures (guide 07 §17): epochs train and
validate, history/checkpoint commit, best-rule eligibility, exact resume
equivalence, weight-only init, non-finite hard failure and dry-run behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from conftest import make_full_model_fixture
from stage2.checkpoint import CheckpointError, load_stage2_checkpoint
from stage2.config import Stage2Config, config_hash
from stage2.history import HISTORY_COLUMNS
from stage2.trainer import (
    NonFiniteLossError,
    Stage2Trainer,
    TrainerLimits,
)


def make_cfg(stack, *, alignment_epochs=0, classification_epochs=2):
    raw = stack["cfg"].to_dict()
    raw["optimization"]["alignment_epochs"] = alignment_epochs
    raw["optimization"]["classification_epochs"] = classification_epochs
    raw["validation"]["calibrate"] = False
    cfg = Stage2Config.from_dict(raw)
    return cfg, config_hash(cfg.to_dict())


def make_trainer(
    tmp_path,
    *,
    alignment_epochs=0,
    classification_epochs=2,
    limits=None,
    resume=None,
    load_weights=None,
    dry_run=False,
    stack_dir=None,
):
    stack = make_full_model_fixture(stack_dir or tmp_path / "stack")
    cfg, cfg_hash = make_cfg(
        stack, alignment_epochs=alignment_epochs,
        classification_epochs=classification_epochs,
    )
    run_root = tmp_path / "run"
    trainer = Stage2Trainer(
        cfg=cfg,
        run_root=run_root,
        run_id="synthetic_test_run",
        config_hash=cfg_hash,
        device="cpu",
        limits=limits or TrainerLimits(),
        resume_path=Path(resume) if resume else None,
        load_weights_path=Path(load_weights) if load_weights else None,
        dry_run=dry_run,
    )
    return stack, trainer


def read_history(run_root: Path) -> pd.DataFrame:
    return pd.read_csv(run_root / "fold_0" / "history.csv")


def test_one_synthetic_epoch_trains_and_validates(tmp_path):
    stack, trainer = make_trainer(
        tmp_path, classification_epochs=2, limits=TrainerLimits(max_epochs=1)
    )
    summary = trainer.train()
    assert summary["best_epoch"] == 0
    run_root = tmp_path / "run"
    history = read_history(run_root)
    assert len(history) == 1
    assert list(history.columns) == list(HISTORY_COLUMNS)
    row = history.iloc[0]
    assert row["training_phase"] == "2B"
    assert bool(row["eligible_for_best"]) is True
    assert bool(row["is_best_epoch"]) is True
    assert str(row["val_balanced_accuracy"]) not in ("", "nan")
    # Checkpoints, commit marker and validation artifacts exist.
    ckpt_dir = run_root / "fold_0" / "checkpoints"
    assert (ckpt_dir / "last_stage2_fold0.pt").is_file()
    assert (ckpt_dir / "best_stage2_fold0.pt").is_file()
    commit = json.loads((run_root / "fold_0" / "epoch_commit.json").read_text())
    assert commit["epoch"] == 0
    assert commit["history_row_hash"] and commit["checkpoint_sha256"]
    val_dir = run_root / "fold_0" / "validation"
    for name in (
        "metrics.json", "subject_predictions.parquet",
        "stimulus_attributions.npz", "calibration.json",
    ):
        assert (val_dir / name).is_file(), name
    assert (run_root / "run_metadata.json").is_file()
    assert (run_root / "config_resolved.yaml").is_file()
    assert (run_root / "fold_0" / "audit" / "leakage_checks.json").is_file()
    assert (run_root / "fold_0" / "audit" / "tensor_shapes.json").is_file()
    meta = json.loads((run_root / "run_metadata.json").read_text())
    assert meta["validation_scope"] == "outer_fold_exploratory"
    assert meta["status"] == "completed"
    assert meta["stop_reason"] == "max_epochs_limit"


def test_alignment_epochs_are_ineligible_for_best(tmp_path):
    _, trainer = make_trainer(
        tmp_path, alignment_epochs=1, classification_epochs=1,
        limits=TrainerLimits(max_epochs=2),
    )
    trainer.train()
    history = read_history(tmp_path / "run")
    assert len(history) == 2
    assert history.iloc[0]["training_phase"] == "2A"
    assert bool(history.iloc[0]["eligible_for_best"]) is False
    assert bool(history.iloc[0]["is_best_epoch"]) is False
    assert str(history.iloc[0]["train_match_loss"]) not in ("", "nan")
    assert history.iloc[1]["training_phase"] == "2B"
    assert bool(history.iloc[1]["eligible_for_best"]) is True
    assert bool(history.iloc[1]["is_best_epoch"]) is True  # only eligible epoch
    assert int(history.iloc[1]["best_epoch_so_far"]) == 1


def test_exact_resume_reproduces_next_epoch(tmp_path):
    # Run A: uninterrupted 3 epochs.
    _, trainer_a = make_trainer(tmp_path / "A", classification_epochs=3)
    trainer_a.train()
    history_a = read_history(tmp_path / "A" / "run")
    assert list(history_a["epoch"]) == [0, 1, 2]

    # Run B: identical fixture, stopped after 2 epochs.
    _, trainer_b = make_trainer(
        tmp_path / "B", classification_epochs=3, limits=TrainerLimits(max_epochs=2)
    )
    trainer_b.train()
    b_last = tmp_path / "B" / "run" / "fold_0" / "checkpoints" / "last_stage2_fold0.pt"

    # Resume B and finish epoch 2. Exact resume reuses the identical
    # configuration (paths included in the hash), run root and run id.
    trainer_b2 = Stage2Trainer(
        cfg=trainer_b.cfg,
        run_root=tmp_path / "B" / "run",
        run_id="synthetic_test_run",
        config_hash=trainer_b.config_hash,
        device="cpu",
        resume_path=b_last,
    )
    assert trainer_b2.epoch == 2  # next epoch after the checkpoint
    trainer_b2.train()
    history_b = read_history(tmp_path / "B" / "run")
    assert list(history_b["epoch"]) == [0, 1, 2]

    # The resumed epoch's validation metrics match uninterrupted training.
    val_a = history_a[history_a["epoch"] == 2].iloc[0]["val_balanced_accuracy"]
    val_b = history_b[history_b["epoch"] == 2].iloc[0]["val_balanced_accuracy"]
    assert val_a == val_b
    loss_a = history_a[history_a["epoch"] == 2].iloc[0]["val_loss"]
    loss_b = history_b[history_b["epoch"] == 2].iloc[0]["val_loss"]
    assert loss_a == loss_b

    # The resumed model parameters match the uninterrupted run exactly.
    final_a = load_stage2_checkpoint(
        tmp_path / "A" / "run" / "fold_0" / "checkpoints" / "last_stage2_fold0.pt"
    )
    final_b = load_stage2_checkpoint(
        tmp_path / "B" / "run" / "fold_0" / "checkpoints" / "last_stage2_fold0.pt"
    )
    assert set(final_a.state_dict) == set(final_b.state_dict)
    for key in final_a.state_dict:
        assert torch.equal(final_a.state_dict[key], final_b.state_dict[key]), key


def test_incompatible_resume_fails(tmp_path):
    _, trainer = make_trainer(
        tmp_path, classification_epochs=2, limits=TrainerLimits(max_epochs=1)
    )
    trainer.train()
    last = tmp_path / "run" / "fold_0" / "checkpoints" / "last_stage2_fold0.pt"
    # A different configuration (3 epochs) -> different config hash -> resume
    # must be rejected rather than silently continuing.
    with pytest.raises(CheckpointError, match="resume rejected"):
        make_trainer(tmp_path / "C", classification_epochs=3, resume=str(last))


def test_weight_only_init_starts_fresh_history(tmp_path):
    _, trainer = make_trainer(
        tmp_path, classification_epochs=1, limits=TrainerLimits(max_epochs=1)
    )
    trainer.train()
    last = tmp_path / "run" / "fold_0" / "checkpoints" / "last_stage2_fold0.pt"
    _, trainer2 = make_trainer(
        tmp_path / "D", classification_epochs=1, limits=TrainerLimits(max_epochs=1),
        load_weights=str(last),
    )
    assert trainer2.history.n_rows == 0  # fresh history before training
    trainer2.train()
    history = read_history(tmp_path / "D" / "run")
    assert len(history) == 1
    assert history.iloc[0]["epoch"] == 0
    # The model was initialized from the previous run's tensors.
    assert trainer2.epoch == 1


def test_nonfinite_loss_fails_hard(tmp_path):
    stack, trainer = make_trainer(
        tmp_path, classification_epochs=2, limits=TrainerLimits(max_epochs=1)
    )
    with torch.no_grad():
        trainer.model.pooler.score.weight.fill_(float("nan"))
    with pytest.raises(NonFiniteLossError):
        trainer.train()
    failure = json.loads((tmp_path / "run" / "run_failure.json").read_text())
    assert failure["exception_type"] == "NonFiniteLossError"
    assert failure["last_committed_epoch"] is None
    assert (tmp_path / "run" / "run_failure.json").is_file()


def test_dry_run_writes_audits_but_no_checkpoint(tmp_path):
    _, trainer = make_trainer(tmp_path, dry_run=True)
    run_root = tmp_path / "run"
    assert (run_root / "run_dry_run.json").is_file()
    dry = json.loads((run_root / "run_dry_run.json").read_text())
    assert dry["best_checkpoint_written"] is False
    assert dry["gradients_finite"] is True
    assert (run_root / "fold_0" / "audit" / "tensor_shapes.json").is_file()
    assert (run_root / "fold_0" / "audit" / "leakage_checks.json").is_file()
    assert not (run_root / "fold_0" / "checkpoints").exists()
    assert not (run_root / "fold_0" / "history.csv").exists()


def test_predictions_and_attributions_share_id_order(tmp_path):
    _, trainer = make_trainer(
        tmp_path, classification_epochs=1, limits=TrainerLimits(max_epochs=1)
    )
    trainer.train()
    val_dir = tmp_path / "run" / "fold_0" / "validation"
    df = pd.read_parquet(val_dir / "subject_predictions.parquet")
    npz = dict(
        __import__("numpy").load(val_dir / "stimulus_attributions.npz", allow_pickle=True)
    )
    assert list(df["subject_id"]) == list(npz["subject_ids"])
    assert df["fold"].tolist() == [0] * len(df)
    assert set(df.columns) >= {
        "subject_id", "fold", "split_scope", "label", "raw_logit",
        "uncalibrated_probability", "calibrated_probability", "threshold",
        "prediction", "is_correct", "num_available_stimuli",
    }
