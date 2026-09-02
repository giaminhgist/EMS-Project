"""History writer tests (guide 07 §12, §17.5, §17.10)."""

from __future__ import annotations

import csv
import json

import pytest

from stage2.history import (
    HISTORY_COLUMNS,
    HistoryWriter,
    atomic_write_bytes,
    read_epoch_commit,
    row_hash,
    write_epoch_commit,
)


def make_row(**overrides) -> dict:
    row = {c: "" for c in HISTORY_COLUMNS}
    row.update(
        {
            "run_id": "test_run",
            "fold": 0,
            "epoch": 0,
            "phase_epoch": 0,
            "global_step": 0,
            "training_phase": "2B",
            "validation_scope": "outer_fold_exploratory",
            "eligible_for_best": True,
            "is_best_epoch": True,
            "best_epoch_so_far": 0,
            "learning_rate": 1e-4,
            "seed": 2026,
        }
    )
    row.update(overrides)
    return row


def test_stable_column_set_is_complete():
    # Contract §22 fields plus the execution-state fields all exist.
    required = {
        "run_id", "fold", "epoch", "phase_epoch", "global_step", "training_phase",
        "validation_scope", "eligible_for_best", "is_best_epoch", "best_epoch_so_far",
        "learning_rate", "encoder_learning_rate", "learning_rate_min", "learning_rate_max",
        "weight_decay", "lambda_aux", "lambda_match", "lambda_cons", "lambda_entropy",
        "lambda_anchor", "train_loss", "train_cls_loss", "train_aux_loss",
        "train_match_loss", "train_trialmatch_loss", "train_bankrank_loss",
        "train_tokenmatch_loss", "train_cons_loss", "train_entropy_loss",
        "train_anchor_loss", "val_loss", "val_cls_loss", "val_aux_loss",
        "val_match_loss", "val_cons_loss", "train_accuracy", "train_balanced_accuracy",
        "val_accuracy", "val_balanced_accuracy", "val_auroc", "val_f1",
        "val_sensitivity", "val_specificity", "val_brier",
        "train_attention_entropy", "val_attention_entropy",
        "train_matched_cosine", "train_wrong_cosine",
        "val_matched_cosine", "val_wrong_cosine", "bank_rank_accuracy",
        "grad_norm_mean", "grad_norm_max", "grad_clip_fraction",
        "optimizer_step_count", "skipped_optimizer_step_count",
        "num_train_subjects", "num_val_subjects", "num_train_trials", "num_val_trials",
        "n_train_batches", "n_val_batches", "nonfinite_batch_count",
        "epoch_time_seconds", "peak_gpu_memory_mb", "seed",
    }
    assert required <= set(HISTORY_COLUMNS)
    assert len(set(HISTORY_COLUMNS)) == len(HISTORY_COLUMNS)


def test_append_rows_and_rewrite_atomically(tmp_path):
    path = tmp_path / "history.csv"
    writer = HistoryWriter(path, "test_run")
    writer.append_row(make_row(epoch=0, val_balanced_accuracy=0.5))
    writer.append_row(make_row(epoch=1, val_balanced_accuracy=0.75))
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(HISTORY_COLUMNS)
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert not list(tmp_path.glob("*.tmp")), "no partial temp file may remain"


def test_reopen_preserves_rows(tmp_path):
    path = tmp_path / "history.csv"
    writer = HistoryWriter(path, "test_run")
    writer.append_row(make_row(epoch=0))
    writer2 = HistoryWriter(path, "test_run")
    assert writer2.n_rows == 1
    assert writer2.epochs_recorded() == [0]


def test_epochs_must_increase_monotonically(tmp_path):
    writer = HistoryWriter(tmp_path / "history.csv", "test_run")
    writer.append_row(make_row(epoch=0))
    with pytest.raises(ValueError, match="monotonically"):
        writer.append_row(make_row(epoch=0))


def test_missing_and_unknown_columns_rejected(tmp_path):
    writer = HistoryWriter(tmp_path / "history.csv", "test_run")
    with pytest.raises(ValueError, match="missing columns"):
        writer.append_row({"epoch": 0})
    with pytest.raises(ValueError, match="unknown columns"):
        writer.append_row(make_row(epoch=0, bogus_column=1))


def test_row_hash_is_deterministic():
    assert row_hash(make_row(epoch=0)) == row_hash(make_row(epoch=0))
    assert row_hash(make_row(epoch=0)) != row_hash(make_row(epoch=1))


def test_epoch_commit_protocol_written_last(tmp_path):
    commit_path = tmp_path / "epoch_commit.json"
    write_epoch_commit(
        commit_path, epoch=3, history_row_hash="abc123", checkpoint_sha256="def456"
    )
    payload = read_epoch_commit(commit_path)
    assert payload == {
        "epoch": 3,
        "history_row_hash": "abc123",
        "checkpoint_sha256": "def456",
    }


def test_atomic_write_leaves_no_temp_files_on_failure(tmp_path):
    target = tmp_path / "data.json"
    atomic_write_bytes(target, b'{"ok": true}')
    assert target.read_bytes() == b'{"ok": true}'
    # A writer-level failure after a good commit leaves the last good state
    # and no partial temp files behind.
    writer = HistoryWriter(tmp_path / "history.csv", "test_run")
    writer.append_row(make_row(epoch=0))
    with pytest.raises(ValueError, match="unknown columns"):
        writer.append_row(make_row(epoch=1, bogus_column=1))
    assert writer.n_rows == 1  # the bad row was never accepted
    assert list(tmp_path.glob("*.tmp")) == []
    with (tmp_path / "history.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 1
