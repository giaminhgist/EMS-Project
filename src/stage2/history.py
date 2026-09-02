"""Durable per-epoch history (guide 07 §12, contracts §22).

One row is appended atomically immediately after every successfully completed
epoch; the column set is fixed from the start (no column may appear or
disappear after epoch 0). Epochs are zero-based globally across training
phases. Unused loss components are stored as empty strings while their
columns remain present.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

HISTORY_COLUMNS: tuple[str, ...] = (
    # Identity and scheduling
    "run_id", "fold", "epoch", "phase_epoch", "global_step", "training_phase",
    "validation_scope", "eligible_for_best", "is_best_epoch", "best_epoch_so_far",
    # Optimization
    "learning_rate", "encoder_learning_rate", "learning_rate_min", "learning_rate_max",
    "weight_decay", "lambda_aux", "lambda_match", "lambda_cons", "lambda_entropy",
    "lambda_anchor",
    # Train losses
    "train_loss", "train_cls_loss", "train_aux_loss", "train_match_loss",
    "train_trialmatch_loss", "train_bankrank_loss", "train_tokenmatch_loss",
    "train_cons_loss", "train_entropy_loss", "train_anchor_loss",
    # Validation losses
    "val_loss", "val_cls_loss", "val_aux_loss", "val_match_loss", "val_cons_loss",
    # Metrics
    "train_accuracy", "train_balanced_accuracy",
    "val_accuracy", "val_balanced_accuracy", "val_auroc", "val_f1",
    "val_sensitivity", "val_specificity", "val_brier",
    # Interpretation diagnostics
    "train_attention_entropy", "val_attention_entropy",
    "train_matched_cosine", "train_wrong_cosine",
    "val_matched_cosine", "val_wrong_cosine", "bank_rank_accuracy",
    # Execution state
    "grad_norm_mean", "grad_norm_max", "grad_clip_fraction",
    "optimizer_step_count", "skipped_optimizer_step_count",
    # Counts
    "num_train_subjects", "num_val_subjects", "num_train_trials", "num_val_trials",
    "n_train_batches", "n_val_batches", "nonfinite_batch_count",
    "epoch_time_seconds", "peak_gpu_memory_mb", "seed",
)

EPOCH_BASE = 0  # zero-based global epochs, documented here


class HistoryError(ValueError):
    pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # fsync of directories is not supported everywhere


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write through a temporary sibling, flush, fsync, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def row_hash(row: dict[str, Any]) -> str:
    canonical = json.dumps(
        {k: row.get(k, "") for k in HISTORY_COLUMNS}, sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoryWriter:
    """Append-only CSV over the stable column set with atomic rewrites."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
                    raise HistoryError(
                        f"{self.path}: history columns differ from the stable contract"
                    )
                for row in reader:
                    if row.get("run_id") not in ("", self.run_id):
                        raise HistoryError(
                            f"{self.path}: history run_id {row.get('run_id')!r} "
                            f"!= {self.run_id!r}"
                        )
                    self.rows.append(row)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def epochs_recorded(self) -> list[int]:
        return [int(row["epoch"]) for row in self.rows]

    def append_row(self, row: dict[str, Any]) -> str:
        """Append one complete row atomically and return its hash."""
        missing = [c for c in HISTORY_COLUMNS if c not in row]
        if missing:
            raise HistoryError(f"history row missing columns: {missing}")
        extra = [c for c in row if c not in HISTORY_COLUMNS]
        if extra:
            raise HistoryError(f"history row has unknown columns: {extra}")
        if self.rows and int(row["epoch"]) <= int(self.rows[-1]["epoch"]):
            raise HistoryError(
                f"epochs must increase monotonically: "
                f"{row['epoch']} after {self.rows[-1]['epoch']}"
            )
        self.rows.append({c: row[c] for c in HISTORY_COLUMNS})
        self._rewrite()
        return row_hash(row)

    def _rewrite(self) -> None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in self.rows:
            writer.writerow({c: row.get(c, "") for c in HISTORY_COLUMNS})
        atomic_write_bytes(self.path, buf.getvalue().encode("utf-8"))


def write_epoch_commit(
    path: Path, *, epoch: int, history_row_hash: str, checkpoint_sha256: str
) -> None:
    """The commit marker is written last (guide 07 §12.1)."""
    payload = {
        "epoch": epoch,
        "history_row_hash": history_row_hash,
        "checkpoint_sha256": checkpoint_sha256,
    }
    atomic_write_bytes(
        path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def read_epoch_commit(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))
