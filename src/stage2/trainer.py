"""Stage-2 training engine (guide 07 §5-§16): two-phase training (2A bank
alignment, 2B diagnostic), subject-level validation, durable history with the
epoch commit protocol, best/last checkpoints with exact resume, calibration
and validation artifact export.

The root ``stage2_trainer.py`` stays thin; everything testable lives here.
Epochs are zero-based and global across phases (documented in history.py).
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .attribution import write_stimulus_attributions
from .bank import NormativeBankStore, audit_data_boundary
from .calibration import (
    CalibrationState,
    calibration_disabled_metadata,
    fit_temperature,
)
from .checkpoint import (
    CheckpointError,
    load_stage2_checkpoint,
    load_weights_only_stage2,
    restore_rng_from_checkpoint,
    save_stage2_checkpoint,
    verify_resume_compatibility,
)
from .collate import collate_subject_samples
from .config import Stage2Config
from .dataset import Stage2SubjectDataset
from .history import (
    HistoryWriter,
    atomic_write_bytes,
    row_hash,
    write_epoch_commit,
)
from .losses import (
    bank_rank_loss,
    compute_stage2_losses,
    generate_subset_masks,
    token_match_loss,
    trial_match_loss,
)
from .metrics import best_epoch_rule, is_better_candidate, subject_metrics
from .model import Stage2Model
from .sampler import BalancedSubjectBatchSampler
from .validation import attention_entropy, run_validation

log = logging.getLogger("stage2.trainer")


class TrainerError(ValueError):
    pass


class NonFiniteLossError(TrainerError):
    pass


@dataclass
class TrainerLimits:
    max_train_subjects: int | None = None
    max_val_subjects: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_epochs: int | None = None  # internal smoke/test hook, not a CLI flag

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def make_run_id(
    experiment_name: str, ablation: str, seed: int, config_hash: str, *, is_smoke: bool
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        f"{experiment_name}_{ablation}_seed{seed}_{stamp}_{config_hash[:8]}"
        f"{'_smoke' if is_smoke else ''}"
    )


def git_state() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=root
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, cwd=root
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        dirty = ""
    return {"commit": commit, "dirty_summary": dirty[:2000] or ""}


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }


def seed_everything(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path, json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"
    )


class _LabeledSubset(Dataset):
    """A subject-index subset exposing the labels the sampler needs."""

    def __init__(self, dataset: Any, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self._labels = [dataset.subject_labels()[i] for i in self.indices]
        self.subject_ids = [dataset.subject_ids[i] for i in self.indices]

    def subject_labels(self) -> list[int]:
        return self._labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class _LimitedLoader:
    """Iterates at most ``max_batches`` batches; the exposed batch sampler
    lists exactly those batches so validation subject checks align."""

    def __init__(self, loader: Any, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset
        self.batch_sampler = list(loader.batch_sampler)[:max_batches]

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch


@dataclass
class PhaseState:
    name: str
    phase_epoch: int
    optimizer: Any
    scheduler: Any
    scaler: Any
    optimizer_step_count: int = 0
    skipped_optimizer_step_count: int = 0


class Stage2Trainer:
    """One fold of one run: init metadata, train 2A/2B, commit history, export."""

    def __init__(
        self,
        *,
        cfg: Stage2Config,
        run_root: Path,
        run_id: str,
        config_hash: str,
        ablation_spec: dict[str, Any] | None = None,
        ablation_diff: list[str] | None = None,
        device: str = "cpu",
        limits: TrainerLimits | None = None,
        is_smoke: bool = False,
        deterministic: bool = False,
        resume_path: Path | None = None,
        load_weights_path: Path | None = None,
        dry_run: bool = False,
    ):
        self.cfg = cfg
        self.fold = cfg.fold
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.config_hash = config_hash
        self.ablation_spec = ablation_spec or {}
        self.ablation_diff = ablation_diff or []
        self.device = device
        self.limits = limits or TrainerLimits()
        self.is_smoke = is_smoke
        self.dry_run_mode = dry_run
        self.fold_dir = self.run_root / f"fold_{self.fold}"
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.status = "initializing"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        log_handler = logging.FileHandler(self.fold_dir / "train.log", mode="a")
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logging.getLogger().addHandler(log_handler)

        if deterministic:
            torch.use_deterministic_algorithms(True)

        self.bank_store = NormativeBankStore(cfg, self.fold, device=device)
        self.train_ds = Stage2SubjectDataset(cfg, self.fold, "train", bank_store=self.bank_store)
        self.val_ds = Stage2SubjectDataset(cfg, self.fold, "val", bank_store=self.bank_store)
        if self.limits.max_train_subjects:
            self.train_ds.subjects = self.train_ds.subjects.iloc[
                : self.limits.max_train_subjects
            ].reset_index(drop=True)
            self.train_ds.subject_ids = [str(x) for x in self.train_ds.subjects.subject_id]
            self.train_ds.labels = [int(x) for x in self.train_ds.subjects.label]
        if self.limits.max_val_subjects:
            self.val_ds.subjects = self.val_ds.subjects.iloc[
                : self.limits.max_val_subjects
            ].reset_index(drop=True)
            self.val_ds.subject_ids = [str(x) for x in self.val_ds.subjects.subject_id]
            self.val_ds.labels = [int(x) for x in self.val_ds.subjects.label]

        self.model = Stage2Model(cfg, self.bank_store, device=device)
        # Phase scheduling: no_match_loss/no_bank skip the alignment warm-up.
        self.alignment_epochs = (
            0 if not self.model.bank_features_active else cfg.optimization.alignment_epochs
        )
        self.classification_epochs = cfg.optimization.classification_epochs
        seed_everything(cfg.seed, device)
        self._make_loaders()
        self._write_init_metadata()
        self.history = HistoryWriter(self.fold_dir / "history.csv", self.run_id)
        self.phase: PhaseState | None = None
        self.epoch = 0  # global, zero-based
        self.global_step = 0
        self.best_metric = None
        self.best_epoch: int | None = None
        self.best_rule_tuple: tuple = ()
        self.early_stopping_counter = 0
        self.calibration_state: CalibrationState | None = None

        if resume_path is not None and load_weights_path is not None:
            raise TrainerError("--resume and --load-stage2-weights are mutually exclusive")
        if resume_path is not None:
            self._resume(Path(resume_path))
        elif load_weights_path is not None:
            load_weights_only_stage2(load_weights_path, self.model, device=device)
            log.info(
                "loaded Stage-2 weights only from %s (fresh optimizer/history)",
                load_weights_path,
            )
            self._start_phase_2a()
        else:
            self._start_phase_2a()

        self._shape_check()
        self._write_audits()
        if self.dry_run_mode:
            self._dry_run_once()
            return
        self.status = "running"

    def _dry_run_once(self) -> None:
        """One batch forward/backward (guide §17.20): audit outputs only, no
        production best checkpoint and no history row."""
        self.status = "dry_run"
        batch = next(iter(self.train_loader)).to_device(self.device)
        self.model.train()
        self._set_phase_trainable("2B")
        optimizer, _, _ = self._make_optimizer("2B")
        self.phase = PhaseState("2B", 0, optimizer, None, None)
        subset_masks = None
        if self.cfg.subsets.enabled:
            subset_masks = generate_subset_masks(
                trial_mask=batch.trial_mask,
                category_ids=batch.category_ids,
                subject_ids=list(batch.subject_ids),
                seed=self.cfg.seed,
                fold=self.fold,
                epoch=0,
                train=True,
            )
        optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            enc = self.model.encode_trials(batch, "train")
            full = self.model.aggregate_subject(enc)
            subsets = {
                name: self.model.aggregate_subject(enc, mask)
                for name, mask in (subset_masks or {}).items()
            }
            match = self.model.matching_inputs(batch, enc=enc, epoch=0)
            anchor_current, anchor_stage1 = self.model.transferred_encoder.anchor_vectors()
            if self.cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
                anchor_current = anchor_stage1 = None
            losses = compute_stage2_losses(
                loss_cfg=self.cfg.loss,
                subsets_cfg=self.cfg.subsets,
                labels=batch.labels,
                full=full,
                subsets=subsets,
                match_inputs=match,
                anchor_current=anchor_current,
                anchor_stage1=anchor_stage1,
                epoch=0,
            )
        if not torch.isfinite(losses.total):
            self._write_failure(NonFiniteLossError("non-finite loss in dry run"))
            raise NonFiniteLossError("non-finite loss in dry run")
        losses.total.backward()
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in self.model.parameters() if p.requires_grad
        )
        write_json(
            self.run_root / "run_dry_run.json",
            {
                "run_id": self.run_id,
                "fold": self.fold,
                "ablation": self.cfg.ablation,
                "batch_subjects": batch.n_subjects,
                "valid_trials": batch.n_valid_trials,
                "total_loss": float(losses.total.detach()),
                "gradients_finite": bool(grads_finite),
                "audits_written": sorted(
                    str(p.relative_to(self.fold_dir))
                    for p in (self.fold_dir / "audit").glob("*.json")
                ),
                "best_checkpoint_written": False,
            },
        )
        log.info("dry run complete: 1 batch, no best checkpoint, no history row")

    # ------------------------------------------------------------------- setup

    def _make_loaders(self) -> None:
        # DataLoader iterator construction draws from its generator; a
        # dedicated seeded generator keeps the global torch RNG untouched so
        # exact resume reproduces the next epoch bit-for-bit.
        def loader_generator() -> torch.Generator:
            return torch.Generator().manual_seed(self.cfg.seed * 31 + self.fold * 7)

        self.train_sampler = BalancedSubjectBatchSampler(
            self.train_ds,
            batch_size=self.cfg.sampler.subject_batch_size,
            seed=self.cfg.seed,
            fold=self.fold,
            epoch=0,
            balance_groups=self.cfg.sampler.balance_groups,
            drop_last=self.cfg.sampler.drop_last,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_sampler=self.train_sampler,
            collate_fn=collate_subject_samples,
            num_workers=self.cfg.runtime.num_workers,
            pin_memory=self.cfg.runtime.pin_memory,
            persistent_workers=self.cfg.runtime.persistent_workers,
            generator=loader_generator(),
        )
        self.val_sampler = BalancedSubjectBatchSampler(
            self.val_ds,
            batch_size=self.cfg.sampler.subject_batch_size,
            seed=self.cfg.seed,
            fold=self.fold,
            epoch=0,
            balance_groups=self.cfg.sampler.balance_groups,
            drop_last=False,
            shuffle=False,  # fixed validation order
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_sampler=self.val_sampler,
            collate_fn=collate_subject_samples,
            generator=loader_generator(),
        )
        hc_indices = [
            i for i, label in enumerate(self.train_ds.subject_labels()) if label == 0
        ]
        self.alignment_ds: Dataset | None = None
        self.alignment_loader: DataLoader | None = None
        if hc_indices and self.alignment_epochs > 0:
            self.alignment_ds = _LabeledSubset(self.train_ds, hc_indices)
            self.alignment_sampler = BalancedSubjectBatchSampler(
                self.alignment_ds,
                batch_size=self.cfg.sampler.subject_batch_size,
                seed=self.cfg.seed,
                fold=self.fold,
                epoch=0,
                balance_groups=False,
                drop_last=self.cfg.sampler.drop_last,
                shuffle=True,
            )
            self.alignment_loader = DataLoader(
                self.alignment_ds,
                batch_sampler=self.alignment_sampler,
                collate_fn=collate_subject_samples,
                generator=loader_generator(),
            )

    def _write_init_metadata(self) -> None:
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        (self.fold_dir / "audit").mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            self.run_root / "config_resolved.yaml",
            yaml.safe_dump(self.cfg.to_dict(), sort_keys=True).encode("utf-8"),
        )
        write_json(self.run_root / "environment.json", environment_info())
        self.bank_checksums = {
            "array_sha256": self.bank_store.metadata.get("array_sha256", {}),
            "stage1_checkpoint_sha256": self.bank_store.metadata.get(
                "stage1_checkpoint_sha256"
            ),
        }
        write_json(
            self.run_root / "source_checksums.json",
            {
                "bank_metadata": self.bank_checksums,
                "git": git_state(),
                "config_hash": self.config_hash,
            },
        )
        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "fold": self.fold,
            "seed": self.cfg.seed,
            "ablation": self.cfg.ablation,
            "ablation_spec": self.ablation_spec,
            "ablation_diff": self.ablation_diff,
            "evaluation_regime": self.cfg.evaluation_regime,
            "validation_scope": "outer_fold_exploratory",
            "config_hash": self.config_hash,
            "is_smoke": self.is_smoke,
            "limits": self.limits.to_dict(),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": None,
            "status": self.status,
            "device": self.device,
            "amp_configured": self.cfg.optimization.amp,
            "amp_effective": bool(self.cfg.optimization.amp and self.device == "cuda"),
            "stage1_checkpoint_path": str(self.bank_store.registry_entry["checkpoint"]),
            "stage1_checkpoint_sha256": str(self.bank_store.registry_entry["sha256"]),
            "bank_root": str(self.bank_store.bank_root),
            "num_train_subjects": len(self.train_ds),
            "num_val_subjects": len(self.val_ds),
            "num_train_trials": int(self.train_ds.subjects.number_of_observed_trials.sum()),
            "num_val_trials": int(self.val_ds.subjects.number_of_observed_trials.sum()),
            "parameters": self.model.parameter_report(),
            "epoch_base": "zero-based",
        }
        write_json(self.run_root / "run_metadata.json", self.metadata)

    def _shape_check(self) -> None:
        first_batch = next(iter(self.train_loader)).to_device(self.device)
        self.model.eval()
        with torch.no_grad():
            enc = self.model.encode_trials(first_batch, "train")
            out = self.model.aggregate_subject(enc)
        self.shape_report = {
            "input_heatmaps": list(first_batch.heatmaps.shape),
            "valid_trials": first_batch.n_valid_trials,
            "main_logit": list(out.main_logit.shape),
            "subject_embedding": list(out.subject_embedding.shape),
            "trial_embeddings": list(out.trial_embeddings.shape),
            "semantic_patch_map": (
                list(out.semantic_patch_map.shape)
                if out.semantic_patch_map is not None
                else None
            ),
        }

    def _write_audits(self) -> None:
        audit = audit_data_boundary(self.train_ds, self.val_ds, self.bank_store)
        write_json(self.fold_dir / "audit" / "leakage_checks.json", audit)
        write_json(self.fold_dir / "audit" / "tensor_shapes.json", self.shape_report)
        if audit.get("status") != "ok":
            raise TrainerError(f"data-boundary audit failed: {audit}")

    # ----------------------------------------------------------------- phases

    def _set_phase_trainable(self, phase_name: str) -> None:
        """2A trains only pooler/relation/token branch; 2B trains all Stage-2
        modules plus whatever the ablation unfreezes."""
        all_params = {name: p for name, p in self.model.named_parameters()}
        for name, p in all_params.items():
            p.requires_grad_(False)
        if phase_name == "2A":
            for name, p in all_params.items():
                if name.startswith(("pooler.", "relation.", "token_branch.")):
                    p.requires_grad_(True)
        else:
            encoder_unfreeze = self.cfg.model.encoder_unfreeze_last_block
            for name, p in all_params.items():
                if name.startswith("transferred_encoder.encoder."):
                    p.requires_grad_(
                        encoder_unfreeze
                        and name.startswith(
                            "transferred_encoder.encoder.residual_blocks.1."
                        )
                    )
                elif not name.startswith("transferred_encoder."):
                    p.requires_grad_(True)
        # The frozen encoder must stay in eval mode.
        self.model.transferred_encoder.train(False)
        self.model.train()

    def _make_optimizer(self, phase_name: str) -> tuple[Any, Any, Any]:
        stage2_params = []
        encoder_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("transferred_encoder.encoder."):
                encoder_params.append(p)
            else:
                stage2_params.append(p)
        groups = [
            {
                "params": stage2_params,
                "lr": self.cfg.optimization.learning_rate,
                "weight_decay": self.cfg.optimization.weight_decay,
            }
        ]
        if encoder_params:
            groups.append(
                {
                    "params": encoder_params,
                    "lr": self.cfg.optimization.encoder_learning_rate,
                    "weight_decay": self.cfg.optimization.weight_decay,
                }
            )
        optimizer = torch.optim.AdamW(groups)
        steps_per_epoch = max(len(self.train_loader), 1)
        n_epochs = {
            "2A": self.alignment_epochs,
            "2B": self.classification_epochs,
        }[phase_name]
        total_steps = max(n_epochs, 1) * steps_per_epoch
        warmup_steps = min(
            self.cfg.optimization.warmup_epochs * steps_per_epoch, total_steps
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = None
        if self.cfg.optimization.amp and self.device == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        return optimizer, scheduler, scaler

    def _start_phase_2a(self) -> None:
        if self.alignment_epochs <= 0:
            self._start_phase_2b()
            return
        self._set_phase_trainable("2A")
        optimizer, scheduler, scaler = self._make_optimizer("2A")
        self.phase = PhaseState("2A", 0, optimizer, scheduler, scaler)
        log.info("Phase 2A (bank alignment warm-up): %d epochs", self.alignment_epochs)

    def _start_phase_2b(self) -> None:
        self._set_phase_trainable("2B")
        optimizer, scheduler, scaler = self._make_optimizer("2B")
        self.phase = PhaseState("2B", 0, optimizer, scheduler, scaler)
        log.info("Phase 2B (diagnostic training): %d epochs", self.classification_epochs)

    # -------------------------------------------------------------- objectives

    def _alignment_objective(
        self, batch: Any, epoch: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        enc = self.model.encode_trials(batch, "train")
        match = self.model.matching_inputs(batch, enc=enc, epoch=epoch)
        if match is None:
            zero = enc.q.sum() * 0.0
            return zero, {"trialmatch": 0.0, "bankrank": 0.0, "tokenmatch": 0.0}
        trialmatch = trial_match_loss(
            match.cos_pos, match.cos_neg, match.hc_match_mask, self.cfg.loss.match_margin
        )
        bankrank = bank_rank_loss(
            match.comparator_pos,
            match.comparator_neg,
            match.hc_match_mask,
            self.cfg.loss.bank_rank_margin,
        )
        if match.Q is not None:
            tokenmatch = token_match_loss(
                match.Q, match.N_mu_pos, match.N_mu_neg, match.token_rho,
                match.token_omega, match.hc_match_mask, self.cfg.loss.token_match_margin,
            )
            total = trialmatch + 0.5 * bankrank + 0.25 * tokenmatch
        else:
            tokenmatch = enc.q.sum() * 0.0
            total = trialmatch + 0.5 * bankrank
        return total, {
            "trialmatch": float(trialmatch.detach()),
            "bankrank": float(bankrank.detach()),
            "tokenmatch": float(tokenmatch.detach()),
        }

    # --------------------------------------------------------------- training

    def _step(self, loss: torch.Tensor) -> float:
        """Backward, unscale, measure pre-clip norm, clip, step (guide §8.1)."""
        assert self.phase is not None
        if self.phase.scaler is not None:
            self.phase.scaler.scale(loss).backward()
            self.phase.scaler.unscale_(self.phase.optimizer)
        else:
            loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.grad is not None],
            self.cfg.optimization.gradient_clip_norm,
        )
        if self.phase.scaler is not None:
            self.phase.scaler.step(self.phase.optimizer)
            self.phase.scaler.update()
        else:
            self.phase.optimizer.step()
        self.phase.scheduler.step()
        self.phase.optimizer_step_count += 1
        self.global_step += 1
        return float(total_norm)

    def _autocast(self):
        if self.cfg.optimization.amp and self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _run_2a_epoch(self, epoch: int) -> dict[str, Any]:
        assert self.phase is not None and self.alignment_loader is not None
        self.alignment_sampler.set_epoch(epoch)
        sums: dict[str, float] = {}
        n_subjects = 0
        n_trials = 0
        norm_list: list[float] = []
        clip_steps = 0
        attempted = 0
        self.model.train()
        max_batches = self.limits.max_train_batches
        for batch_idx, batch in enumerate(self.alignment_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to_device(self.device)
            n = batch.n_subjects
            n_subjects += n
            n_trials += batch.n_valid_trials
            self.phase.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss, components = self._alignment_objective(batch, epoch)
            if not torch.isfinite(loss):
                raise NonFiniteLossError(
                    f"non-finite 2A loss at epoch {epoch} batch {batch_idx}"
                )
            total_norm = self._step(loss)
            norm_list.append(total_norm)
            attempted += 1
            if total_norm > self.cfg.optimization.gradient_clip_norm:
                clip_steps += 1
            for key, value in components.items():
                sums[key] = sums.get(key, 0.0) + value * n
        if n_subjects == 0:
            raise TrainerError("2A epoch processed no subjects")
        return {
            "match": sums.get("trialmatch", 0.0) / n_subjects
            + 0.5 * sums.get("bankrank", 0.0) / n_subjects
            + 0.25 * sums.get("tokenmatch", 0.0) / n_subjects,
            "trialmatch": sums.get("trialmatch", 0.0) / n_subjects,
            "bankrank": sums.get("bankrank", 0.0) / n_subjects,
            "tokenmatch": sums.get("tokenmatch", 0.0) / n_subjects,
            "grad_norm_mean": float(np.mean(norm_list)) if norm_list else 0.0,
            "grad_norm_max": float(np.max(norm_list)) if norm_list else 0.0,
            "grad_clip_fraction": clip_steps / attempted if attempted else 0.0,
            "n_subjects": n_subjects,
            "n_trials": n_trials,
            "n_batches": attempted,
        }

    def _run_2b_epoch(self, epoch: int) -> dict[str, Any]:
        assert self.phase is not None
        self.train_sampler.set_epoch(epoch)
        sums: dict[str, float] = {}
        components_sum: dict[str, float] = {}
        n_subjects = 0
        n_trials = 0
        norm_list: list[float] = []
        clip_steps = 0
        attempted = 0
        train_logits: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        entropy_sum = 0.0
        matched_cos_sum = 0.0
        wrong_cos_sum = 0.0
        rank_acc_sum = 0.0
        n_hc_match = 0
        self.model.train()
        max_batches = self.limits.max_train_batches
        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to_device(self.device)
            n = batch.n_subjects
            n_subjects += n
            n_trials += batch.n_valid_trials
            subset_masks = None
            if self.cfg.subsets.enabled:
                subset_masks = generate_subset_masks(
                    trial_mask=batch.trial_mask,
                    category_ids=batch.category_ids,
                    subject_ids=list(batch.subject_ids),
                    seed=self.cfg.seed,
                    fold=self.fold,
                    epoch=epoch,
                    min_fraction=self.cfg.subsets.min_fraction,
                    max_fraction=self.cfg.subsets.max_fraction,
                    train=True,
                )
            self.phase.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                enc = self.model.encode_trials(batch, "train")
                full = self.model.aggregate_subject(enc)
                subsets = {
                    name: self.model.aggregate_subject(enc, mask)
                    for name, mask in (subset_masks or {}).items()
                }
                match = self.model.matching_inputs(batch, enc=enc, epoch=epoch)
                anchor_current, anchor_stage1 = (
                    self.model.transferred_encoder.anchor_vectors()
                )
                if self.cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
                    anchor_current = anchor_stage1 = None
                losses = compute_stage2_losses(
                    loss_cfg=self.cfg.loss,
                    subsets_cfg=self.cfg.subsets,
                    labels=batch.labels,
                    full=full,
                    subsets=subsets,
                    match_inputs=match,
                    anchor_current=anchor_current,
                    anchor_stage1=anchor_stage1,
                    epoch=epoch,
                )
            if not torch.isfinite(losses.total):
                raise NonFiniteLossError(
                    f"non-finite 2B loss at epoch {epoch} batch {batch_idx}: "
                    f"total {float(losses.total)}"
                )
            total_norm = self._step(losses.total)
            norm_list.append(total_norm)
            attempted += 1
            if total_norm > self.cfg.optimization.gradient_clip_norm:
                clip_steps += 1

            sums["total"] = sums.get("total", 0.0) + float(losses.total.detach()) * n
            for key in (
                "cls", "aux", "match", "trialmatch", "bankrank", "tokenmatch",
                "cons", "latent_cons", "prob_cons", "entropy", "anchor",
            ):
                components_sum[key] = (
                    components_sum.get(key, 0.0) + float(getattr(losses, key).detach()) * n
                )
            train_logits.append(full.main_logit.detach().clone())
            train_labels.append(batch.labels.detach().clone())
            entropy = attention_entropy(full.stimulus_attention, full.trial_mask)
            entropy_sum += float(entropy.mean()) * n
            if match is not None and int(match.hc_match_mask.sum()) > 0:
                hc_mask = match.hc_match_mask
                n_hc = int(hc_mask.sum())
                n_hc_match += n_hc
                matched_cos_sum += float(match.cos_pos[hc_mask].mean()) * n_hc
                wrong_cos_sum += float(match.cos_neg[hc_mask].mean()) * n_hc
                rank_acc_sum += float(
                    (match.comparator_pos[hc_mask] > match.comparator_neg[hc_mask])
                    .float().mean()
                ) * n_hc

        if n_subjects == 0:
            raise TrainerError("2B epoch processed no subjects")
        train_metrics = subject_metrics(
            torch.cat(train_labels).numpy(),
            torch.cat(train_logits).numpy(),
            torch.sigmoid(torch.cat(train_logits)).numpy(),
        )
        return {
            "total": sums["total"] / n_subjects,
            **{k: components_sum[k] / n_subjects for k in components_sum},
            "grad_norm_mean": float(np.mean(norm_list)) if norm_list else 0.0,
            "grad_norm_max": float(np.max(norm_list)) if norm_list else 0.0,
            "grad_clip_fraction": clip_steps / attempted if attempted else 0.0,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_attention_entropy": entropy_sum / n_subjects,
            "train_matched_cosine": matched_cos_sum / n_hc_match if n_hc_match else None,
            "train_wrong_cosine": wrong_cos_sum / n_hc_match if n_hc_match else None,
            "bank_rank_accuracy": rank_acc_sum / n_hc_match if n_hc_match else None,
            "n_subjects": n_subjects,
            "n_trials": n_trials,
            "n_batches": attempted,
        }

    # ---------------------------------------------------------------- epochs

    def _val_loader_for_epoch(self) -> Any:
        if self.limits.max_val_batches is not None:
            return _LimitedLoader(self.val_loader, self.limits.max_val_batches)
        return self.val_loader

    def _run_epoch(self, epoch: int) -> tuple[dict[str, Any], Any]:
        t0 = time.time()
        assert self.phase is not None
        loader = self._val_loader_for_epoch()
        if self.phase.name == "2A":
            summary = self._run_2a_epoch(epoch)
            val_result = run_validation(self.model, loader, cfg=self.cfg, epoch=epoch)
            summary.update(
                {
                    "val_loss": val_result.mean_losses.get("val_match_loss", 0.0),
                    "eligible_for_best": False,  # alignment epochs never select the final model
                }
            )
        else:
            summary = self._run_2b_epoch(epoch)
            val_result = run_validation(self.model, loader, cfg=self.cfg, epoch=epoch)
            summary.update(
                {
                    "val_loss": val_result.mean_losses.get("val_loss", 0.0),
                    "val_cls_loss": val_result.mean_losses.get("val_cls_loss", 0.0),
                    "val_aux_loss": val_result.mean_losses.get("val_aux_loss", 0.0),
                    "val_match_loss": val_result.mean_losses.get("val_match_loss", 0.0),
                    "val_cons_loss": val_result.mean_losses.get("val_cons_loss", 0.0),
                    "val_accuracy": val_result.metrics["accuracy"],
                    "val_balanced_accuracy": val_result.metrics["balanced_accuracy"],
                    "val_auroc": val_result.metrics["auroc"],
                    "val_f1": val_result.metrics["f1"],
                    "val_sensitivity": val_result.metrics["sensitivity"],
                    "val_specificity": val_result.metrics["specificity"],
                    "val_brier": val_result.metrics["brier"],
                    "val_attention_entropy": val_result.attention_entropy_mean,
                    "val_matched_cosine": val_result.matched_cosine_mean,
                    "val_wrong_cosine": val_result.wrong_cosine_mean,
                    "eligible_for_best": True,
                }
            )
        summary["epoch_time_seconds"] = time.time() - t0
        return summary, val_result

    def _commit_epoch(self, epoch: int, summary: dict[str, Any], val_result: Any) -> None:
        """Epoch commit protocol (guide §12.1): checkpoint, history, marker."""
        assert self.phase is not None
        lrs = [group["lr"] for group in self.phase.optimizer.param_groups]
        is_eligible = bool(summary.get("eligible_for_best", False))
        is_best = False
        if is_eligible:
            candidate = best_epoch_rule(
                summary.get("val_balanced_accuracy"),
                summary.get("val_auroc"),
                summary["val_loss"],
                epoch,
            )
            if not self.best_rule_tuple or is_better_candidate(candidate, self.best_rule_tuple):
                self.best_rule_tuple = candidate
                self.best_metric = summary.get("val_balanced_accuracy")
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                is_best = True
            else:
                self.early_stopping_counter += 1

        phase_epoch = self.phase.phase_epoch
        row = {
            "run_id": self.run_id,
            "fold": self.fold,
            "epoch": epoch,
            "phase_epoch": phase_epoch,
            "global_step": self.global_step,
            "training_phase": self.phase.name,
            "validation_scope": "outer_fold_exploratory",
            "eligible_for_best": is_eligible,
            "is_best_epoch": is_best,
            "best_epoch_so_far": self.best_epoch if self.best_epoch is not None else "",
            "learning_rate": lrs[0],
            "encoder_learning_rate": lrs[1] if len(lrs) > 1 else "",
            "learning_rate_min": min(lrs),
            "learning_rate_max": max(lrs),
            "weight_decay": self.cfg.optimization.weight_decay,
            "lambda_aux": self.cfg.loss.lambda_aux,
            "lambda_match": self.cfg.loss.lambda_match,
            "lambda_cons": self.cfg.loss.lambda_cons,
            "lambda_entropy": self.cfg.loss.lambda_entropy,
            "lambda_anchor": self.cfg.loss.lambda_anchor,
            "train_loss": summary.get("total", ""),
            "train_cls_loss": summary.get("cls", ""),
            "train_aux_loss": summary.get("aux", ""),
            "train_match_loss": summary.get("match", ""),
            "train_trialmatch_loss": summary.get("trialmatch", ""),
            "train_bankrank_loss": summary.get("bankrank", ""),
            "train_tokenmatch_loss": summary.get("tokenmatch", ""),
            "train_cons_loss": summary.get("cons", ""),
            "train_entropy_loss": summary.get("entropy", ""),
            "train_anchor_loss": summary.get("anchor", ""),
            "val_loss": summary.get("val_loss", ""),
            "val_cls_loss": summary.get("val_cls_loss", ""),
            "val_aux_loss": summary.get("val_aux_loss", ""),
            "val_match_loss": summary.get("val_match_loss", ""),
            "val_cons_loss": summary.get("val_cons_loss", ""),
            "train_accuracy": summary.get("train_accuracy", ""),
            "train_balanced_accuracy": summary.get("train_balanced_accuracy", ""),
            "val_accuracy": summary.get("val_accuracy", ""),
            "val_balanced_accuracy": summary.get("val_balanced_accuracy", ""),
            "val_auroc": summary.get("val_auroc", ""),
            "val_f1": summary.get("val_f1", ""),
            "val_sensitivity": summary.get("val_sensitivity", ""),
            "val_specificity": summary.get("val_specificity", ""),
            "val_brier": summary.get("val_brier", ""),
            "train_attention_entropy": summary.get("train_attention_entropy", ""),
            "val_attention_entropy": summary.get("val_attention_entropy", ""),
            "train_matched_cosine": summary.get("train_matched_cosine", ""),
            "train_wrong_cosine": summary.get("train_wrong_cosine", ""),
            "val_matched_cosine": summary.get("val_matched_cosine", ""),
            "val_wrong_cosine": summary.get("val_wrong_cosine", ""),
            "bank_rank_accuracy": summary.get("bank_rank_accuracy", ""),
            "grad_norm_mean": summary.get("grad_norm_mean", ""),
            "grad_norm_max": summary.get("grad_norm_max", ""),
            "grad_clip_fraction": summary.get("grad_clip_fraction", ""),
            "optimizer_step_count": self.phase.optimizer_step_count,
            "skipped_optimizer_step_count": self.phase.skipped_optimizer_step_count,
            "num_train_subjects": summary.get("n_subjects", ""),
            "num_val_subjects": val_result.n_subjects if val_result else "",
            "num_train_trials": summary.get("n_trials", ""),
            "num_val_trials": int(val_result.trial_mask.sum()) if val_result else "",
            "n_train_batches": summary.get("n_batches", ""),
            "n_val_batches": "",
            "nonfinite_batch_count": 0,
            "epoch_time_seconds": summary.get("epoch_time_seconds", ""),
            "peak_gpu_memory_mb": 0,
            "seed": self.cfg.seed,
        }
        row_hash_value = row_hash(row)

        checkpoint_kwargs = dict(
            model=self.model,
            optimizer=self.phase.optimizer,
            scheduler=self.phase.scheduler,
            scaler=self.phase.scaler,
            epoch=epoch,
            phase_epoch=phase_epoch,
            global_step=self.global_step,
            training_phase=self.phase.name,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            best_rule_tuple=self.best_rule_tuple,
            early_stopping_counter=self.early_stopping_counter,
            optimizer_step_count=self.phase.optimizer_step_count,
            skipped_optimizer_step_count=self.phase.skipped_optimizer_step_count,
            fold=self.fold,
            run_id=self.run_id,
            cfg=self.cfg,
            config_hash=self.config_hash,
            source_checksums=self.bank_store.metadata.get("source_checksums", {}),
            stage1_checkpoint_path=str(self.bank_store.registry_entry["checkpoint"]),
            stage1_checkpoint_sha256=str(self.bank_store.registry_entry["sha256"]),
            bank_manifest_path=str(self.bank_store.bank_root / "manifest.json"),
            bank_checksums=self.bank_checksums,
            ablation_spec=self.ablation_spec,
            ablation_diff=self.ablation_diff,
            history_row_hash=row_hash_value,
            calibration_state=(
                self.calibration_state.to_dict() if self.calibration_state else None
            ),
            device=self.device,
            sampler_epoch=epoch,
        )
        ckpt_dir = self.fold_dir / "checkpoints"
        last_path = ckpt_dir / f"last_stage2_fold{self.fold}.pt"
        last_sha = save_stage2_checkpoint(last_path, **checkpoint_kwargs)
        if is_best:
            save_stage2_checkpoint(
                ckpt_dir / f"best_stage2_fold{self.fold}.pt", **checkpoint_kwargs
            )
        self.history.append_row(row)
        write_epoch_commit(
            self.fold_dir / "epoch_commit.json",
            epoch=epoch,
            history_row_hash=row_hash_value,
            checkpoint_sha256=last_sha,
        )
        log.info(
            "epoch committed: %d (phase %s/%d, best=%s, row %s…, checkpoint %s…)",
            epoch, self.phase.name, phase_epoch, self.best_epoch,
            row_hash_value[:12], last_sha[:12],
        )

    # ------------------------------------------------------------------ train

    def train(self) -> dict[str, Any]:
        """Run 2A then 2B to completion; export final validation artifacts."""
        log.info(
            "starting fold %d (run %s, regime %s, validation_scope=outer_fold_exploratory)",
            self.fold, self.run_id, self.cfg.evaluation_regime,
        )
        stop_reason = "max_epochs"
        try:
            while True:
                if self.phase is None:
                    raise TrainerError("no active training phase")
                if self.limits.max_epochs is not None and self.epoch >= self.limits.max_epochs:
                    stop_reason = "max_epochs_limit"
                    break
                if self.phase.name == "2A" and self.phase.phase_epoch >= self.alignment_epochs:
                    self._start_phase_2b()
                    continue
                if self.phase.name == "2B" and self.phase.phase_epoch >= self.classification_epochs:
                    break
                summary, val_result = self._run_epoch(self.epoch)
                self._commit_epoch(self.epoch, summary, val_result)
                self.phase.phase_epoch += 1
                self.epoch += 1
                if (
                    self.phase.name == "2B"
                    and self.early_stopping_counter >= self.cfg.validation.early_stopping_patience
                ):
                    stop_reason = "early_stopping"
                    log.info(
                        "early stopping after epoch %d (patience %d)",
                        self.epoch - 1, self.cfg.validation.early_stopping_patience,
                    )
                    break
        except TrainerError as exc:
            self._write_failure(exc)
            raise
        self._export_final_artifacts()
        self.status = "completed"
        self.finished_at_utc = datetime.now(timezone.utc).isoformat()
        self.metadata["status"] = self.status
        self.metadata["finished_at_utc"] = self.finished_at_utc
        self.metadata["stop_reason"] = stop_reason
        write_json(self.run_root / "run_metadata.json", self.metadata)
        return {
            "stop_reason": stop_reason,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
        }

    def _write_failure(self, error: Exception) -> None:
        write_json(
            self.run_root / "run_failure.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "phase": self.phase.name if self.phase else None,
                "epoch": self.epoch,
                "last_committed_epoch": (
                    self.history.epochs_recorded()[-1] if self.history.n_rows else None
                ),
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ------------------------------------------------------------------ export

    def _export_final_artifacts(self) -> None:
        val_dir = self.fold_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.fold_dir / "checkpoints" / f"best_stage2_fold{self.fold}.pt"
        if not best_path.is_file():
            raise TrainerError("no best checkpoint to export validation artifacts from")
        contents = load_stage2_checkpoint(best_path, device=self.device)
        self.model.load_state_dict(contents.state_dict)
        self.model.eval()
        val_result = run_validation(self.model, self.val_loader, cfg=self.cfg, epoch=0)
        metrics = dict(val_result.metrics)
        metrics["validation_scope"] = "outer_fold_exploratory"
        metrics["evaluation_regime"] = self.cfg.evaluation_regime
        write_json(val_dir / "metrics.json", metrics)

        # Calibration on training-partition predictions only (pilot scope).
        if self.cfg.validation.calibrate:
            train_logits, train_labels, train_ids = self._collect_train_predictions()
            if train_logits.numel() >= 2 and len(set(train_labels.tolist())) >= 2:
                self.calibration_state = fit_temperature(
                    train_logits,
                    train_labels,
                    subject_ids=train_ids,
                    fit_scope="outer_training_partition_pilot",
                )
                write_json(val_dir / "calibration.json", self.calibration_state.to_dict())
            else:
                write_json(val_dir / "calibration.json", calibration_disabled_metadata())
        else:
            write_json(val_dir / "calibration.json", calibration_disabled_metadata())

        temperature = (
            self.calibration_state.temperature if self.calibration_state is not None else 1.0
        )
        calibrated_p = torch.sigmoid(val_result.raw_logits / temperature).numpy()
        rows = []
        for i, sid in enumerate(val_result.subject_ids):
            rows.append(
                {
                    "subject_id": sid,
                    "fold": self.fold,
                    "split_scope": "outer_fold_exploratory",
                    "label": int(val_result.labels[i]),
                    "raw_logit": float(val_result.raw_logits[i]),
                    "uncalibrated_probability": float(val_result.probabilities[i]),
                    "calibrated_probability": float(calibrated_p[i]),
                    "threshold": 0.5,
                    "prediction": int(calibrated_p[i] >= 0.5),
                    "is_correct": int(
                        (calibrated_p[i] >= 0.5) == (int(val_result.labels[i]) == 1)
                    ),
                    "num_available_stimuli": int(val_result.trial_mask[i].sum()),
                }
            )
        df = pd.DataFrame(rows)
        tmp = val_dir / "subject_predictions.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(val_dir / "subject_predictions.parquet")
        write_stimulus_attributions(val_dir / "stimulus_attributions.npz", val_result)
        write_json(
            self.fold_dir / "audit" / "bank_match_metrics.json",
            {
                "val_matched_cosine_mean": val_result.matched_cosine_mean,
                "val_wrong_cosine_mean": val_result.wrong_cosine_mean,
                "val_bank_rank_accuracy": val_result.bank_rank_accuracy,
                "n_hc_val_subjects": val_result.metrics["n_hc"],
            },
        )

    def _collect_train_predictions(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        logits: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        ids: list[str] = []
        self.model.eval()
        with torch.inference_mode():
            for batch in self.train_loader:
                batch = batch.to_device(self.device)
                enc = self.model.encode_trials(batch, "train")
                out = self.model.aggregate_subject(enc)
                logits.append(out.main_logit.clone())
                labels.append(batch.labels.long().clone())
                ids.extend(list(batch.subject_ids))
        return torch.cat(logits), torch.cat(labels), ids

    # ------------------------------------------------------------------ resume

    def _resume(self, path: Path) -> None:
        contents = load_stage2_checkpoint(path, device=self.device)
        verify_resume_compatibility(
            contents,
            cfg=self.cfg,
            fold=self.fold,
            run_id=self.run_id,
            config_hash=self.config_hash,
            ablation_name=self.cfg.ablation,
            stage1_checkpoint_sha256=str(self.bank_store.registry_entry["sha256"]),
            bank_checksums=self.bank_checksums,
        )
        self.model.load_state_dict(contents.state_dict)
        meta = contents.meta
        self.epoch = int(meta["epoch"]) + 1
        self.best_metric = meta.get("best_metric")
        self.best_epoch = meta.get("best_epoch")
        self.best_rule_tuple = tuple(meta.get("best_rule_tuple") or ())
        self.early_stopping_counter = int(meta.get("early_stopping_counter", 0))
        training_phase = str(meta["training_phase"])
        self._set_phase_trainable(training_phase)
        optimizer, scheduler, scaler = self._make_optimizer(training_phase)
        optimizer.load_state_dict(meta["optimizer_state"])
        scheduler.load_state_dict(meta["scheduler_state"])
        if scaler is not None and meta.get("scaler_state"):
            scaler.load_state_dict(meta["scaler_state"])
        self.global_step = int(meta["global_step"])
        self.phase = PhaseState(
            training_phase,
            int(meta["phase_epoch"]) + 1,
            optimizer,
            scheduler,
            scaler,
            int(meta.get("optimizer_step_count", 0)),
            int(meta.get("skipped_optimizer_step_count", 0)),
        )
        restore_rng_from_checkpoint(contents)
        self.train_sampler.set_epoch(self.epoch)

        # History repair (guide §12.1): checkpoint ahead of history -> rerun
        # deterministic validation and reconstruct the missing row; history
        # ahead of the checkpoint -> corrupted state, never truncated.
        recorded = self.history.epochs_recorded()
        if recorded and recorded[-1] > int(meta["epoch"]):
            raise TrainerError(
                f"corrupted state: history ahead of checkpoint "
                f"({recorded[-1]} > {meta['epoch']}) — refusing to truncate"
            )
        if not recorded or recorded[-1] < int(meta["epoch"]):
            log.info(
                "repairing history: checkpoint epoch %s ahead of recorded %s",
                meta["epoch"], recorded[-1] if recorded else "none",
            )
            self.model.eval()
            val_result = run_validation(
                self.model, self.val_loader, cfg=self.cfg, epoch=int(meta["epoch"])
            )
            row = {
                "run_id": self.run_id,
                "fold": self.fold,
                "epoch": int(meta["epoch"]),
                "phase_epoch": int(meta["phase_epoch"]),
                "global_step": int(meta["global_step"]),
                "training_phase": training_phase,
                "validation_scope": "outer_fold_exploratory",
                "eligible_for_best": training_phase == "2B",
                "is_best_epoch": False,
                "best_epoch_so_far": self.best_epoch if self.best_epoch is not None else "",
                "learning_rate": "", "encoder_learning_rate": "",
                "learning_rate_min": "", "learning_rate_max": "",
                "weight_decay": self.cfg.optimization.weight_decay,
                "lambda_aux": self.cfg.loss.lambda_aux,
                "lambda_match": self.cfg.loss.lambda_match,
                "lambda_cons": self.cfg.loss.lambda_cons,
                "lambda_entropy": self.cfg.loss.lambda_entropy,
                "lambda_anchor": self.cfg.loss.lambda_anchor,
                "train_loss": "", "train_cls_loss": "", "train_aux_loss": "",
                "train_match_loss": "", "train_trialmatch_loss": "",
                "train_bankrank_loss": "", "train_tokenmatch_loss": "",
                "train_cons_loss": "", "train_entropy_loss": "", "train_anchor_loss": "",
                "val_loss": val_result.mean_losses.get("val_loss", 0.0),
                "val_cls_loss": val_result.mean_losses.get("val_cls_loss", 0.0),
                "val_aux_loss": val_result.mean_losses.get("val_aux_loss", 0.0),
                "val_match_loss": val_result.mean_losses.get("val_match_loss", 0.0),
                "val_cons_loss": val_result.mean_losses.get("val_cons_loss", 0.0),
                "train_accuracy": "", "train_balanced_accuracy": "",
                "val_accuracy": val_result.metrics["accuracy"],
                "val_balanced_accuracy": val_result.metrics["balanced_accuracy"],
                "val_auroc": val_result.metrics["auroc"],
                "val_f1": val_result.metrics["f1"],
                "val_sensitivity": val_result.metrics["sensitivity"],
                "val_specificity": val_result.metrics["specificity"],
                "val_brier": val_result.metrics["brier"],
                "train_attention_entropy": "",
                "val_attention_entropy": val_result.attention_entropy_mean,
                "train_matched_cosine": "", "train_wrong_cosine": "",
                "val_matched_cosine": val_result.matched_cosine_mean,
                "val_wrong_cosine": val_result.wrong_cosine_mean,
                "bank_rank_accuracy": "",
                "grad_norm_mean": "", "grad_norm_max": "", "grad_clip_fraction": "",
                "optimizer_step_count": int(meta.get("optimizer_step_count", 0)),
                "skipped_optimizer_step_count": int(meta.get("skipped_optimizer_step_count", 0)),
                "num_train_subjects": "", "num_val_subjects": val_result.n_subjects,
                "num_train_trials": "", "num_val_trials": int(val_result.trial_mask.sum()),
                "n_train_batches": "", "n_val_batches": "",
                "nonfinite_batch_count": 0,
                "epoch_time_seconds": 0.0,
                "peak_gpu_memory_mb": 0,
                "seed": self.cfg.seed,
            }
            self.history.append_row(row)
            write_epoch_commit(
                self.fold_dir / "epoch_commit.json",
                epoch=int(meta["epoch"]),
                history_row_hash=meta.get("history_row_hash") or row_hash(row),
                checkpoint_sha256=contents.sha256,
            )
        log.info("resumed from %s at epoch %d (phase %s)", path, self.epoch, training_phase)
