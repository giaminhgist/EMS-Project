"""Stage-1 training loop (contract §3/§4): phases A/B/C, validation, history,
checkpoints, best-selection, and exact resume."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_weights_only,
    resume_checkpoint,
    save_checkpoint,
)
from .config import Stage1Config
from .dataset import Stage1Dataset
from .experiment import initialize_run_dir, make_run_id, source_checksums
from .history import read_history, row_from_metrics, write_history
from .losses import stage1_loss
from .masking import training_token_masks
from .model import Stage1Model, summarize_model
from .reproducibility import seed_all
from .sampler import StimulusGroupedHCBatchSampler
from .validation import (
    best_checkpoint_eligible,
    effective_lambda_norm,
    effective_lambda_spread,
    run_validation,
)

logger = logging.getLogger("stage1.trainer")


class TrainingError(ValueError):
    pass


@dataclass
class TrainLimits:
    max_epochs: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    run_name: str | None = None


@dataclass
class TrainOutcome:
    run_id: str
    run_dir: Path
    fold: int
    epochs_completed: int
    best_epoch: int | None
    best_val_loss: float | None
    history_rows: int
    notes: list[str] = field(default_factory=list)


def sha256_of_file(path: Path) -> str:
    from preprocessing.storage import sha256_of_file as _sha

    return _sha(path)


def _build_datasets(
    cfg: Stage1Config, device: str
) -> tuple[Stage1Dataset, Stage1Dataset, StimulusGroupedHCBatchSampler]:
    kwargs = dict(
        verify_checksums=True,
        fold=cfg.fold,
        active_channels=tuple(cfg.model.active_channels),
    )
    train_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train", **kwargs
    )
    val_ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "val", **kwargs
    )
    sampler = StimulusGroupedHCBatchSampler(
        train_ds.stimulus_groups(),
        stimuli_per_batch=cfg.sampler.stimuli_per_batch,
        hc_per_stimulus=cfg.sampler.hc_per_stimulus,
        min_hc_per_stimulus=cfg.sampler.min_hc_per_stimulus,
        replacement=cfg.sampler.replacement,
        seed=cfg.seed,
        fold=cfg.fold,
        epoch=0,
        subject_by_row=train_ds.row_subject_ids(),
    )
    return train_ds, val_ds, sampler


def _make_optimizer_scheduler_scaler(
    cfg: Stage1Config, model: torch.nn.Module, device: str
) -> tuple[Any, Any, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimization.learning_rate, weight_decay=cfg.optimization.weight_decay
    )
    steps = max(cfg.optimization.epochs, 1)
    warmup = min(cfg.optimization.lr_warmup_epochs, steps)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp_enabled = cfg.optimization.amp and (
        device == "cuda" or torch.is_autocast_enabled("cpu")
    )
    scaler = torch.amp.GradScaler(device, enabled=amp_enabled) if amp_enabled else None
    return optimizer, scheduler, scaler


def _autocast_context(device: str, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    if device == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)


def train_one_epoch(
    model: Any,
    train_ds: Stage1Dataset,
    sampler: StimulusGroupedHCBatchSampler,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    cfg: Stage1Config,
    epoch: int,
    device: str,
    max_train_batches: int | None,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(epoch)
    lambda_norm = effective_lambda_norm(cfg, epoch)
    lambda_spread = effective_lambda_spread(cfg, epoch)
    amp_enabled = scaler is not None
    autocast = _autocast_context(device, amp_enabled)

    total_loss = 0.0
    recon_loss = 0.0
    recon_ch = np.zeros(3)
    norm_loss = 0.0
    spread_loss = 0.0
    within_disp = 0.0
    between_disp = 0.0
    n_trials = 0
    n_batches = 0
    n_skipped = 0
    nonfinite_count = 0
    grad_norms: list[float] = []
    grad_clip_count = 0
    realized_masks: list[float] = []
    n_groups = 0

    for batch_idx, indices in enumerate(sampler):
        if max_train_batches is not None and n_batches >= max_train_batches:
            break
        batch = train_ds.collate_from_indices(indices)
        batch.to_device(device)
        mask = training_token_masks(
            batch.n_trials, cfg.masking.train_mask_ratio, cfg.seed, cfg.fold, batch_idx
        ).to(device)

        if not (torch.all(torch.isfinite(batch.heatmaps)) and torch.all(torch.isfinite(batch.unique_dino_tokens))):
            raise TrainingError(f"epoch {epoch} batch {batch_idx}: non-finite inputs")

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            out = model(batch, mask)
            losses = stage1_loss(
                out.reconstruction, batch.heatmaps, mask,
                out.trial_embedding, batch.trial_to_stimulus_slot,
                lambda_norm=lambda_norm,
                lambda_spread=lambda_spread,
                spread_floor=cfg.loss.spread_floor,
                reconstruction_loss=cfg.loss.reconstruction,
                channel_weights=tuple(cfg.loss.channel_weights),
                channel_map=tuple(cfg.model.active_channels),
                scope=cfg.masking.reconstruction_scope,
                min_hc_per_stimulus=2,
            )
        if not torch.isfinite(losses.total):
            nonfinite_count += 1
            continue  # skip the update; the epoch remains valid
        if scaler is not None:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
        else:
            losses.total.backward()
        # Detect non-finite gradients before clipping/stepping.
        grads_finite = all(
            p.grad is None or torch.all(torch.isfinite(p.grad)) for p in model.parameters()
        )
        if not grads_finite:
            nonfinite_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is None and cfg.optimization.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.gradient_clip_norm)

        total_norm = float(
            sum(
                (p.grad.detach().norm().item() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        grad_norms.append(total_norm)
        if scaler is not None and cfg.optimization.gradient_clip_norm > 0:
            before = total_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.gradient_clip_norm)
            after = float(
                sum(
                    (p.grad.detach().norm().item() ** 2)
                    for p in model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            if after < before - 1e-9:
                grad_clip_count += 1
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        n = batch.n_trials
        total_loss += float(losses.total.item()) * n
        recon_loss += float(losses.reconstruction.item()) * n
        recon_ch[0] += float(losses.recon_fixation.item()) * n
        recon_ch[1] += float(losses.recon_transition.item()) * n
        recon_ch[2] += float(losses.recon_temporal.item()) * n
        norm_loss += float(losses.normative.item()) * n
        spread_loss += float(losses.spread_loss.item()) * n
        within_disp += losses.within_stimulus_dispersion * n
        between_disp += losses.between_stimulus_dispersion * n
        n_skipped += losses.n_skipped_norm_groups
        n_groups += losses.details.get("n_norm_groups", 0)
        realized_masks.append(float(mask.float().mean()))
        n_trials += n
        n_batches += 1

    if n_trials == 0:
        raise TrainingError(f"epoch {epoch}: no training batches produced")
    return {
        "train_loss": total_loss / n_trials,
        "train_recon_loss": recon_loss / n_trials,
        "train_recon_fixation": recon_ch[0] / n_trials,
        "train_recon_transition": recon_ch[1] / n_trials,
        "train_recon_temporal": recon_ch[2] / n_trials,
        "train_norm_loss": norm_loss / n_trials,
        "train_spread_loss": spread_loss / n_trials,
        "train_within_stimulus_dispersion": within_disp / n_trials,
        "train_between_stimulus_dispersion": between_disp / n_trials,
        "num_train_trials": n_trials,
        "n_train_batches": n_batches,
        "n_train_stimulus_groups": n_groups,
        "n_skipped_norm_groups_train": n_skipped,
        "nonfinite_batch_count": nonfinite_count,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
        "grad_clip_fraction": grad_clip_count / max(n_batches, 1),
        "train_mask_ratio_realized": float(np.mean(realized_masks)) if realized_masks else 0.0,
    }


def run_training(
    cfg: Stage1Config,
    *,
    device: str = "cpu",
    limits: TrainLimits | None = None,
    resume_path: Path | None = None,
    weights_path: Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    max_epochs: int | None = None,
) -> TrainOutcome:
    """Run (or resume) training for one fold; returns the outcome."""
    limits = limits or TrainLimits(max_epochs=max_epochs)
    seed_all(cfg.seed)

    train_ds, val_ds, sampler = _build_datasets(cfg, device)
    model = Stage1Model(cfg)
    model.to(device)

    if dry_run:
        summary = summarize_model(model)
        return TrainOutcome(
            run_id=run_id or "(dry-run)",
            run_dir=Path("."),
            fold=cfg.fold,
            epochs_completed=0,
            best_epoch=None,
            best_val_loss=None,
            history_rows=0,
            notes=[
                f"dry-run: {summary.n_parameters_total} trainable parameters",
                f"dry-run: train HC trials {len(train_ds)}, val HC trials {len(val_ds)}",
                f"dry-run: sampler batches {len(sampler)}, eligible stimuli {len(sampler.eligible_stimuli)}",
            ],
        )

    checksums = source_checksums(cfg)
    if resume_path is not None:
        # Exact resume continues the checkpoint's own run directory.
        ckpt_contents = load_checkpoint(resume_path, device)
        run_id = str(ckpt_contents.meta["run_id"])
        run_dir = cfg.paths.output_root / run_id
        if not run_dir.is_dir():
            raise TrainingError(f"run directory for resume not found: {run_dir}")
        existing_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if existing_meta.get("config_hash") != cfg.config_hash():
            raise TrainingError("resume config hash does not match the run directory")
    else:
        if run_id is None:
            run_id = make_run_id(cfg.experiment_name, cfg.ablation, limits.run_name)
        run_dir = initialize_run_dir(cfg.paths.output_root, run_id, cfg, {})
    fold_dir = run_dir / f"fold_{cfg.fold}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "validation").mkdir(parents=True, exist_ok=True)

    optimizer, scheduler, scaler = _make_optimizer_scheduler_scaler(cfg, model, device)

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    history_rows = read_history(fold_dir / "history.csv")

    if weights_path is not None:
        load_weights_only(weights_path, model, fold=cfg.fold, cfg=cfg, device=device)
        logger.info("loaded weights from %s (optimizer/scheduler fresh)", weights_path)
    if resume_path is not None:
        meta = resume_checkpoint(
            resume_path, model, optimizer, scheduler, scaler,
            fold=cfg.fold, run_id=run_id, cfg=cfg,
            source_checksums=checksums, device=device,
        )
        start_epoch = int(meta["epoch"]) + 1
        global_step = int(meta.get("global_step", 0))
        best_metric = meta.get("best_metric")
        best_epoch = meta.get("best_epoch")
        # History must be consistent with the checkpoint epoch.
        expected_rows = start_epoch
        if len(history_rows) != expected_rows:
            raise TrainingError(
                f"history has {len(history_rows)} rows but checkpoint resumes at epoch {start_epoch}"
            )

    best_ckpt_path = fold_dir / "checkpoints" / f"best_stage1_fold{cfg.fold}.pt"
    last_ckpt_path = fold_dir / "checkpoints" / f"last_stage1_fold{cfg.fold}.pt"

    total_epochs = limits.max_epochs if limits.max_epochs is not None else cfg.optimization.epochs
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        try:
            train_metrics = train_one_epoch(
                model, train_ds, sampler, optimizer, scheduler, scaler,
                cfg, epoch, device, limits.max_train_batches,
            )
            scheduler.step()  # documented order: optimizer steps inside, scheduler per epoch
            val_result = run_validation(
                model, val_ds, cfg, epoch, device, max_batches=limits.max_val_batches
            )
        except TrainingError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Epoch failed: retain previous complete checkpoint/history and
            # write a clear error report (no false completed epoch).
            error_report = {
                "failed_epoch": epoch,
                "error": repr(exc),
                "note": "previous checkpoint and history remain valid",
            }
            from preprocessing.storage import atomic_write_json

            atomic_write_json(fold_dir / "error_report.json", error_report)
            raise TrainingError(f"epoch {epoch} failed: {exc}") from exc

        val_metrics = val_result.metrics
        eligible = best_checkpoint_eligible(cfg, epoch)
        lambda_norm = effective_lambda_norm(cfg, epoch)
        is_best = False
        if eligible and (best_metric is None or val_metrics["val_loss"] < best_metric):
            best_metric = float(val_metrics["val_loss"])
            best_epoch = epoch
            is_best = True
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, scaler,
                epoch=epoch, global_step=global_step,
                best_metric=best_metric, best_epoch=best_epoch,
                fold=cfg.fold, run_id=run_id, cfg=cfg,
                source_checksums=checksums, device=device,
            )

        save_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step,
            best_metric=best_metric, best_epoch=best_epoch,
            fold=cfg.fold, run_id=run_id, cfg=cfg,
            source_checksums=checksums, device=device,
        )

        if is_best and val_result.embeddings is not None:
            np.savez(
                fold_dir / "validation" / "best_val_embeddings.npz",
                embeddings=val_result.embeddings,
                trial_uids=np.asarray(val_result.trial_uids, dtype=object),
                stimulus_indices=np.asarray(val_result.stimulus_indices),
                subject_ids=np.asarray(val_result.subject_ids, dtype=object),
            )
            from preprocessing.storage import atomic_write_json

            atomic_write_json(
                fold_dir / "validation" / "metrics.json",
                {"best_epoch": best_epoch, "best_val_loss": best_metric, **val_metrics.get("diagnostics", {})},
            )

        training_phase = (
            "warmup" if epoch < cfg.loss.norm_start_epoch
            else "ramp" if epoch < cfg.loss.norm_start_epoch + cfg.loss.norm_ramp_epochs
            else "joint"
        )
        row = row_from_metrics(
            run_id=run_id,
            fold=cfg.fold,
            epoch=epoch,
            training_phase=training_phase,
            eligible_for_best=eligible,
            is_best_epoch=is_best,
            best_epoch_so_far=best_epoch,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            lr_bounds=(cfg.optimization.learning_rate, cfg.optimization.learning_rate),
            weight_decay=cfg.optimization.weight_decay,
            lambda_norm=lambda_norm,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            extra={
                "seed": cfg.seed,
                "semantic_gamma_attention1": val_metrics["diagnostics"].get("semantic_gamma_attention1"),
                "semantic_gamma_attention2": val_metrics["diagnostics"].get("semantic_gamma_attention2"),
                "spatial_bridge_eta": val_metrics["diagnostics"].get("spatial_bridge_eta"),
                "train_mask_ratio_realized": train_metrics["train_mask_ratio_realized"],
                "val_mask_ratio_realized": val_metrics["val_mask_ratio_realized"],
                "num_train_trials": train_metrics["num_train_trials"],
                "num_val_trials": val_metrics["num_val_trials"],
                "n_train_batches": train_metrics["n_train_batches"],
                "n_val_batches": val_metrics["n_val_batches"],
                "n_train_stimulus_groups": train_metrics["n_train_stimulus_groups"],
                "n_val_stimulus_groups": val_metrics["n_val_stimulus_groups"],
                "n_skipped_norm_groups_train": train_metrics["n_skipped_norm_groups_train"],
                "n_skipped_norm_groups_val": val_metrics["n_skipped_norm_groups_val"],
                "nonfinite_batch_count": train_metrics["nonfinite_batch_count"],
                "epoch_time_seconds": time.time() - t0,
                "peak_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" and torch.cuda.is_available() else 0.0
                ),
                "grad_norm_mean": train_metrics["grad_norm_mean"],
                "grad_norm_max": train_metrics["grad_norm_max"],
                "grad_clip_fraction": train_metrics["grad_clip_fraction"],
            },
        )
        history_rows.append(row)
        write_history(fold_dir / "history.csv", history_rows)

        logger.info(
            "epoch %d/%d phase=%s loss=%.4f (recon %.4f, norm %.4f, spread %.4f) "
            "val=%.4f (recon %.4f, norm %.4f, spread %.4f) lr=%.2e%s",
            epoch, total_epochs - 1, training_phase,
            train_metrics["train_loss"], train_metrics["train_recon_loss"],
            train_metrics["train_norm_loss"], train_metrics["train_spread_loss"],
            val_metrics["val_loss"], val_metrics["val_recon_loss"],
            val_metrics["val_norm_loss"], val_metrics["val_spread_loss"],
            optimizer.param_groups[0]["lr"],
            "  [BEST]" if is_best else "",
        )

    return TrainOutcome(
        run_id=run_id,
        run_dir=run_dir,
        fold=cfg.fold,
        epochs_completed=total_epochs - start_epoch if total_epochs > start_epoch else 0,
        best_epoch=best_epoch,
        best_val_loss=best_metric,
        history_rows=len(history_rows),
        notes=[],
    )


def build_norm_bank_from_checkpoint(
    checkpoint_path: Path,
    *,
    cfg: Stage1Config | None = None,
    device: str = "cpu",
    output_dir: Path | None = None,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Rebuild the fold norm bank from a selected checkpoint (contract §9/CLI)."""
    from .normative_bank import NormBankConfig, build_normative_bank, save_normative_bank
    from .checkpoint import load_checkpoint

    contents = load_checkpoint(checkpoint_path, device)
    meta = contents.meta
    if cfg is None:
        cfg = Stage1Config.from_dict(meta["config_resolved"])
    if meta.get("fold") != cfg.fold:
        raise TrainingError(
            f"checkpoint fold {meta.get('fold')} does not match config fold {cfg.fold}"
        )
    seed_all(cfg.seed)
    train_ds, val_ds, _ = _build_datasets(cfg, device)
    model = Stage1Model(cfg)
    model.load_state_dict(contents.state_dict)
    model.to(device)

    val_ids = set(val_ds.trial_rows.subject_id.astype(str))
    n_stimuli = len(train_ds.dino_row_by_stimulus_index)
    result = build_normative_bank(
        model, train_ds,
        fold=cfg.fold,
        seed=cfg.seed,
        n_stimuli=n_stimuli,
        checkpoint_sha256=sha256_of_file(checkpoint_path),
        processed_checksums=source_checksums(cfg),
        dino_checksum=sha256_of_file(cfg.paths.dino_root / "patch_tokens.npy"),
        config=NormBankConfig(min_samples=min_samples, batch_size=64),
        forbidden_subject_ids=val_ids,
        device=device,
    )
    import pandas as pd

    fm = pd.read_csv(cfg.paths.dino_root / "feature_manifest.csv", dtype=str)
    stimulus_ids = [str(x) for x in fm.stimulus_id]
    out = output_dir or (cfg.paths.output_root / meta["run_id"] / f"fold_{cfg.fold}" / "normative_bank")
    save_meta = save_normative_bank(result, out, stimulus_ids)
    return save_meta
