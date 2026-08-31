"""Fold-safe HC normative bank builder (contract §9).

Unmasked inference over the current fold's outer-training HC trials; per-
stimulus mean, diagonal standard deviation, and sample count. Validation HC
subjects can never contribute (the input dataset is the train partition and
forbidden subject IDs are asserted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import Stage1Dataset


class NormBankError(ValueError):
    pass


@dataclass(frozen=True)
class NormBankConfig:
    estimator: str = "mean"  # mean | median
    epsilon: float = 1e-6
    min_samples: int = 2
    include_token_level: bool = False
    on_insufficient: str = "error"  # error | warn
    batch_size: int = 64

    def validate(self) -> None:
        if self.estimator not in ("mean", "median"):
            raise NormBankError(f"unsupported estimator {self.estimator!r}")
        if self.epsilon < 0 or self.min_samples < 1:
            raise NormBankError("epsilon must be >= 0; min_samples >= 1")
        if self.on_insufficient not in ("error", "warn"):
            raise NormBankError("on_insufficient must be 'error' or 'warn'")
        if self.batch_size <= 0:
            raise NormBankError("batch_size must be positive")


@dataclass
class NormBankResult:
    mu_trial: np.ndarray  # float32 [n_stimuli, 128]
    sigma_trial: np.ndarray  # float32 [n_stimuli, 128]
    count_trial: np.ndarray  # int32 [n_stimuli]
    mu_token: np.ndarray | None  # float32 [n_stimuli, 192, 128]
    sigma_token: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_normative_bank(
    model: Any,
    dataset: Stage1Dataset,
    *,
    fold: int,
    seed: int,
    n_stimuli: int,
    checkpoint_sha256: str,
    processed_checksums: dict[str, str],
    dino_checksum: str,
    config: NormBankConfig,
    forbidden_subject_ids: set[str] | None = None,
    device: str = "cpu",
) -> NormBankResult:
    config.validate()
    if dataset.split != "train":
        raise NormBankError("normative bank must be built from the train partition only")

    # NOTE: CPU BLAS GEMM tiling is batch-shape dependent, so embeddings carry
    # ~1e-6 batch-dependent float noise. Statistics are reproducible for a
    # fixed (batch_size, data) pair; keep batch_size fixed across runs.
    model.eval()
    dim = model.d_model
    sums = np.zeros((n_stimuli, dim), dtype=np.float64)
    sumsq = np.zeros((n_stimuli, dim), dtype=np.float64)
    counts = np.zeros(n_stimuli, dtype=np.int64)
    medians_acc: dict[int, list[np.ndarray]] = {}
    subject_ids: set[str] = set()
    token_sums: dict[int, np.ndarray] = {}
    token_sumsq: dict[int, np.ndarray] = {}
    n_token_cells = 192

    indices = list(range(len(dataset)))
    with torch.inference_mode():
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            batch = dataset.collate_from_indices(batch_indices)
            batch.to_device(device)
            out = model(
                batch,
                token_mask=None,  # force unmasked inference
                return_fused=config.include_token_level,
            )
            emb = out.trial_embedding.cpu().numpy().astype(np.float64)  # [b, 128]
            fused = out.fused_tokens.cpu().numpy() if config.include_token_level else None
            for k, row_pos in enumerate(batch_indices):
                row = dataset.trial_rows.iloc[row_pos]
                si = int(row.stimulus_index)
                z = emb[k]
                sums[si] += z
                sumsq[si] += z * z
                counts[si] += 1
                if config.estimator == "median":
                    medians_acc.setdefault(si, []).append(z)
                subject_ids.add(str(row.subject_id))
                if config.include_token_level and fused is not None:
                    t = fused[k]  # [192, 128]
                    if si not in token_sums:
                        token_sums[si] = np.zeros((n_token_cells, dim), dtype=np.float64)
                        token_sumsq[si] = np.zeros((n_token_cells, dim), dtype=np.float64)
                    token_sums[si] += t
                    token_sumsq[si] += t * t

    if forbidden_subject_ids and (subject_ids & forbidden_subject_ids):
        raise NormBankError(
            f"forbidden (validation) subjects contributed to the bank: "
            f"{sorted(subject_ids & forbidden_subject_ids)}"
        )

    n_valid = int(np.count_nonzero(counts))
    missing = [int(i) for i in range(n_stimuli) if counts[i] == 0]
    insufficient = [int(i) for i in range(n_stimuli) if 0 < counts[i] < config.min_samples]
    if missing and config.on_insufficient == "error":
        raise NormBankError(f"stimuli with zero training-HC samples: {missing}")
    if insufficient and config.on_insufficient == "error":
        raise NormBankError(f"stimuli below min_samples={config.min_samples}: {insufficient}")

    mu = np.zeros((n_stimuli, dim), dtype=np.float32)
    sigma = np.zeros((n_stimuli, dim), dtype=np.float32)
    for si in range(n_stimuli):
        if counts[si] == 0:
            continue  # stays zero; recorded as missing in metadata
        if config.estimator == "mean":
            mu64 = sums[si] / counts[si]
        else:
            mu64 = np.median(np.stack(medians_acc[si]), axis=0)
        # Variance in float64 with the un-rounded mean (a float32-rounded mean
        # injects avoidable ~1e-6 cancellation error).
        var = sumsq[si] / counts[si] - mu64 * mu64
        mu[si] = mu64  # cast only at storage
        sigma[si] = np.sqrt(np.maximum(var, config.epsilon**2)).astype(np.float32)

    mu_token = sigma_token = None
    if config.include_token_level:
        mu_token = np.zeros((n_stimuli, n_token_cells, dim), dtype=np.float32)
        sigma_token = np.zeros((n_stimuli, n_token_cells, dim), dtype=np.float32)
        for si in token_sums:
            c = counts[si]
            m = token_sums[si] / c
            var = token_sumsq[si] / c - m * m
            mu_token[si] = m.astype(np.float32)
            sigma_token[si] = np.sqrt(np.maximum(var, config.epsilon**2)).astype(np.float32)

    metadata = {
        "fold": fold,
        "seed": seed,
        "estimator": config.estimator,
        "epsilon": config.epsilon,
        "min_samples": config.min_samples,
        "n_stimuli": n_stimuli,
        "n_valid_stimuli": n_valid,
        "missing_stimuli": missing,
        "insufficient_stimuli": insufficient,
        "n_trials": int(counts.sum()),
        "subject_ids": sorted(subject_ids),
        "n_subjects": len(subject_ids),
        "checkpoint_sha256": checkpoint_sha256,
        "processed_checksums": processed_checksums,
        "dino_checksum": dino_checksum,
        "include_token_level": config.include_token_level,
    }
    return NormBankResult(
        mu_trial=mu.astype(np.float32),
        sigma_trial=sigma.astype(np.float32),
        count_trial=counts.astype(np.int32),
        mu_token=mu_token,
        sigma_token=sigma_token,
        metadata=metadata,
    )


def save_normative_bank(result: NormBankResult, output_dir: Path, stimulus_ids: list[str]) -> dict[str, Any]:
    """Atomically publish the bank arrays + metadata + stimulus manifest."""
    from preprocessing.storage import atomic_write_json, atomic_write_npy

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        {"stimulus_index": i, "stimulus_id": stimulus_ids[i]}
        for i in range(len(stimulus_ids))
    ]
    manifest_df = pd.DataFrame(manifest_rows)
    import io

    from preprocessing.storage import atomic_write_bytes

    buf = io.StringIO()
    manifest_df.to_csv(buf, index=False)
    atomic_write_bytes(out / "feature_manifest.csv", buf.getvalue().encode("utf-8"))

    shas = {
        "mu_trial": atomic_write_npy(out / "mu_trial.npy", result.mu_trial),
        "sigma_trial": atomic_write_npy(out / "sigma_trial.npy", result.sigma_trial),
        "count_trial": atomic_write_npy(out / "count_trial.npy", result.count_trial),
    }
    if result.mu_token is not None and result.sigma_token is not None:
        shas["mu_token"] = atomic_write_npy(out / "mu_token.npy", result.mu_token)
        shas["sigma_token"] = atomic_write_npy(out / "sigma_token.npy", result.sigma_token)
    metadata = {**result.metadata, "array_sha256": shas}
    atomic_write_json(out / "metadata.json", metadata)
    return metadata
