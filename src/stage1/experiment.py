"""Run identity, output layout, and metadata (contract §7)."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from preprocessing.storage import atomic_write_json, atomic_write_text, sha256_of_file


class ExperimentError(ValueError):
    pass


_run_id_counter = 0


def make_run_id(experiment_name: str, ablation: str, run_name: str | None = None) -> str:
    """Timestamped run ID that includes the ablation name; never reused."""
    global _run_id_counter
    _run_id_counter += 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if run_name:
        return f"{experiment_name}_{ablation}_{run_name}_{stamp}_{_run_id_counter}"
    return f"{experiment_name}_{ablation}_{stamp}_{_run_id_counter}"


def source_checksums(cfg: Any) -> dict[str, str]:
    from .dataset import verify_input_checksums  # noqa: F401 (validation happens at dataset init)

    return {
        "processed_subject_manifest_sha256": sha256_of_file(cfg.paths.processed_root / "subject_manifest.csv"),
        "processed_trial_manifest_sha256": sha256_of_file(cfg.paths.processed_root / "trial_manifest.parquet"),
        "processed_dataset_metadata_sha256": sha256_of_file(cfg.paths.processed_root / "dataset_metadata.json"),
        "dino_patch_tokens_sha256": sha256_of_file(cfg.paths.dino_root / "patch_tokens.npy"),
        "cv_metadata_sha256": sha256_of_file(cfg.paths.cv_fold_dir.parent / "cv_metadata.json"),
    }


def initialize_run_dir(output_root: Path, run_id: str, cfg: Any, cli_overrides: dict[str, Any]) -> Path:
    """Create the isolated run directory; refuse to reuse a completed run."""
    run_dir = output_root / run_id
    if run_dir.exists():
        raise ExperimentError(
            f"run directory already exists: {run_dir} (completed runs are never overwritten)"
        )
    run_dir.mkdir(parents=True)
    atomic_write_text(run_dir / "config_resolved.yaml", cfg.to_yaml())
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    atomic_write_json(run_dir / "environment.json", environment)
    atomic_write_json(run_dir / "source_checksums.json", source_checksums(cfg))
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": cfg.config_hash(),
            "ablation": cfg.ablation,
            "experiment_name": cfg.experiment_name,
            "fold": cfg.fold,
            "seed": cfg.seed,
            "cli_overrides": cli_overrides,
        },
    )
    return run_dir
