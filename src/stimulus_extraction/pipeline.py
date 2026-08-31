"""Extraction orchestration: images -> frozen DINO tokens -> validated storage."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import DINOExtractionConfig
from .dino_extractor import (
    ExtractionError,
    extract_patch_tokens,
    load_frozen_dino,
    preprocess_image,
)
from .storage import (
    FeaturePaths,
    StorageError,
    open_memmap,
    token_sha256,
    verify_model_identity,
    verify_row,
    write_extraction_config,
    write_feature_manifest,
    write_model_metadata,
    write_validation_report,
)
from .validate import build_validation_report


@dataclass
class ExtractionOptions:
    device: str | None = None
    batch_size: int | None = None
    resume: bool = False
    force: bool = False
    stimulus_limit: int | None = None
    output_root_override: Path | None = None
    model: Any = None  # injectable for tests
    model_metadata: Any = None


def _load_image_rows(cfg: DINOExtractionConfig, limit: int | None) -> pd.DataFrame:
    manifest = pd.read_csv(
        cfg.image_manifest,
        dtype={
            "stimulus_index": int,
            "stimulus_id": str,
            "source_image_name": str,
            "category": str,
            "relative_image_path": str,
            "width": int,
            "height": int,
            "sha256": str,
        },
    )
    manifest = manifest.sort_values("stimulus_index").reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise StorageError("stimulus_limit must be positive")
        manifest = manifest.head(limit).reset_index(drop=True)
    return manifest


def run_extraction(cfg: DINOExtractionConfig, options: ExtractionOptions) -> dict[str, Any]:
    device = options.device or cfg.device
    batch_size = options.batch_size or cfg.batch_size
    if options.stimulus_limit is not None and options.output_root_override is None:
        raise StorageError(
            "--stimulus-limit is for smoke tests and requires an output-root override"
        )
    out_dir = (
        Path(options.output_root_override) / cfg.output_subdir
        if options.output_root_override is not None
        else cfg.feature_output_dir
    )
    paths = FeaturePaths.for_dir(out_dir)
    rows = _load_image_rows(cfg, options.stimulus_limit)
    n_tokens = cfg.expected_patch_grid[0] * cfg.expected_patch_grid[1]
    dim = cfg.expected_token_dim
    t0 = time.time()

    # --- existing output handling ------------------------------------------
    existing = paths.patch_tokens.is_file() and paths.feature_manifest.is_file()
    if existing and not (options.resume or options.force):
        raise StorageError(
            f"feature directory already exists: {out_dir} (use --resume or --force)"
        )

    previous_rows: pd.DataFrame | None = None
    previous_memmap = None
    if existing and options.resume:
        from .storage import read_feature_manifest

        previous_rows = read_feature_manifest(paths)
        previous_memmap = np.load(paths.patch_tokens, mmap_mode="r")
        for _, row in previous_rows.iterrows():
            verify_row(previous_memmap, int(row.feature_row_index), str(row.feature_sha256))
        if len(previous_rows) == len(rows) and list(previous_rows.stimulus_id) == list(rows.stimulus_id):
            return {
                "status": "already_complete",
                "n_stimuli": len(rows),
                "output_dir": str(out_dir),
                "elapsed_seconds": time.time() - t0,
                "notes": ["all rows already extracted and verified; no model was loaded"],
            }

    # --- model loading (identity-verified on resume) ------------------------
    if options.model is not None:
        model, model_metadata = options.model, options.model_metadata
    else:
        model, model_metadata = load_frozen_dino(cfg)
    if existing and options.resume:
        verify_model_identity(cfg, paths, model_metadata)
    if existing and options.force:
        verify_model_identity_if_present(cfg, paths)

    # --- staging ------------------------------------------------------------
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / f".{out_dir.name}.staging_{os.getpid()}_{int(time.time() * 1000)}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        staged_paths = FeaturePaths.for_dir(staging)
        memmap = open_memmap(staged_paths.patch_tokens, len(rows), n_tokens, dim)
        done: set[int] = set()

        if existing and options.resume and previous_rows is not None:
            # Copy verified existing rows into the staged array.
            for _, row in previous_rows.iterrows():
                idx = int(row.feature_row_index)
                if idx < len(rows) and rows.iloc[idx].stimulus_id == row.stimulus_id:
                    memmap[idx] = np.asarray(previous_memmap[idx], dtype=np.float32)
                    done.add(idx)

        manifest_rows: list[dict[str, Any]] = []
        if existing and options.resume:
            for _, row in previous_rows.iterrows():
                idx = int(row.feature_row_index)
                if idx in done:
                    manifest_rows.append(row.to_dict())

        # --- extraction ----------------------------------------------------
        model.to(device)
        pending = [i for i in range(len(rows)) if i not in done]
        for start in range(0, len(pending), batch_size):
            idxs = pending[start : start + batch_size]
            tensors = []
            for i in idxs:
                row = rows.iloc[i]
                image_path = cfg.image_source_root / str(row.relative_image_path)
                tensors.append(
                    preprocess_image(
                        image_path,
                        cfg,
                        expected_width=int(row.width),
                        expected_height=int(row.height),
                        expected_sha256=str(row.sha256),
                    )
                )
            batch = torch.stack(tensors)
            tokens = extract_patch_tokens(model, batch, cfg, device)  # [B, n_tokens, dim]
            tokens_np = tokens.numpy()
            for k, i in enumerate(idxs):
                memmap[i] = tokens_np[k]
                row = rows.iloc[i]
                manifest_rows.append(
                    {
                        "stimulus_index": int(row.stimulus_index),
                        "stimulus_id": str(row.stimulus_id),
                        "feature_row_index": int(row.stimulus_index),
                        "image_sha256": str(row.sha256),
                        "feature_shape": f"[{n_tokens}, {dim}]",
                        "feature_dtype": "float32",
                        "feature_sha256": token_sha256(tokens_np[k]),
                    }
                )
        memmap.flush()

        manifest_rows.sort(key=lambda r: r["feature_row_index"])
        if len(manifest_rows) != len(rows):
            raise StorageError(f"manifest rows {len(manifest_rows)} != expected {len(rows)}")

        elapsed = time.time() - t0
        write_feature_manifest(manifest_rows, staged_paths)
        write_extraction_config(
            cfg,
            staged_paths,
            {
                "mode": "rgb_convert -> float32 [0,1] -> torchvision tensor resize "
                "(bicubic, antialias=True, align_corners=False) -> ImageNet normalization",
                "resize_size": [cfg.input_height, cfg.input_width],
                "interpolation": cfg.interpolation,
                "antialias": cfg.antialias,
                "normalization_mean": list(cfg.normalization_mean),
                "normalization_std": list(cfg.normalization_std),
                "center_crop": cfg.center_crop,
                "random_augmentation": cfg.random_augmentation,
            },
        )
        write_model_metadata(model_metadata, staged_paths)
        report = build_validation_report(
            memmap,
            manifest_rows,
            n_tokens=n_tokens,
            dim=dim,
            elapsed_seconds=elapsed,
            device=device,
            config_hash=cfg.config_hash(),
            model_metadata=model_metadata.to_dict() if hasattr(model_metadata, "to_dict") else dict(model_metadata),
        )
        write_validation_report(report, staged_paths)

        # --- atomic publish -------------------------------------------------
        old = None
        if out_dir.exists():
            old = out_dir.parent / f".{out_dir.name}.old_{os.getpid()}_{int(time.time() * 1000)}"
            os.replace(out_dir, old)
        try:
            os.replace(staging, out_dir)
        except BaseException:
            if old is not None and old.exists():
                os.replace(old, out_dir)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "complete",
        "n_stimuli": len(rows),
        "output_dir": str(out_dir),
        "elapsed_seconds": elapsed,
        "notes": [],
    }


def verify_model_identity_if_present(cfg: DINOExtractionConfig, paths: FeaturePaths) -> None:
    """Best-effort identity check when --force replaces an existing artifact."""
    if not paths.model_metadata.is_file():
        return
    stored = json.loads(paths.model_metadata.read_text(encoding="utf-8"))
    if stored.get("hub_source") != cfg.hub_source or stored.get("model_name") != cfg.model_name:
        raise StorageError(
            f"refusing --force: existing features were extracted with "
            f"{stored.get('hub_source')}/{stored.get('model_name')} but the config "
            f"requests {cfg.hub_source}/{cfg.model_name}"
        )
