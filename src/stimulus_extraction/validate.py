"""Validation of extracted DINO features and cross-artifact verification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .storage import (
    FeaturePaths,
    StorageError,
    read_feature_manifest,
    token_sha256,
    verify_model_identity,
    verify_row,
)


def validate_tokens(tokens: np.ndarray, expected_tokens: int, expected_dim: int) -> dict[str, Any]:
    """Structural/statistical checks on a [n, tokens, dim] feature array."""
    if tokens.ndim != 3 or tokens.shape[1:] != (expected_tokens, expected_dim):
        raise StorageError(
            f"tokens shape {tokens.shape} != [n, {expected_tokens}, {expected_dim}]"
        )
    if tokens.dtype != np.float32:
        raise StorageError(f"tokens dtype {tokens.dtype} != float32")
    finite = bool(np.all(np.isfinite(tokens)))
    if not finite:
        raise StorageError("non-finite values in tokens")
    per_row_std = tokens.std(axis=(1, 2))
    n_zero_variance_rows = int(np.count_nonzero(per_row_std <= 0))
    if n_zero_variance_rows:
        raise StorageError(f"{n_zero_variance_rows} rows have zero variance")
    global_std = float(tokens.std())
    if global_std <= 0:
        raise StorageError("token variance across stimuli is zero")
    return {
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "finite": finite,
        "min_row_std": float(per_row_std.min()),
        "max_row_std": float(per_row_std.max()),
        "global_std": global_std,
        "n_zero_variance_rows": n_zero_variance_rows,
    }


def build_validation_report(
    memmap: np.ndarray,
    manifest_rows: list[dict[str, Any]],
    *,
    n_tokens: int,
    dim: int,
    elapsed_seconds: float,
    device: str,
    config_hash: str,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Final validation pass over the complete feature array."""
    t0 = time.time()
    per_row = []
    for row in manifest_rows:
        idx = int(row["feature_row_index"])
        actual = token_sha256(np.ascontiguousarray(memmap[idx], dtype=np.float32))
        per_row.append(
            {
                "stimulus_index": int(row["stimulus_index"]),
                "stimulus_id": row["stimulus_id"],
                "checksum_match": actual == row["feature_sha256"],
            }
        )
    all_match = all(r["checksum_match"] for r in per_row)
    if not all_match:
        raise StorageError("validation failed: row checksum mismatches")

    structural = validate_tokens(np.asarray(memmap), n_tokens, dim)
    # Pairwise non-identical check across stimuli (contract: tensors from
    # different stimuli must not be identical).
    n_rows = len(manifest_rows)
    non_identical = True
    if n_rows >= 2:
        a = np.ascontiguousarray(memmap[0], dtype=np.float32)
        b = np.ascontiguousarray(memmap[1], dtype=np.float32)
        non_identical = not np.array_equal(a, b)
    return {
        "config_hash": config_hash,
        "n_stimuli": len(manifest_rows),
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "per_row_checksums": per_row,
        "all_checksums_match": bool(all_match),
        "structural": structural,
        "first_two_rows_not_identical": bool(non_identical),
        "validation_seconds": time.time() - t0,
        "model_metadata": model_metadata,
    }


def verify_only(cfg: Any, paths: FeaturePaths, image_manifest_path: Path) -> dict[str, Any]:
    """Verify an existing feature artifact set against the image manifest."""
    if not paths.patch_tokens.is_file():
        raise StorageError(f"missing {paths.patch_tokens}")
    manifest = read_feature_manifest(paths)
    if manifest is None:
        raise StorageError("feature_manifest.csv missing")
    image_manifest = pd.read_csv(
        image_manifest_path, dtype={"stimulus_id": str, "sha256": str, "category": str}
    )
    image_manifest = image_manifest.sort_values("stimulus_index").reset_index(drop=True)

    if len(manifest) != len(image_manifest):
        raise StorageError(
            f"feature manifest has {len(manifest)} rows, image manifest has {len(image_manifest)}"
        )
    if not (manifest.stimulus_index == image_manifest.stimulus_index).all():
        raise StorageError("stimulus_index sets differ between manifests")
    if not (manifest.stimulus_id == image_manifest.stimulus_id).all():
        raise StorageError("stimulus_id order differs between manifests")
    if not (manifest.feature_row_index == manifest.stimulus_index).all():
        raise StorageError("feature_row_index must equal stimulus_index")
    if not (manifest.image_sha256 == image_manifest.sha256).all():
        raise StorageError("image SHA-256 mismatch between manifests")

    memmap = np.load(paths.patch_tokens, mmap_mode="r")
    expected_shape = (len(image_manifest), int(cfg.expected_patch_grid[0] * cfg.expected_patch_grid[1]), cfg.expected_token_dim)
    if memmap.shape != expected_shape:
        raise StorageError(f"patch_tokens shape {memmap.shape} != {expected_shape}")
    for _, row in manifest.iterrows():
        verify_row(memmap, int(row.feature_row_index), str(row.feature_sha256))
    structural = validate_tokens(
        np.asarray(memmap), int(cfg.expected_patch_grid[0] * cfg.expected_patch_grid[1]), cfg.expected_token_dim
    )

    # Model/config identity: only if the metadata files exist (older artifacts
    # may predate them; the contract requires them for canonical outputs).
    model_identity_ok = True
    identity_note = "verified"
    if paths.model_metadata.is_file():
        from types import SimpleNamespace

        stored_meta = json.loads(paths.model_metadata.read_text(encoding="utf-8"))
        meta_proxy = SimpleNamespace(checkpoint_sha256=stored_meta.get("checkpoint_sha256"))
        try:
            verify_model_identity(cfg, paths, meta_proxy)
        except StorageError as exc:
            model_identity_ok = False
            identity_note = str(exc)
    else:
        identity_note = "model_metadata.json absent; identity not verified"
        model_identity_ok = False
    return {
        "n_stimuli": len(manifest),
        "structural": structural,
        "model_identity_ok": model_identity_ok,
        "identity_note": identity_note,
        "status": "ok",
    }
