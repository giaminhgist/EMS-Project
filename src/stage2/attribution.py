"""Validation attribution export (guide 07 §14, contracts §25).

``stimulus_attributions.npz`` stores per-subject panels aligned with explicit
``subject_ids`` and ``stimulus_ids`` arrays. Attention is an attribution, not
an explanation; the export keeps importance, deviation and semantic
information as separate families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .validation import ValidationResult


def write_stimulus_attributions(path: Path | str, result: ValidationResult) -> None:
    """Atomically write the attribution archive with explicit ID alignment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "subject_ids": np.array(result.subject_ids, dtype=object),
        "stimulus_ids": np.array([f"s{i:03d}.jpg" for i in range(100)], dtype=object),
        "stimulus_indices": np.arange(100, dtype=np.int64),
        "stimulus_attention": result.stimulus_attention.numpy(),
        "stimulus_evidence": result.stimulus_evidence.numpy(),
        "stimulus_contribution": result.stimulus_contribution.numpy(),
        "semantic_compatibility": result.semantic_compatibility.numpy(),
        "normative_deviation": result.normative_deviation.numpy(),
        "trial_mask": result.trial_mask.numpy(),
    }
    if result.semantic_patch_maps is not None:
        arrays["semantic_patch_maps"] = result.semantic_patch_maps.numpy()
    # savez_compressed appends ".npz" unless the name already ends with it.
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def read_stimulus_attributions(path: Path | str) -> dict[str, np.ndarray]:
    return dict(np.load(Path(path), allow_pickle=True))
