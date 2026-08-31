"""Deterministic patch masking (contract §4).

Training masks vary reproducibly by epoch; validation masks are fixed per
trial via SHA-256(seed, fold, trial_uid) so validation losses stay comparable
across epochs and resumes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

N_TOKENS = 192
GRID_H, GRID_W = 12, 16


def derive_seed_int(*parts: object, n_bytes: int = 8) -> int:
    """Stable SHA-256-derived integer (never Python's hash())."""
    material = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[: n_bytes * 2], 16)


def make_rng(*parts: object) -> np.random.Generator:
    """Deterministic NumPy generator from SHA-256-derived seed parts."""
    return np.random.default_rng(derive_seed_int(*parts))


def sample_token_mask(
    n_trials: int, ratio: float, rng: np.random.Generator, n_tokens: int = N_TOKENS
) -> torch.Tensor:
    """Sample a boolean token mask [n_trials, n_tokens] with ~ratio True."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"mask ratio must be in [0, 1), got {ratio}")
    n_mask = max(0, min(int(round(ratio * n_tokens)), n_tokens))
    mask = np.zeros((n_trials, n_tokens), dtype=bool)
    if n_mask > 0:
        idx = np.arange(n_tokens, dtype=np.int64)
        for i in range(n_trials):
            chosen = rng.choice(idx, size=n_mask, replace=False)
            mask[i, chosen] = True
    return torch.from_numpy(mask)


def training_token_masks(
    n_trials: int, ratio: float, seed: int, fold: int, epoch: int
) -> torch.Tensor:
    """Reproducible per-epoch training masks."""
    rng = make_rng("train_mask", seed, fold, epoch)
    return sample_token_mask(n_trials, ratio, rng)


def validation_token_masks(
    trial_uids: list[str], ratio: float, seed: int, fold: int
) -> torch.Tensor:
    """Fixed per-trial validation masks (stable across calls and restarts)."""
    masks = []
    for uid in trial_uids:
        rng = make_rng("val_mask", seed, fold, uid)
        masks.append(sample_token_mask(1, ratio, rng)[0])
    return torch.stack(masks) if masks else torch.zeros(0, N_TOKENS, dtype=torch.bool)


def token_mask_to_pixel_mask(
    token_mask: torch.Tensor, patch_h: int = 4, patch_w: int = 4
) -> torch.Tensor:
    """Upsample a [N, 192] token mask to a [N, 1, 48, 64] pixel mask by
    repeating each 12x16 token over its 4x4 output patch."""
    n = token_mask.shape[0]
    grid = token_mask.reshape(n, GRID_H, GRID_W).float()  # [N, 12, 16]
    grid = grid.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
    return grid.unsqueeze(1).bool()


def realized_mask_ratio(mask: torch.Tensor) -> float:
    return float(mask.float().mean()) if mask.numel() else 0.0
