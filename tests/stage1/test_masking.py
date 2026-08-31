"""Deterministic patch masking tests."""

from __future__ import annotations

import pytest
import torch

from stage1.losses import masked_reconstruction_loss
from stage1.masking import (
    N_TOKENS,
    make_rng,
    sample_token_mask,
    token_mask_to_pixel_mask,
    training_token_masks,
    validation_token_masks,
)


def test_mask_count_matches_ratio_within_rounding():
    rng = make_rng("t", 2026)
    mask = sample_token_mask(8, 0.35, rng)
    assert mask.shape == (8, N_TOKENS)
    expected = round(0.35 * N_TOKENS)  # 67
    assert int(mask.sum(dim=1).unique()) == expected


def test_training_masks_reproducible_and_epoch_dependent():
    a1 = training_token_masks(4, 0.35, seed=2026, fold=0, epoch=3)
    a2 = training_token_masks(4, 0.35, seed=2026, fold=0, epoch=3)
    assert torch.equal(a1, a2)
    b = training_token_masks(4, 0.35, seed=2026, fold=0, epoch=4)
    assert not torch.equal(a1, b)  # epochs differ
    c = training_token_masks(4, 0.35, seed=2026, fold=1, epoch=3)
    assert not torch.equal(a1, c)  # folds differ


def test_validation_masks_stable_across_calls_and_processes():
    uids = [f"uid-{i}" for i in range(6)]
    m1 = validation_token_masks(uids, 0.35, seed=2026, fold=0)
    m2 = validation_token_masks(uids, 0.35, seed=2026, fold=0)
    assert torch.equal(m1, m2)
    assert m1.shape == (6, N_TOKENS)
    # Different trials get different masks; same uid always the same.
    assert not torch.equal(m1[0], m1[1])


def test_token_to_pixel_mask_structure():
    token_mask = torch.zeros(2, N_TOKENS, dtype=torch.bool)
    token_mask[0, 0] = True  # token 0 = top-left 4x4 block
    token_mask[1, 192 - 1] = True  # bottom-right block
    pixel = token_mask_to_pixel_mask(token_mask)
    assert pixel.shape == (2, 1, 48, 64)
    assert bool(pixel[0, 0, 0, 0]) and not bool(pixel[0, 0, 4, 0])
    assert bool(pixel[1, 0, 47, 63]) and not bool(pixel[1, 0, 43, 63])


def test_masked_loss_requires_masked_pixels():
    recon = torch.zeros(1, 3, 48, 64)
    target = torch.zeros(1, 3, 48, 64)
    empty_mask = torch.zeros(1, N_TOKENS, dtype=torch.bool)
    with pytest.raises(ValueError, match="zero masked"):
        masked_reconstruction_loss(recon, target, empty_mask, scope="masked")


def test_masked_loss_only_sees_masked_pixels():
    torch.manual_seed(0)
    token_mask = torch.zeros(1, N_TOKENS, dtype=torch.bool)
    token_mask[0, :67] = True
    recon = torch.rand(1, 3, 48, 64)
    target = torch.rand(1, 3, 48, 64)
    base = masked_reconstruction_loss(recon, target, token_mask, scope="masked")["total"]
    # Perturb an unmasked pixel (bottom-right block): loss unchanged.
    target2 = target.clone()
    target2[0, 0, 47, 63] += 5.0
    same = masked_reconstruction_loss(recon, target2, token_mask, scope="masked")["total"]
    assert torch.allclose(base, same)
    # Perturb a masked pixel: loss changes.
    target3 = target.clone()
    target3[0, 0, 0, 0] += 5.0
    changed = masked_reconstruction_loss(recon, target3, token_mask, scope="masked")["total"]
    assert not torch.allclose(base, changed)
