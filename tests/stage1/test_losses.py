"""Loss tests: reconstruction scope, LOO cosine norm, stimulus separation."""

from __future__ import annotations

import pytest
import torch

from stage1.losses import (
    loo_cosine_normative_loss,
    masked_reconstruction_loss,
    stage1_loss,
)


def _token_mask(n: int, n_masked: int = 67) -> torch.Tensor:
    mask = torch.zeros(n, 192, dtype=torch.bool)
    mask[:, :n_masked] = True
    return mask


def test_reconstruction_loss_channel_aware_and_scoped():
    torch.manual_seed(0)
    recon = torch.rand(2, 3, 48, 64)
    target = torch.rand(2, 3, 48, 64)
    mask = _token_mask(2)
    out = masked_reconstruction_loss(recon, target, mask, scope="masked")
    assert out["total"].shape == ()
    assert out["fixation"] >= 0 and out["transition"] >= 0 and out["temporal"] >= 0
    full = masked_reconstruction_loss(recon, target, mask, scope="full")
    assert full["n_masked_pixels"] == 2 * 48 * 64  # pixel mask spans channels once


def test_normative_loss_never_mixes_stimuli():
    # Two stimuli with separated centroids; manual LOO computation must match.
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # stimulus 0
            [0.9, 0.1, 0.0],   # stimulus 0
            [0.8, 0.0, 0.1],   # stimulus 0
            [0.0, 1.0, 0.0],   # stimulus 1
            [0.1, 0.9, 0.0],   # stimulus 1
        ]
    )
    slot = torch.tensor([0, 0, 0, 1, 1])
    out = loo_cosine_normative_loss(z, slot, min_hc_per_stimulus=2)
    zz = torch.nn.functional.normalize(z, dim=-1, eps=1e-6)
    expected = 0.0
    for s in (0, 1):
        idx = [i for i in range(5) if int(slot[i]) == s]
        centroid = zz[idx].mean(dim=0)
        for k in idx:
            mu_minus = (centroid - zz[k] / len(idx)) * (len(idx) / (len(idx) - 1.0))
            expected = expected + (1.0 - torch.dot(zz[k], mu_minus))
    expected = expected / 5.0
    assert out["loss"] == pytest.approx(expected.item(), abs=1e-5)
    assert out["n_skipped_groups"] == 0
    assert out["n_groups"] == 2
    # Within-stimulus dispersion < between-stimulus dispersion for this setup.
    assert out["within_dispersion"] < out["between_dispersion"]


def test_normative_loss_requires_two_hc_examples():
    z = torch.randn(6, 8)
    slot = torch.tensor([0, 1, 2, 3, 4, 5])  # six singletons
    out = loo_cosine_normative_loss(z, slot, min_hc_per_stimulus=2)
    assert out["n_skipped_groups"] == 6
    assert out["loss"] == pytest.approx(0.0)
    # One group of 2 + one singleton: singleton skipped.
    z2 = torch.randn(3, 8)
    out2 = loo_cosine_normative_loss(z2, torch.tensor([0, 0, 1]), min_hc_per_stimulus=2)
    assert out2["n_skipped_groups"] == 1
    assert out2["n_groups"] == 2


def test_stage1_loss_total_and_lambda():
    torch.manual_seed(0)
    recon = torch.rand(4, 3, 48, 64)
    target = torch.rand(4, 3, 48, 64)
    mask = _token_mask(4)
    emb = torch.randn(4, 128)
    slots = torch.tensor([0, 0, 1, 1])
    l0 = stage1_loss(recon, target, mask, emb, slots, lambda_norm=0.0)
    l1 = stage1_loss(recon, target, mask, emb, slots, lambda_norm=0.5)
    assert l0.normative.item() >= 0
    assert l1.total.item() == pytest.approx(
        l0.reconstruction.item() + 0.5 * l1.normative.item()
    )
    assert l0.lambda_norm == 0.0 and l1.lambda_norm == 0.5


def test_reconstruction_full_scope_ablation():
    torch.manual_seed(0)
    recon = torch.rand(2, 3, 48, 64)
    target = torch.rand(2, 3, 48, 64)
    mask = _token_mask(2)
    masked = masked_reconstruction_loss(recon, target, mask, scope="masked")["total"]
    full = masked_reconstruction_loss(recon, target, mask, scope="full")["total"]
    assert torch.isfinite(masked) and torch.isfinite(full)
    assert not torch.allclose(masked, full)
