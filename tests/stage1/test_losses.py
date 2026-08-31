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


def test_stage1_loss_total_with_spread():
    torch.manual_seed(0)
    recon = torch.rand(4, 3, 48, 64)
    target = torch.rand(4, 3, 48, 64)
    mask = _token_mask(4)
    emb = torch.randn(4, 128)
    slots = torch.tensor([0, 0, 1, 1])
    l = stage1_loss(recon, target, mask, emb, slots, lambda_norm=0.1, lambda_spread=0.5)
    assert l.total.item() == pytest.approx(
        l.reconstruction.item() + 0.1 * l.normative.item() + 0.5 * l.spread_loss.item()
    )
    assert l.lambda_spread == 0.5


def test_lambda_spread_zero_reproduces_old_behavior():
    torch.manual_seed(0)
    recon = torch.rand(4, 3, 48, 64)
    target = torch.rand(4, 3, 48, 64)
    mask = _token_mask(4)
    emb = torch.randn(4, 128)
    slots = torch.tensor([0, 0, 1, 1])
    l = stage1_loss(recon, target, mask, emb, slots, lambda_norm=0.5, lambda_spread=0.0)
    assert l.total.item() == pytest.approx(l.reconstruction.item() + 0.5 * l.normative.item())


def test_spread_zero_when_dispersion_above_floor():
    # Two orthogonal centroids: between-dispersion far above the floor.
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    slot = torch.tensor([0, 0, 1, 1])
    out = loo_cosine_normative_loss(z, slot, min_hc_per_stimulus=2, spread_floor=0.1)
    assert out["spread_loss"].item() == pytest.approx(0.0)
    # Identical centroids sit at the floor: spread == floor^2.
    z_col = torch.ones(4, 3)
    out_col = loo_cosine_normative_loss(z_col, slot, min_hc_per_stimulus=2, spread_floor=0.1)
    assert out_col["spread_loss"].item() == pytest.approx(0.1**2, abs=1e-6)


def test_spread_hinge_magnitude():
    # Centroid cosine sim 0.95 -> violation = 0.1 - 0.05 = 0.05.
    c0 = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([0.95, (1 - 0.95**2) ** 0.5, 0.0])
    z = torch.stack([c0, c0, v, v])
    slot = torch.tensor([0, 0, 1, 1])
    out = loo_cosine_normative_loss(z, slot, min_hc_per_stimulus=2, spread_floor=0.1)
    assert out["between_dispersion"] == pytest.approx(1 - 0.95, abs=1e-6)
    assert out["spread_loss"].item() == pytest.approx((0.1 - 0.05) ** 2, abs=1e-6)


def test_spread_singleton_guard():
    z = torch.randn(4, 8)
    slot = torch.tensor([0, 1, 2, 3])
    out = loo_cosine_normative_loss(z, slot, min_hc_per_stimulus=2, spread_floor=0.1)
    assert out["spread_loss"].item() == pytest.approx(0.0)


def test_spread_gradient_separates_centroids():
    torch.manual_seed(0)
    # Two near-collapsed groups (tiny asymmetry so the cosine gradient is
    # nonzero); optimizing the spread hinge alone must separate the centroids.
    emb = torch.nn.Parameter(
        torch.full((4, 16), 0.1) + 1e-3 * torch.randn(4, 16)
    )
    slot = torch.tensor([0, 0, 1, 1])
    opt = torch.optim.Adam([emb], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        out = loo_cosine_normative_loss(emb, slot, min_hc_per_stimulus=2, spread_floor=0.1)
        out["spread_loss"].backward()
        assert emb.grad is not None and torch.all(torch.isfinite(emb.grad))
        opt.step()
    final = loo_cosine_normative_loss(
        emb.detach(), slot, min_hc_per_stimulus=2, spread_floor=0.1
    )
    assert final["between_dispersion"] >= 0.1 - 1e-3
    assert final["spread_loss"].item() <= 1e-4


def test_reconstruction_full_scope_ablation():
    torch.manual_seed(0)
    recon = torch.rand(2, 3, 48, 64)
    target = torch.rand(2, 3, 48, 64)
    mask = _token_mask(2)
    masked = masked_reconstruction_loss(recon, target, mask, scope="masked")["total"]
    full = masked_reconstruction_loss(recon, target, mask, scope="full")["total"]
    assert torch.isfinite(masked) and torch.isfinite(full)
    assert not torch.allclose(masked, full)
