"""Trial relation and bank adapter tests (guide 05 §6-§7, §16)."""

from __future__ import annotations

import torch
import pytest

from stage2.relation import (
    BankMeanAdapter,
    BankSigmaAdapter,
    ReliabilityHead,
    TrialRelation,
    TrialRelationBlock,
    row_cosine,
)


def test_projection_shapes_and_feature_dim():
    block = TrialRelationBlock().eval()  # deterministic dropout for the recompute
    q0 = torch.randn(5, 128)
    mu = torch.randn(5, 128)
    sigma = torch.rand(5, 128) + 0.5
    count = torch.tensor([4, 5, 6, 7, 8])
    out = block(q0, mu, sigma, count)
    for key, shape in {
        "q": (5, 128),
        "n_mu": (5, 128),
        "uncertainty_context": (5, 128),
        "rho": (5, 1),
        "cosine": (5, 1),
        "z_trial": (5, 128),
        "comparator": (5,),
    }.items():
        assert out[key].shape == shape, key
    # Relation MLP consumes the explicit 770-dim feature vector.
    features = block.relation.build_features(
        out["q"], out["n_mu"], out["uncertainty_context"], out["cosine"], out["rho"]
    )
    assert features.shape == (5, 770)
    assert torch.allclose(
        out["z_trial"],
        block.relation.norm_out(block.relation.mlp(features) + out["q"]),
    )


def test_wrong_bank_produces_different_relation():
    block = TrialRelationBlock()
    q0 = torch.randn(4, 128)
    sigma = torch.rand(4, 128) + 0.5
    count = torch.tensor([8, 8, 8, 8])
    pos = block(q0, torch.randn(4, 128), sigma, count)
    wrong = block(q0, torch.randn(4, 128) + 3.0, sigma, count)
    assert not torch.allclose(pos["z_trial"], wrong["z_trial"])
    assert not torch.allclose(pos["cosine"], wrong["cosine"])
    assert not torch.allclose(pos["comparator"], wrong["comparator"])


def test_forward_with_query_reuses_query_projection():
    """Wrong-bank runs must not reproject q0; the q supplied is used as-is."""
    block = TrialRelationBlock().eval()
    q0 = torch.randn(4, 128)
    mu = torch.randn(4, 128)
    sigma = torch.rand(4, 128) + 0.5
    count = torch.tensor([8, 8, 8, 8])
    base = block.project(q0, mu, sigma, count)
    reused = block.project_with_query(base["q"], mu, sigma, count)
    assert torch.equal(reused["q"], base["q"])
    assert torch.allclose(reused["cosine"], row_cosine(base["q"], reused["n_mu"]))
    full = block.forward_with_query(base["q"], mu, sigma, count)
    assert torch.allclose(
        full["z_trial"], block.forward(q0, mu, sigma, count)["z_trial"]
    )


def test_reliability_in_unit_interval():
    head = ReliabilityHead()
    rho = head(torch.rand(6, 128) + 0.1, torch.tensor([2, 3, 4, 5, 6, 7]))
    assert rho.shape == (6, 1)
    assert ((rho >= 0.0) & (rho <= 1.0)).all()


def test_gradients_flow_to_all_relation_modules():
    block = TrialRelationBlock()
    q0 = torch.randn(3, 128)
    mu = torch.randn(3, 128)
    sigma = torch.rand(3, 128) + 0.5
    count = torch.tensor([8, 8, 8])
    out = block(q0, mu, sigma, count)
    (out["z_trial"].sum() + out["cosine"].sum() + out["comparator"].sum()).backward()
    for name in (
        "query_projection", "bank_mean_adapter", "bank_sigma_adapter",
        "reliability", "relation", "comparator",
    ):
        module = getattr(block, name)
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), f"{name} missing gradients"
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads), name


def test_sigma_adapter_uses_count():
    """The sigma adapter consumes log_count, so count changes the context."""
    adapter = BankSigmaAdapter()
    sigma = torch.rand(2, 128) + 0.5
    a = adapter(sigma, torch.tensor([1, 1]))
    b = adapter(sigma, torch.tensor([100, 100]))
    assert not torch.allclose(a, b)


def test_cosine_is_bounded():
    block = TrialRelationBlock()
    out = block(torch.randn(16, 128), torch.randn(16, 128), torch.rand(16, 128) + 0.5,
                torch.tensor([8] * 16))
    assert ((out["cosine"] >= -1.0) & (out["cosine"] <= 1.0)).all()
