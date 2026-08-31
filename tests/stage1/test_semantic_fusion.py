"""Serial fusion order, independence, gating, and ablation tests (contract §7)."""

from __future__ import annotations

import pytest
import torch

from stage1.semantic_fusion import SemanticFusion


@pytest.fixture(scope="module")
def fusion():
    f = SemanticFusion(dim=128, heads=4, attention_dropout=0.0, gamma_total_init=0.1)
    f.eval()
    return f


@pytest.fixture(scope="module")
def tokens():
    torch.manual_seed(3)
    h0 = torch.randn(6, 192, 128)
    t = torch.randn(6, 192, 128)
    return h0, t


def test_execution_order_and_attention2_gets_bridge_output(fusion, tokens):
    h0, t = tokens
    calls = []

    def make_hook(name):
        def hook(module, args, output):
            calls.append(name)

        return hook

    h1 = fusion.injection1.register_forward_hook(make_hook("attn1"))
    h2 = fusion.bridge.register_forward_hook(make_hook("bridge"))
    h3 = fusion.injection2.register_forward_hook(make_hook("attn2"))
    try:
        out = fusion(h0, t)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
    assert calls == ["attn1", "bridge", "attn2"]

    # Attention 2 receives the bridge output as its query, not H0. The raw
    # query is the input of norm_q (before the q projection).
    query2 = []

    def qhook(module, args, output):
        query2.append(args[0].clone())

    hook = fusion.injection2.norm_q.register_forward_hook(qhook)
    try:
        fusion(h0, t)
    finally:
        hook.remove()
    assert torch.allclose(query2[0], out["bridge_tokens"])


def test_attention_modules_have_independent_parameters(fusion):
    p1 = [p.data_ptr() for p in fusion.injection1.parameters()]
    p2 = [p.data_ptr() for p in fusion.injection2.parameters()]
    assert not set(p1) & set(p2)
    for name in ["q_proj", "k_proj", "v_proj", "out_proj", "norm_q", "norm_kv"]:
        a = getattr(fusion.injection1, name)
        b = getattr(fusion.injection2, name)
        assert a is not b
    assert not any(
        (p is q) for p in fusion.injection1.parameters() for q in fusion.injection2.parameters()
    )


def test_zero_gamma_makes_fusion_independent_of_semantics(fusion, tokens):
    h0, t = tokens
    with torch.no_grad():
        fusion.gamma1.copy_(torch.tensor(0.0))
        fusion.gamma2.copy_(torch.tensor(0.0))
    out_a = fusion(h0, t)["fused"]
    out_b = fusion(h0, torch.randn_like(t))["fused"]
    assert torch.allclose(out_a, out_b, atol=1e-6)
    # Also equals the manual path: H1 = H0, M = H1 + eta*bridge(H1), Z = LN(M).
    with torch.no_grad():
        h1 = h0
        m = h1 + fusion.eta * fusion.bridge_drop(fusion.bridge(h1))
        expected = fusion.norm_out(m)
    assert torch.allclose(out_a, expected, atol=1e-6)
    with torch.no_grad():
        fusion.gamma1.copy_(torch.tensor(0.05))
        fusion.gamma2.copy_(torch.tensor(0.05))


def test_zero_eta_makes_bridge_residual_identity(fusion, tokens):
    h0, t = tokens
    with torch.no_grad():
        fusion.eta.copy_(torch.tensor(0.0))
        h1 = fusion.injection1(h0, t)[0]
        h1 = h0 + fusion.gamma1 * h1
        out = fusion(h0, t)
    assert torch.allclose(out["bridge_tokens"], h1)
    with torch.no_grad():
        fusion.eta.copy_(torch.tensor(0.1))


def test_bridge_preserves_shape_and_mixes_neighbors():
    from stage1.spatial_bridge import SpatialBridge

    bridge = SpatialBridge(dim=128, expansion_ratio=2.0, kernel_size=3)
    x = torch.zeros(2, 192, 128)
    x[:, 10, 5] = 1.0  # impulse at token index 10 (grid row 0, col 10)
    out = bridge(x)
    assert out.shape == (2, 192, 128)
    # Row-major 12x16: neighbors of token 10 are 9, 11 (row 0) and 26, 6 (row 1).
    neighbor_norms = torch.norm(out[0, [9, 11, 26, 6]], dim=1)
    assert torch.any(neighbor_norms > 0)


def test_single_attention_ablation_preserves_interface(tokens):
    h0, t = tokens
    f = SemanticFusion(
        dim=128, heads=4, attention_dropout=0.0, gamma_total_init=0.1, single_attention=True
    )
    out = f(h0, t)
    assert out["fused"].shape == (6, 192, 128)
    assert out["attention2_weights"] is None
    assert out["attention1_tokens"].shape == (6, 192, 128)
    assert out["bridge_tokens"].shape == (6, 192, 128)


def test_no_semantic_mode_same_interface(tokens):
    h0, t = tokens
    f = SemanticFusion(
        dim=128, heads=4, attention_dropout=0.0, gamma_total_init=0.1,
        semantic_source="none",
    )
    f.eval()  # dropout off: semantic bypass must be exactly deterministic
    out_a = f(h0, None)["fused"]
    out_b = f(h0, torch.randn_like(t))["fused"]
    assert out_a.shape == (6, 192, 128)
    assert torch.allclose(out_a, out_b)  # semantic input fully bypassed
    # The spatial bridge is retained in no-semantic mode.
    assert not isinstance(f.bridge, torch.nn.Identity)


def test_identity_bridge_ablation(tokens):
    h0, t = tokens
    f = SemanticFusion(
        dim=128, heads=4, attention_dropout=0.0, gamma_total_init=0.1,
        bridge_kind="identity",
    )
    out = f(h0, t)
    assert out["fused"].shape == (6, 192, 128)
    assert isinstance(f.bridge, torch.nn.Identity)
