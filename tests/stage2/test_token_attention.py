"""Fused-token branch topology tests (guide 05 §13, §16.3)."""

from __future__ import annotations

import torch
import pytest

from stage2.token_attention import (
    FusedTokenBranch,
    LocalBankMHA,
    SpatialBridge,
    build_window_index,
)


def make_inputs(n: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    return (
        torch.randn(n, 192, 128),
        torch.randn(n, 192, 128),
        torch.rand(n, 192, 128) + 0.5,
        torch.tensor([8] * n),
        torch.randn(n, 128),
    )


def test_mha1_and_mha2_have_distinct_parameter_storage():
    branch = FusedTokenBranch()
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        p1 = getattr(branch.mha1, name).weight
        p2 = getattr(branch.mha2, name).weight
        assert p1.data_ptr() != p2.data_ptr(), name
    assert branch.mha1.rel_pos_bias.data_ptr() != branch.mha2.rel_pos_bias.data_ptr()


def test_both_attention_layers_receive_gradients():
    """The z_extended path reaches both attention layers. Gradients there are
    genuinely small (chained LayerNorms), so the loss is scaled before the
    nonzero check — the tokenmatch loss supplies O(1) gradients in training."""
    branch = FusedTokenBranch()
    out = branch(*make_inputs(), keep_full_attention=True)
    ((out["z_extended"].sum() + out["token_map_flat"].sum()) * 1e9).backward()
    for layer_name, layer in (("mha1", branch.mha1), ("mha2", branch.mha2)):
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{layer_name}.{name} missing gradient"
            assert torch.isfinite(param.grad).all(), f"{layer_name}.{name} non-finite"
            assert param.grad.abs().max() > 0, f"{layer_name}.{name} zero gradient"


def test_attention2_consumes_bridge_output():
    branch = FusedTokenBranch().eval()  # deterministic dropout for the recompute
    h, mu, sigma, count, z = make_inputs()
    captured: list[torch.Tensor] = []
    branch.mha2.register_forward_hook(lambda module, args, output: captured.append(args[0]))
    branch(h, mu, sigma, count, z)
    assert len(captured) == 1
    # Recompute Hb with the branch's own submodules: attention 2's query input
    # must equal Hb = LayerNorm(H1 + gate * bridge(H1)), not R0.
    q_tok, n_mu, n_ctx, rho = branch.projections(h, mu, sigma, count)
    cos = (torch.nn.functional.normalize(q_tok, dim=-1)
           * torch.nn.functional.normalize(n_mu, dim=-1)).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    r0 = branch.token_relation(q_tok, n_mu, cos, rho)
    h1 = branch.norm1(r0 + branch.mha1(r0, n_mu)[0])
    expected_hb = branch.norm_b(h1 + branch.bridge_gate * branch.bridge(h1))
    assert torch.allclose(captured[0], expected_hb)
    assert not torch.allclose(captured[0], r0)  # Hb is not R0


def test_neighbor_windows_do_not_wrap_grid_edges():
    index, valid = build_window_index()
    # Top-left corner (0,0): only self, right, down, down-right are valid.
    assert valid[0].tolist() == [False, False, False, False, True, True, False, True, True]
    # Left edge (1,0) and right edge (0,15): no horizontal wrap. Offsets are
    # ordered row-major: [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)].
    assert valid[16].tolist() == [False, True, True, False, True, True, False, True, True]
    assert valid[15].tolist() == [False, False, False, True, True, False, True, True, False]
    # The first column never receives the last column of the row above.
    for row in range(12):
        t = row * 16
        right_edges = {c for c in range(row * 16, row * 16 + 16) if c % 16 == 15}
        assert index[t, 0].item() not in right_edges


def test_local_attention_weights_shape_and_masking():
    branch = FusedTokenBranch()
    out = branch(*make_inputs(), keep_full_attention=True)
    weights = out["token_attention_weights"]
    assert weights.shape == (2, 4, 192, 9)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    _, valid = build_window_index()
    invalid = ~valid[None, None, :, :]
    assert torch.allclose(weights * invalid, torch.zeros_like(weights), atol=1e-7)


def test_branch_outputs_and_normalized_map():
    branch = FusedTokenBranch()
    out = branch(*make_inputs(), keep_full_attention=True)
    assert out["z_extended"].shape == (2, 128)
    assert out["token_map_flat"].shape == (2, 192)
    assert ((out["token_map_flat"] >= 0.0) & (out["token_map_flat"] <= 1.0)).all()
    assert out["token_cosine"].shape == (2, 192)
    assert out["token_rho"].shape == (2, 192, 1)
    assert out["Q"].shape == (2, 192, 128)
    assert out["N_mu"].shape == (2, 192, 128)
    assert out["N_ctx"].shape == (2, 192, 128)


def test_full_attention_retained_only_on_debug():
    branch = FusedTokenBranch()
    out = branch(*make_inputs(), keep_full_attention=False)
    assert out["token_attention_weights"] is None
    assert out["token_omega"] is not None  # reduced attention is always kept


def test_spatial_bridge_round_trip_shape():
    bridge = SpatialBridge()
    out = bridge(torch.randn(3, 192, 128))
    assert out.shape == (3, 192, 128)


def test_gated_residual_bridge_changes_output():
    branch = FusedTokenBranch()
    h = torch.randn(2, 192, 128)
    branch.bridge_gate.data.fill_(0.0)
    zero_gate = branch.norm_b(h + 0.0 * branch.bridge(h))
    branch.bridge_gate.data.fill_(0.1)
    gated = branch.norm_b(h + 0.1 * branch.bridge(h))
    assert not torch.allclose(zero_gate, gated)
