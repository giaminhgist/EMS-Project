"""Gradient-flow tests: backprop reaches every Stage-1 component."""

from __future__ import annotations

import pytest
import torch

from stage1.config import Stage1Config
from stage1.losses import stage1_loss
from stage1.model import Stage1Model
from stage1.types import Stage1Batch


def _batch(n=4, s=2, seed=0) -> Stage1Batch:
    g = torch.Generator().manual_seed(seed)
    return Stage1Batch(
        heatmaps=torch.randn(n, 3, 48, 64, generator=g),
        unique_dino_tokens=torch.randn(s, 768, 384, generator=g),
        trial_to_stimulus_slot=torch.tensor([i % s for i in range(n)]),
        stimulus_indices=torch.arange(s),
        subject_ids=["000", "005", "000", "005"],
        stimulus_ids=["a1.jpg", "b1.jpg", "a1.jpg", "b1.jpg"],
        trial_uids=[f"u{i}" for i in range(n)],
        groups=["HC"] * n,
    )


def test_backprop_reaches_all_components(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    model = Stage1Model(cfg)
    model.train()
    batch = _batch()
    token_mask = torch.zeros(4, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    out = model(batch, token_mask)
    loss = stage1_loss(
        out.reconstruction,
        batch.heatmaps,  # target = input transform (synthetic passthrough)
        token_mask,
        out.trial_embedding,
        batch.trial_to_stimulus_slot,
        lambda_norm=0.1,
        min_hc_per_stimulus=2,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)

    for name in ["heatmap_encoder", "semantic_adapter", "fusion", "decoder", "pooling"]:
        module = getattr(model, name)
        for pname, p in module.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name}.{pname} has no gradient"
                assert torch.all(torch.isfinite(p.grad)), f"{name}.{pname} gradient non-finite"


def test_attention_modules_both_receive_finite_gradients(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    model = Stage1Model(cfg)
    model.train()
    batch = _batch()
    token_mask = torch.zeros(4, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    out = model(batch, token_mask)
    loss = stage1_loss(
        out.reconstruction, batch.heatmaps, token_mask,
        out.trial_embedding, batch.trial_to_stimulus_slot,
        lambda_norm=0.1, min_hc_per_stimulus=2,
    )
    loss.total.backward()
    for attn in (model.fusion.injection1, model.fusion.injection2):
        grads = [p.grad for p in attn.parameters()]
        assert all(g is not None for g in grads)
        assert all(torch.all(torch.isfinite(g)) for g in grads)


def test_dino_tensors_have_no_gradient_and_are_unchanged(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    model = Stage1Model(cfg)
    model.train()
    batch = _batch()
    dino_before = batch.unique_dino_tokens.clone()
    token_mask = torch.zeros(4, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    out = model(batch, token_mask)
    loss = stage1_loss(
        out.reconstruction, batch.heatmaps, token_mask,
        out.trial_embedding, batch.trial_to_stimulus_slot,
        lambda_norm=0.1, min_hc_per_stimulus=2,
    )
    loss.total.backward()
    assert not batch.unique_dino_tokens.requires_grad
    assert batch.unique_dino_tokens.grad is None
    assert torch.equal(batch.unique_dino_tokens, dino_before)


def test_forward_backward_finite_synthetic(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    model = Stage1Model(cfg)
    batch = _batch(n=2, s=1)
    token_mask = torch.zeros(2, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    out = model(batch, token_mask)
    assert torch.all(torch.isfinite(out.reconstruction))
    assert torch.all(torch.isfinite(out.trial_embedding))
    loss = stage1_loss(
        out.reconstruction, batch.heatmaps, token_mask,
        out.trial_embedding, batch.trial_to_stimulus_slot,
        lambda_norm=0.1, min_hc_per_stimulus=2,
    )
    loss.total.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.all(torch.isfinite(p.grad))
