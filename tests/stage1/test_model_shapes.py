"""Stage-1 model tensor-shape tests (contract §8)."""

from __future__ import annotations

import pytest
import torch

from stage1.config import Stage1Config
from stage1.model import Stage1Model, summarize_model
from stage1.types import Stage1Batch


def _make_batch(n: int, s: int, device="cpu") -> Stage1Batch:
    return Stage1Batch(
        heatmaps=torch.randn(n, 3, 48, 64),
        unique_dino_tokens=torch.randn(s, 768, 384),
        trial_to_stimulus_slot=torch.tensor([i % s for i in range(n)], dtype=torch.int64),
        stimulus_indices=torch.arange(s, dtype=torch.int64),
        subject_ids=[f"{i:03d}" for i in range(n)],
        stimulus_ids=[f"s{i % s}" for i in range(n)],
        trial_uids=[f"uid-{i}" for i in range(n)],
        groups=["HC"] * n,
    ).to_device(device)


@pytest.fixture(scope="module")
def model(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    m = Stage1Model(cfg)
    m.eval()
    return m, cfg


@pytest.mark.parametrize("n,s", [(2, 1), (64, 8), (4, 2)])
def test_forward_tensor_shapes(model, n, s):
    m, cfg = model
    batch = _make_batch(n, s)
    token_mask = torch.zeros(n, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    out = m(
        batch,
        token_mask,
        return_fused=True,
        return_attention1_tokens=True,
        return_bridge_tokens=True,
        return_pooling_weights=True,
        return_semantic_tokens=True,
        return_heatmap_tokens=True,
        debug_attention=True,
    )
    assert out.reconstruction.shape == (n, 3, 48, 64)
    assert out.trial_embedding.shape == (n, 128)
    assert out.mask.shape == (n, 192)
    assert out.fused_tokens.shape == (n, 192, 128)
    assert out.attention1_tokens.shape == (n, 192, 128)
    assert out.bridge_tokens.shape == (n, 192, 128)
    assert out.semantic_tokens.shape == (n, 192, 128)
    assert out.heatmap_tokens.shape == (n, 192, 128)
    assert out.pooling_weights.shape == (n, 192)
    assert out.attention1_weights.shape == (n, 4, 192, 192)
    assert out.attention2_weights.shape == (n, 4, 192, 192)
    assert torch.all(torch.isfinite(out.reconstruction))
    assert torch.all(torch.isfinite(out.trial_embedding))


def test_gamma_initialization_contract(model):
    m, cfg = model
    assert m.fusion.gamma1.item() + m.fusion.gamma2.item() == pytest.approx(
        cfg.model.semantic_gamma_total_init
    )
    assert m.fusion.gamma1.item() == pytest.approx(cfg.model.semantic_gamma_total_init / 2)


def test_unmasked_inference(model):
    m, cfg = model
    batch = _make_batch(2, 1)
    out = m(batch, token_mask=None)
    assert out.mask.shape == (2, 192)
    assert not out.mask.any()  # norm-bank mode: fully unmasked


def test_parameter_counts_reported(model):
    m, cfg = model
    summary = summarize_model(m)
    assert summary.n_parameters_total > 0
    assert summary.n_frozen == 0  # all Stage-1 modules are trainable
    for name in ["heatmap_encoder", "semantic_adapter", "fusion", "decoder", "pooling"]:
        assert any(mod.startswith(name) for mod in summary.trainable_modules)


@pytest.mark.smoke
def test_real_data_one_batch_forward():
    from stage1.dataset import Stage1Dataset
    from stage1.sampler import StimulusGroupedHCBatchSampler

    cfg = Stage1Config.from_dict(
        {
            "seed": 2026,
            "fold": 0,
            "paths": {
                "processed_root": "/root/EMS-Project/processed_dataset",
                "dino_root": "/root/EMS-Project/stimulus_features/dino_vits16",
                "cv_fold_dir": "/root/EMS-Project/CV/5fold_seed2026/fold_0",
                "output_root": "/tmp/stage1_smoke_outputs",
            },
        }
    )
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    sampler = StimulusGroupedHCBatchSampler(
        ds.stimulus_groups(),
        stimuli_per_batch=2,
        hc_per_stimulus=2,
        min_hc_per_stimulus=2,
        seed=cfg.seed,
        fold=0,
        epoch=0,
    )
    batch_indices = next(iter(sampler))
    batch = ds.collate_from_indices(batch_indices)
    assert batch.n_trials == 4
    assert batch.n_unique_stimuli == 2
    assert all(g == "HC" for g in batch.groups)
    model = Stage1Model(cfg)
    model.eval()
    token_mask = torch.zeros(batch.n_trials, 192, dtype=torch.bool)
    token_mask[:, :67] = True
    with torch.inference_mode():
        out = model(batch, token_mask)
    assert out.reconstruction.shape == (4, 3, 48, 64)
    assert out.trial_embedding.shape == (4, 128)
    assert torch.all(torch.isfinite(out.trial_embedding))
