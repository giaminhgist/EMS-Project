"""Named-ablation config tests: parse, single-factor change, forward/backward."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stage1.config import Stage1Config
from stage1.losses import stage1_loss
from stage1.model import Stage1Model
from stage1.types import Stage1Batch

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "configs" / "stage1" / "base.yaml"
ABLATION_NAMES = [
    "no_semantic", "aligned_add_fusion", "concat_fusion", "single_cross_attention",
    "no_spatial_bridge", "token_mlp_bridge", "no_fusion_residual", "fixed_fusion_gates",
    "mean_pooling", "no_norm_loss", "full_reconstruction", "fixation_only",
    "no_transition_channel", "no_temporal_channel", "avgpool_semantic_adapter",
]


def _flat_sections(d: dict) -> dict:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub, v in value.items():
                out[f"{key}.{sub}"] = v
        else:
            out[key] = value
    return out


def test_every_ablation_parses_and_changes_intended_field():
    base_cfg = Stage1Config.load_base_with_ablation(BASE, None)
    base_flat = _flat_sections(base_cfg.to_dict())
    for name in ABLATION_NAMES:
        cfg = Stage1Config.load_base_with_ablation(BASE, name)
        assert cfg.ablation == name
        flat = _flat_sections(cfg.to_dict())
        diffs = {k for k in base_flat if base_flat[k] != flat.get(k)}
        # Every ablation changes at least one field (the intended one) and
        # adds only the ablation marker.
        assert diffs, name
        # No ablation may silently change the seed/fold/sampler sizes.
        for protected in ["seed", "fold", "sampler.stimuli_per_batch", "sampler.hc_per_stimulus"]:
            assert flat.get(protected) == base_flat[protected], (name, protected)


def test_every_ablation_forward_backward_finite():
    for name in ABLATION_NAMES:
        cfg = Stage1Config.load_base_with_ablation(BASE, name)
        cfg = Stage1Config(**{**cfg.to_dict(), "paths": None})
        model = Stage1Model(cfg)
        model.train()
        n, s = 4, 2
        batch = Stage1Batch(
            heatmaps=torch.randn(n, cfg.model.input_channels, 48, 64),
            unique_dino_tokens=torch.randn(s, 768, 384),
            trial_to_stimulus_slot=torch.tensor([0, 1, 0, 1]),
            stimulus_indices=torch.arange(s),
            subject_ids=["000", "005", "000", "005"],
            stimulus_ids=["a1.jpg", "b1.jpg", "a1.jpg", "b1.jpg"],
            trial_uids=[f"u{i}" for i in range(n)],
            groups=["HC"] * n,
        )
        mask = torch.zeros(n, 192, dtype=torch.bool)
        mask[:, :67] = True
        out = model(batch, mask)
        # Full reconstruction ablation ignores masks; channel ablations use
        # the declared channel count.
        assert out.reconstruction.shape == (n, cfg.model.input_channels, 48, 64)
        loss = stage1_loss(
            out.reconstruction, batch.heatmaps, mask,
            out.trial_embedding, batch.trial_to_stimulus_slot,
            lambda_norm=0.1, min_hc_per_stimulus=2,
            channel_weights=tuple(cfg.loss.channel_weights),
            channel_map=tuple(cfg.model.active_channels),
            scope=cfg.masking.reconstruction_scope,
        )
        loss.total.backward()
        assert torch.isfinite(loss.total), name
        assert all(
            p.grad is None or torch.all(torch.isfinite(p.grad))
            for p in model.parameters()
        ), name


def test_invalid_combinations_rejected():
    from stage1.config import ConfigError

    with pytest.raises(ConfigError, match="masked-only"):
        Stage1Config.load_base_with_ablation(
            BASE, None
        ).__class__.from_dict(
            {
                **Stage1Config.load_base_with_ablation(BASE, None).to_dict(),
                "masking": {"train_mask_ratio": 0.0},
            }
        )
    with pytest.raises(ConfigError, match="channel_weights"):
        # channel weights must match input_channels for channel ablations.
        cfg = Stage1Config.load_base_with_ablation(BASE, "fixation_only")
        Stage1Config.from_dict({**cfg.to_dict(), "loss": {**cfg.loss.to_dict(), "channel_weights": [1.0, 1.0]}})


def test_channel_ablations_use_declared_channels():
    cfg = Stage1Config.load_base_with_ablation(BASE, "fixation_only")
    assert cfg.model.input_channels == 1
    assert cfg.model.active_channels == (0,)
    assert cfg.loss.channel_weights == (1.0,)
    cfg2 = Stage1Config.load_base_with_ablation(BASE, "no_temporal_channel")
    assert cfg2.model.active_channels == (0, 1)
    # The dataset slices channels accordingly.
    from stage1.dataset import Stage1Dataset

    ds = Stage1Dataset(
        cfg2.paths.processed_root, cfg2.paths.dino_root, cfg2.paths.cv_fold_dir, "train",
        active_channels=cfg2.model.active_channels,
    )
    assert ds[0].heatmap.shape == (2, 48, 64)
