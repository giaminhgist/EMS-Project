"""Checkpoint tests (guide 07 §13, §17.10-§17.13, §17.16)."""

from __future__ import annotations

import torch
import pytest

from conftest import make_full_model_fixture
from stage2.checkpoint import (
    CheckpointError,
    load_stage2_checkpoint,
    load_weights_only_stage2,
    save_stage2_checkpoint,
    verify_resume_compatibility,
)
from stage2.ablations import resolve_ablation_config, BASE_CONFIG_DEFAULT


def _save_synthetic_checkpoint(tmp_path, model, cfg, config_hash, **overrides):
    kwargs = dict(
        model=model,
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]),
        scheduler=None,
        scaler=None,
        epoch=1,
        phase_epoch=0,
        global_step=12,
        training_phase="2B",
        best_metric=0.75,
        best_epoch=1,
        best_rule_tuple=(0.75, 0.8, -0.4, -1),
        early_stopping_counter=2,
        optimizer_step_count=12,
        skipped_optimizer_step_count=0,
        fold=0,
        run_id="synthetic_run",
        cfg=cfg,
        config_hash=config_hash,
        source_checksums={},
        stage1_checkpoint_path="/tmp/ckpt.pt",
        stage1_checkpoint_sha256="a" * 64,
        bank_manifest_path="/tmp/manifest.json",
        bank_checksums={"array_sha256": {"mu_trial": "b" * 64}},
        ablation_spec={"name": "base", "scientific_question": "q", "declared_changes": [],
                       "required_bank_capabilities": ["trial_bank"], "forbidden_with": [],
                       "interpretation": "", "is_negative_control": False, "reference": "base"},
        ablation_diff=[],
        history_row_hash="c" * 64,
        calibration_state=None,
        device="cpu",
        sampler_epoch=1,
    )
    kwargs.update(overrides)
    path = tmp_path / "ckpt.pt"
    sha = save_stage2_checkpoint(path, **kwargs)
    return path, sha, kwargs


def test_checkpoint_roundtrip_preserves_all_meta(tmp_path):
    stack = make_full_model_fixture(tmp_path)
    model = stack["model"]
    cfg = stack["cfg"]
    path, sha, kwargs = _save_synthetic_checkpoint(tmp_path, model, cfg, "d" * 64)
    contents = load_stage2_checkpoint(path)
    assert contents.sha256 == sha
    meta = contents.meta
    assert meta["epoch"] == 1
    assert meta["training_phase"] == "2B"
    assert meta["best_rule_tuple"] == [0.75, 0.8, -0.4, -1]
    assert meta["ablation_spec"]["name"] == "base"
    assert meta["ablation_diff"] == []
    assert meta["config_resolved"]["seed"] == cfg.seed
    assert meta["rng_state"] is not None
    # Model state reloads exactly.
    model2 = stack["model"]
    model2.load_state_dict(contents.state_dict)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), n1


def test_atomic_save_leaves_no_partial_target(tmp_path):
    stack = make_full_model_fixture(tmp_path)
    path, sha, _ = _save_synthetic_checkpoint(tmp_path, stack["model"], stack["cfg"], "d" * 64)
    assert path.is_file()
    assert not list(tmp_path.glob(".ckpt.pt.*.tmp")), "no partial temp file may remain"
    # Repeated saves replace the target cleanly.
    path2, sha2, _ = _save_synthetic_checkpoint(tmp_path, stack["model"], stack["cfg"], "d" * 64, epoch=2)
    assert sha2 != sha
    assert load_stage2_checkpoint(path2).meta["epoch"] == 2


def test_incompatible_resume_rejected(tmp_path):
    stack = make_full_model_fixture(tmp_path)
    path, _, kwargs = _save_synthetic_checkpoint(tmp_path, stack["model"], stack["cfg"], "d" * 64)
    contents = load_stage2_checkpoint(path)
    with pytest.raises(CheckpointError, match="resume rejected"):
        verify_resume_compatibility(
            contents,
            cfg=stack["cfg"],
            fold=1,  # wrong fold
            run_id="synthetic_run",
            config_hash="d" * 64,
            ablation_name="base",
            stage1_checkpoint_sha256="a" * 64,
            bank_checksums={"array_sha256": {"mu_trial": "b" * 64}},
        )
    with pytest.raises(CheckpointError, match="resume rejected"):
        verify_resume_compatibility(
            contents,
            cfg=stack["cfg"],
            fold=0,
            run_id="synthetic_run",
            config_hash="e" * 64,  # different config
            ablation_name="base",
            stage1_checkpoint_sha256="a" * 64,
            bank_checksums={"array_sha256": {"mu_trial": "b" * 64}},
        )
    with pytest.raises(CheckpointError, match="resume rejected"):
        verify_resume_compatibility(
            contents,
            cfg=stack["cfg"],
            fold=0,
            run_id="synthetic_run",
            config_hash="d" * 64,
            ablation_name="no_bank",  # different ablation
            stage1_checkpoint_sha256="a" * 64,
            bank_checksums={"array_sha256": {"mu_trial": "b" * 64}},
        )
    # The compatible case passes.
    verify_resume_compatibility(
        contents,
        cfg=stack["cfg"],
        fold=0,
        run_id="synthetic_run",
        config_hash="d" * 64,
        ablation_name="base",
        stage1_checkpoint_sha256="a" * 64,
        bank_checksums={"array_sha256": {"mu_trial": "b" * 64}},
    )


def test_weight_only_init_reports_mismatch(tmp_path):
    stack = make_full_model_fixture(tmp_path)
    path, _, _ = _save_synthetic_checkpoint(tmp_path, stack["model"], stack["cfg"], "d" * 64)
    meta = load_weights_only_stage2(path, stack["model"])
    assert meta["run_id"] == "synthetic_run"
    # A structurally different model must be rejected.
    import copy

    broken = copy.deepcopy(stack["model"])
    broken.aggregator.main_head.head = torch.nn.Linear(128, 2)
    with pytest.raises(CheckpointError, match="architecture mismatch"):
        load_weights_only_stage2(path, broken)


def test_ablation_spec_reaches_checkpoint(tmp_path):
    stack = make_full_model_fixture(tmp_path)
    resolved = resolve_ablation_config(BASE_CONFIG_DEFAULT, "no_aux_loss", fold=0)
    cfg = resolved.config
    spec = {
        "name": resolved.spec.name,
        "scientific_question": resolved.spec.scientific_question,
        "declared_changes": list(resolved.spec.declared_changes),
        "required_bank_capabilities": list(resolved.spec.required_bank_capabilities),
        "forbidden_with": list(resolved.spec.forbidden_with),
        "interpretation": resolved.spec.interpretation,
        "is_negative_control": resolved.spec.is_negative_control,
        "reference": resolved.spec.reference,
    }
    path, _, _ = _save_synthetic_checkpoint(
        tmp_path, stack["model"], cfg, resolved.config_hash,
        ablation_spec=spec, ablation_diff=resolved.diff_entries,
    )
    contents = load_stage2_checkpoint(path)
    assert contents.meta["ablation_spec"]["name"] == "no_aux_loss"
    assert contents.meta["ablation_diff"] == resolved.diff_entries
