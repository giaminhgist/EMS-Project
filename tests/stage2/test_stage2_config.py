"""Configuration framework tests (guide 06 §3-§4, §10): strict parsing,
duplicate-key rejection, hashing invariance and resolution order."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stage2.ablations import resolve_ablation_config
from stage2.config import (
    ConfigError,
    Stage2Config,
    config_hash,
    deep_merge,
    load_yaml_dict,
)

BASE_PATH = Path(__file__).resolve().parents[2] / "configs" / "stage2" / "base.yaml"


def test_base_yaml_resolves_into_typed_config():
    cfg = Stage2Config.from_yaml(BASE_PATH)
    assert cfg.ablation == "base"
    assert cfg.model.bank_mode == "trial"
    assert cfg.model.freeze_encoder is True
    assert cfg.model.query_pooling == "attention"
    assert cfg.model.category_balanced_attention is True
    assert cfg.model.subject_transformer_layers == 1
    assert cfg.model.auxiliary_evidence_head is True
    assert cfg.bank.train_mode == "crossfit"
    assert cfg.bank.require_token_banks is False
    assert cfg.loss.lambda_aux == 0.3
    assert cfg.loss.lambda_match == 0.1
    assert cfg.optimization.alignment_epochs == 10
    assert cfg.optimization.classification_epochs == 50
    assert cfg.validation.selection_metric == "val_balanced_accuracy"
    assert cfg.evaluation_regime == "pilot_existing_stage1"


def test_unknown_keys_fail(tmp_path):
    raw = load_yaml_dict(BASE_PATH)
    raw["model"]["bogus_field"] = True
    with pytest.raises(ConfigError, match="unknown model fields"):
        Stage2Config.from_dict(raw)
    raw2 = load_yaml_dict(BASE_PATH)
    raw2["bogus_section"] = {}
    with pytest.raises(ConfigError, match="unknown config fields"):
        Stage2Config.from_dict(raw2)


def test_duplicate_yaml_keys_fail(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("model:\n  d_model: 128\n  d_model: 256\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_yaml_dict(path)


def test_string_false_rejected(tmp_path):
    raw = load_yaml_dict(BASE_PATH)
    raw["model"]["freeze_encoder"] = "false"
    with pytest.raises(ConfigError, match="expected boolean"):
        Stage2Config.from_dict(raw)
    raw2 = load_yaml_dict(BASE_PATH)
    raw2["loss"]["lambda_match"] = "0.1"
    with pytest.raises(ConfigError, match="expected number"):
        Stage2Config.from_dict(raw2)
    raw3 = load_yaml_dict(BASE_PATH)
    raw3["sampler"]["subject_batch_size"] = "4"
    with pytest.raises(ConfigError, match="expected integer"):
        Stage2Config.from_dict(raw3)


def test_config_hash_is_key_order_invariant():
    a = load_yaml_dict(BASE_PATH)
    # Rebuild with every section and nested mapping in reversed insertion order.
    b = {}
    for section in reversed(list(a)):
        if isinstance(a[section], dict):
            b[section] = {k: a[section][k] for k in reversed(list(a[section]))}
        else:
            b[section] = a[section]
    assert config_hash(a) == config_hash(b)
    changed = load_yaml_dict(BASE_PATH)
    changed["loss"]["lambda_aux"] = 0.5
    assert config_hash(changed) != config_hash(a)


def test_deep_merge_does_not_mutate_base():
    base = {"model": {"d_model": 128, "bank_mode": "trial"}, "loss": {"lambda_aux": 0.3}}
    snapshot = yaml.safe_dump(base, sort_keys=True)
    merged = deep_merge(base, {"model": {"bank_mode": "trial_and_fused_token"}})
    assert merged["model"]["bank_mode"] == "trial_and_fused_token"
    assert merged["model"]["d_model"] == 128
    assert yaml.safe_dump(base, sort_keys=True) == snapshot  # base untouched


def test_cli_overrides_apply_after_overlay():
    resolved = resolve_ablation_config(
        BASE_PATH, "no_aux_loss", overrides={"loss.lambda_aux": 0.7}, fold=0
    )
    assert resolved.config.loss.lambda_aux == 0.7  # CLI wins over the overlay
    resolved2 = resolve_ablation_config(BASE_PATH, "no_aux_loss", fold=0)
    assert resolved2.config.loss.lambda_aux == 0.0


def test_override_unknown_key_fails():
    with pytest.raises(ConfigError):
        resolve_ablation_config(BASE_PATH, "no_bank", overrides={"model.bogus": 1}, fold=0)


def test_multiple_named_ablations_rejected():
    with pytest.raises(ConfigError, match="multiple named ablations"):
        resolve_ablation_config(BASE_PATH, "no_bank+no_aux_loss", fold=0)


def test_unknown_ablation_name_rejected():
    with pytest.raises(ConfigError, match="unknown ablation"):
        resolve_ablation_config(BASE_PATH, "does_not_exist", fold=0)


def test_incompatible_bank_model_combination_rejected(tmp_path):
    raw = load_yaml_dict(BASE_PATH)
    raw["model"]["bank_mode"] = "trial_and_fused_token"
    raw["model"]["bank_features_active"] = False
    with pytest.raises(ConfigError, match="bank_features_active"):
        Stage2Config.from_dict(raw)
    raw2 = load_yaml_dict(BASE_PATH)
    raw2["model"]["encoder_unfreeze_last_block"] = True
    raw2["model"]["freeze_encoder"] = False
    with pytest.raises(ConfigError, match="requires model.freeze_encoder"):
        Stage2Config.from_dict(raw2)


def test_resolved_config_records_spec_and_diff():
    resolved = resolve_ablation_config(BASE_PATH, "no_match_loss", fold=0)
    assert resolved.spec.name == "no_match_loss"
    assert set(resolved.changed_keys) == {"loss.lambda_match", "optimization.alignment_epochs"}
    assert any("loss.lambda_match: 0.1 -> 0.0" in e for e in resolved.diff_entries)
    assert resolved.config_hash
    assert resolved.config.fold == 0
