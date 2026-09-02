"""Transferred encoder tests (guide 05 §4, §16.2): key loading, fold/SHA
verification, freezing, the train-mode invariant and the fine-tuning escape
hatch."""

from __future__ import annotations

import torch
import pytest

from conftest import make_stage1_checkpoint
from stage2.transferred_encoder import (
    TransferredEncoderError,
    TransferredHeatmapEncoder,
)


def make_fixture(tmp_path, seed=7, fold=0):
    path, sha = make_stage1_checkpoint(tmp_path, fold=fold, seed=seed)
    return path, sha


def test_load_and_forward_shape(tmp_path):
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0, freeze=True)
    report = enc.parameter_report()
    assert report.total > 0 and report.trainable == 0 and report.frozen == report.total
    assert report.trainable_names == []
    heatmaps = torch.randn(2, 3, 48, 64)
    tokens = enc(heatmaps)
    assert tokens.shape == (2, 192, 128)
    assert torch.isfinite(tokens).all()


def test_rejects_sha_mismatch(tmp_path):
    path, sha = make_fixture(tmp_path)
    with pytest.raises(TransferredEncoderError, match="SHA-256"):
        TransferredHeatmapEncoder(path, expected_sha256="0" * 64, fold=0)


def test_rejects_fold_mismatch(tmp_path):
    path, sha = make_fixture(tmp_path, fold=2)
    with pytest.raises(TransferredEncoderError, match="fold"):
        TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)


def test_rejects_missing_encoder_key(tmp_path):
    path, sha = make_fixture(tmp_path)
    payload = torch.load(path, weights_only=False)
    del payload["model_state"]["heatmap_encoder.patch_embed.weight"]
    torch.save(payload, path)
    with pytest.raises(TransferredEncoderError, match="missing"):
        TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)


def test_rejects_unexpected_encoder_key(tmp_path):
    path, sha = make_fixture(tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["model_state"]["heatmap_encoder.bogus.weight"] = torch.zeros(1)
    torch.save(payload, path)
    with pytest.raises(TransferredEncoderError, match="unexpected"):
        TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)


def test_only_encoder_keys_are_loaded(tmp_path):
    """Non-encoder checkpoint keys (semantic adapter, pooling) are ignored."""
    path, sha = make_fixture(tmp_path)
    payload = torch.load(path, weights_only=False)
    assert any(not k.startswith("heatmap_encoder.") for k in payload["model_state"])
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)
    assert enc.parameter_report().total > 0  # loaded fine, extra keys ignored


def test_frozen_encoder_gets_no_gradients(tmp_path):
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0, freeze=True)
    heatmaps = torch.randn(2, 3, 48, 64)
    tokens = enc(heatmaps)
    assert tokens.requires_grad is False  # no path to any trainable parameter
    for name, param in enc.encoder.named_parameters():
        assert param.requires_grad is False, f"{name} must not require gradients"


def test_train_keeps_frozen_encoder_in_eval(tmp_path):
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0, freeze=True)
    enc.train()
    assert enc.training is False
    assert enc.encoder.training is False
    enc.eval()
    assert enc.training is False


def test_unfreeze_last_block_only(tmp_path):
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0, freeze=True)
    enc.unfreeze_last_block()
    assert enc.frozen is False
    trainable = [n for n, p in enc.encoder.named_parameters() if p.requires_grad]
    assert trainable, "last block must be trainable"
    assert all(name.startswith("residual_blocks.1.") for name in trainable)
    for name, param in enc.encoder.named_parameters():
        if not name.startswith("residual_blocks.1."):
            assert param.requires_grad is False, f"{name} must stay frozen"
    # Unfrozen wrapper may now enter train mode.
    enc.train()
    assert enc.training is True


def test_forward_rejects_wrong_shape(tmp_path):
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)
    with pytest.raises(TransferredEncoderError, match="N,3,48,64"):
        enc(torch.randn(2, 48, 64))


def test_forward_has_no_token_mask_argument(tmp_path):
    """Stage 2 always runs the encoder unmasked; no mask can be passed."""
    path, sha = make_fixture(tmp_path)
    enc = TransferredHeatmapEncoder(path, expected_sha256=sha, fold=0)
    with pytest.raises(TypeError):
        enc(torch.randn(2, 3, 48, 64), token_mask=None)
