"""Transferred Stage-1 heatmap encoder wrapper (guide 05 §4, contracts §4/§8).

Instantiates the exact Stage-1 ``HeatmapPatchEncoder`` from the checkpoint's
resolved config, loads only ``heatmap_encoder.*`` keys, verifies fold and
SHA-256 identity, freezes the encoder in base mode and exposes an explicit
``unfreeze_last_block()`` for the named fine-tuning ablation only.

Stage 2 always calls the encoder unmasked (``token_mask=None``); the wrapper
does not accept a mask argument at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from stage1.checkpoint import load_checkpoint
from stage1.heatmap_encoder import HeatmapPatchEncoder

from .contracts import sha256_of_file

ENCODER_PREFIX = "heatmap_encoder."


class TransferredEncoderError(ValueError):
    pass


@dataclass
class EncoderParameterReport:
    total: int
    trainable: int
    frozen: int
    trainable_names: list[str]
    frozen_names: list[str]


class TransferredHeatmapEncoder(nn.Module):
    """Frozen-by-default Stage-1 heatmap encoder loaded from one approved checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        expected_sha256: str,
        fold: int,
        freeze: bool = True,
        device: str = "cpu",
        random_init: bool = False,
        init_seed: int = 2026,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise TransferredEncoderError(f"Stage-1 checkpoint not found: {checkpoint_path}")

        actual_sha = sha256_of_file(checkpoint_path)
        if actual_sha != expected_sha256:
            raise TransferredEncoderError(
                f"checkpoint SHA-256 mismatch for {checkpoint_path}:\n"
                f"  expected {expected_sha256}\n  actual   {actual_sha}"
            )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = actual_sha
        self.expected_sha256 = expected_sha256

        contents = load_checkpoint(checkpoint_path, device)
        meta = contents.meta
        if meta.get("fold") != fold:
            raise TransferredEncoderError(
                f"checkpoint fold {meta.get('fold')} does not match requested fold {fold}"
            )
        resolved = meta.get("config_resolved")
        if not isinstance(resolved, dict) or not isinstance(resolved.get("model"), dict):
            raise TransferredEncoderError(
                f"{checkpoint_path}: checkpoint lacks a resolved model config"
            )
        model_cfg = resolved["model"]
        if model_cfg.get("positional_encoding") != "fixed_2d_sincos":
            raise TransferredEncoderError(
                f"unsupported positional encoding: {model_cfg.get('positional_encoding')!r}"
            )
        self.fold = fold
        self.stage1_run_id = meta.get("run_id")
        self.stage1_config_hash = meta.get("config_hash")
        self.random_init = bool(random_init)

        # Load only the heatmap_encoder.* keys; reject missing or unexpected ones.
        encoder_state = {
            key[len(ENCODER_PREFIX):]: tensor
            for key, tensor in contents.state_dict.items()
            if key.startswith(ENCODER_PREFIX)
        }

        # The random_encoder ablation builds the exact architecture and keeps
        # the Stage-1 weights out; checkpoint existence, SHA and fold are
        # still verified so artifact provenance is unchanged.
        generator = torch.Generator(device="cpu")
        if self.random_init:
            generator.manual_seed(int(init_seed))
        with torch.random.fork_rng(devices=[]):
            if self.random_init:
                torch.manual_seed(int(init_seed))
            encoder = HeatmapPatchEncoder(
                in_channels=int(model_cfg.get("input_channels", 3)),
                d_model=int(model_cfg.get("d_model", 128)),
                patch_size=int(model_cfg.get("heatmap_patch_size", 4)),
                grid_h=12,
                grid_w=16,
                n_residual_blocks=int(model_cfg.get("heatmap_residual_blocks", 2)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                positional_encoding=str(model_cfg.get("positional_encoding", "fixed_2d_sincos")),
            )
        expected_keys = set(encoder.state_dict())
        missing = sorted(expected_keys - set(encoder_state))
        unexpected = sorted(set(encoder_state) - expected_keys)
        if missing:
            raise TransferredEncoderError(f"encoder checkpoint keys missing: {missing}")
        if unexpected:
            raise TransferredEncoderError(f"unexpected encoder checkpoint keys: {unexpected}")
        if not self.random_init:
            encoder.load_state_dict(encoder_state)
        self.encoder = encoder

        # Per-name Stage-1 reference for the anchor loss (Stage 2C).
        self._stage1_ref = {
            name: param.detach().clone() for name, param in self.encoder.named_parameters()
        }
        self.frozen = bool(freeze)
        self.set_frozen(bool(freeze))

    # ------------------------------------------------------------- train mode

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze every encoder parameter and record the new state."""
        self.frozen = bool(frozen)
        for param in self.encoder.parameters():
            param.requires_grad_(not self.frozen)
        if self.frozen:
            self.encoder.eval()

    def unfreeze_last_block(self) -> None:
        """Named fine-tuning ablation (Stage 2C): unfreeze only the final residual block."""
        if not self.frozen:
            return
        blocks = list(self.encoder.residual_blocks)
        if not blocks:
            raise TransferredEncoderError("encoder has no residual blocks to unfreeze")
        self.frozen = False
        for name, param in self.encoder.named_parameters():
            param.requires_grad_(name.startswith(f"residual_blocks.{len(blocks) - 1}."))

    def train(self, mode: bool = True) -> "TransferredHeatmapEncoder":
        """Invariant: a frozen encoder can never re-enter training mode.

        GroupNorm has no running statistics and dropout is 0 in the frozen
        base mode, but the wrapper enforces the invariant regardless.
        """
        if mode and self.frozen:
            return super().train(False)
        return super().train(mode)

    # ------------------------------------------------------------- anchor

    def anchor_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(current, stage1_reference)`` over the unfrozen parameters only.

        ``L_anchor`` must be computed only for parameters that are actually
        trained (guide 06 §7.5); frozen parameters contribute nothing.
        """
        names = [name for name, p in self.encoder.named_parameters() if p.requires_grad]
        if not names:
            empty = torch.zeros(0)
            return empty, empty
        current = torch.cat(
            [self.encoder.get_parameter(name).reshape(-1) for name in names]
        )
        reference = torch.cat([self._stage1_ref[name].reshape(-1) for name in names])
        return current, reference

    def _weight_vector(self) -> torch.Tensor:
        parts = [p.detach().reshape(-1) for p in self.encoder.parameters()]
        return torch.cat(parts) if parts else torch.zeros(0)

    # ------------------------------------------------------------- report

    def parameter_report(self) -> EncoderParameterReport:
        trainable: list[str] = []
        frozen: list[str] = []
        for name, param in self.encoder.named_parameters():
            (trainable if param.requires_grad else frozen).append(name)
        total = sum(p.numel() for p in self.encoder.parameters())
        trainable_count = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        return EncoderParameterReport(
            total=total,
            trainable=trainable_count,
            frozen=total - trainable_count,
            trainable_names=trainable,
            frozen_names=frozen,
        )

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        # Route checkpoint loads of the Stage-2 model through the inner encoder.
        return self.encoder.load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------- forward

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """``[N,3,48,64] -> [N,192,128]``; always unmasked in Stage 2."""
        if heatmaps.dim() != 4 or heatmaps.shape[1:] != (3, 48, 64):
            raise TransferredEncoderError(
                f"heatmaps must be [N,3,48,64], got {tuple(heatmaps.shape)}"
            )
        return self.encoder(heatmaps, token_mask=None)
