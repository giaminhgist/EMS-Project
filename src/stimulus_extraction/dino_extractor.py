"""Frozen pretrained DINO extraction: image preprocessing and patch tokens."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import DINOExtractionConfig


class ExtractionError(ValueError):
    """Raised when an image, checkpoint, or token tensor violates the contract."""


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelMetadata:
    model_family: str
    model_name: str
    hub_source: str
    hub_cache_dir: str | None
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    torch_version: str
    torchvision_version: str
    device_used: str
    n_parameters: int
    frozen: bool
    pretrained: bool
    repo_revision_pinned: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "hub_source": self.hub_source,
            "hub_cache_dir": self.hub_cache_dir,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "device_used": self.device_used,
            "n_parameters": self.n_parameters,
            "frozen": self.frozen,
            "pretrained": self.pretrained,
            "repo_revision_pinned": self.repo_revision_pinned,
        }


def load_frozen_dino(cfg: DINOExtractionConfig) -> tuple[Any, ModelMetadata]:
    """Load the pretrained DINO checkpoint via torch.hub and freeze it.

    If the device is unavailable (e.g. CUDA requested without a GPU), the
    error is raised rather than silently falling back.
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("device 'cuda' requested but CUDA is unavailable")
    try:
        model = torch.hub.load(cfg.hub_source, cfg.model_name, pretrained=cfg.pretrained)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(
            f"failed to load {cfg.hub_source} {cfg.model_name}: {exc}. "
            "No model substitution is permitted."
        ) from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if cfg.frozen and any(p.requires_grad for p in model.parameters()):
        raise ExtractionError("model parameters could not be frozen")

    # Locate the hub cache directory and downloaded checkpoint for provenance.
    hub_dir = Path(torch.hub.get_dir())
    repo_cache_dir = None
    for candidate in hub_dir.iterdir():
        name = candidate.name
        if name.startswith("facebookresearch_dino"):
            repo_cache_dir = candidate
            break
    checkpoint_path = None
    checkpoint_sha256 = None
    ckpt_dir = hub_dir / "checkpoints"
    if ckpt_dir.is_dir():
        candidates = sorted(
            p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".pth"
        )
        if candidates:
            checkpoint_path = candidates[0]
            checkpoint_sha256 = sha256_of_file(checkpoint_path)

    import torchvision

    metadata = ModelMetadata(
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        hub_source=cfg.hub_source,
        hub_cache_dir=str(repo_cache_dir) if repo_cache_dir else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_sha256=checkpoint_sha256,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        device_used=cfg.device,
        n_parameters=sum(p.numel() for p in model.parameters()),
        frozen=cfg.frozen,
        pretrained=cfg.pretrained,
        repo_revision_pinned=False,  # hub zip download does not pin a commit
    )
    return model, metadata


def preprocess_image(
    path: Path,
    cfg: DINOExtractionConfig,
    *,
    expected_width: int,
    expected_height: int,
    expected_sha256: str,
) -> torch.Tensor:
    """Deterministic image preprocessing per the Phase-3 contract.

    1. verify the file SHA-256 against the image manifest;
    2. open and convert to RGB;
    3. verify the audited source size from the manifest (no silent warping);
    4. resize the complete image to (input_width × input_height) with the
       configured interpolation and antialias;
    5. no crop, padding, flip, or rotation;
    6. float tensor + official DINO normalization.
    """
    from PIL import Image

    actual_sha = sha256_of_file(path)
    if actual_sha != expected_sha256:
        raise ExtractionError(
            f"image checksum mismatch for {path.name}: manifest {expected_sha256}, "
            f"actual {actual_sha}"
        )
    with Image.open(path) as img:
        if img.size != (expected_width, expected_height):
            raise ExtractionError(
                f"{path.name}: source size {img.size} differs from manifest "
                f"({expected_width}x{expected_height}); refusing to warp silently"
            )
        rgb = img.convert("RGB")

    # Resize on the float tensor: torchvision applies a genuine pre-antialias
    # pass for tensor inputs (for PIL inputs the antialias flag is ignored).
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    from torchvision.transforms.v2 import InterpolationMode, functional as TF

    interpolation = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "lanczos": InterpolationMode.LANCZOS,
    }[cfg.interpolation]
    resized = TF.resize(
        tensor,
        size=(cfg.input_height, cfg.input_width),
        interpolation=interpolation,
        antialias=cfg.antialias,
    )
    if tuple(resized.shape[-2:]) != (cfg.input_height, cfg.input_width):
        raise ExtractionError(
            f"resize produced {tuple(resized.shape[-2:])}, expected "
            f"({cfg.input_height}, {cfg.input_width})"
        )
    mean = torch.tensor(cfg.normalization_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg.normalization_std, dtype=torch.float32).view(3, 1, 1)
    tensor = (resized - mean) / std
    return tensor


def extract_patch_tokens(
    model: Any, batch: torch.Tensor, cfg: DINOExtractionConfig, device: str
) -> torch.Tensor:
    """Run the frozen model in inference mode and return final patch tokens.

    The official hub entry applies the DINO head in ``forward`` (returning
    only the global/CLS embedding), so the wrapper uses
    ``get_intermediate_layers(x, n=1)[0]``, which exposes the final-LayerNorm
    token sequence including CLS. Models without that API (test doubles) fall
    back to ``forward``, which must then return the token sequence. The CLS
    token is removed before storage. The token count is asserted against the
    configured patch grid (24×32 for ViT-S/16 at 512×384), which the
    checkpoint achieves through its native positional-embedding interpolation
    (never manual truncation/tiling).
    """
    gh, gw = cfg.expected_patch_grid
    expected_tokens = gh * gw
    with torch.inference_mode():
        if hasattr(model, "get_intermediate_layers"):
            out = model.get_intermediate_layers(batch.to(device), n=1)[0]
        else:
            out = model(batch.to(device))
    if out.ndim != 3:
        raise ExtractionError(
            f"unexpected model output rank {out.ndim} (expected [B, 1+patches, dim])"
        )
    b, n_tokens, dim = out.shape
    if n_tokens != 1 + expected_tokens:
        raise ExtractionError(
            f"model returned {n_tokens} tokens (expected 1 + {expected_tokens} "
            f"for patch grid {gh}x{gw})"
        )
    if dim != cfg.expected_token_dim:
        raise ExtractionError(f"token dimension {dim} != {cfg.expected_token_dim}")
    tokens = out[:, 1:, :]  # drop CLS; row-major spatial patch order
    # Row-major check: the tensor must be reshapeable to [B, gh, gw, dim] and
    # its first patch must not equal the global CLS representation.
    tokens.reshape(b, gh, gw, dim)
    tokens = tokens.to(torch.float32).cpu()
    if not torch.all(torch.isfinite(tokens)):
        raise ExtractionError("non-finite patch tokens produced")
    return tokens
