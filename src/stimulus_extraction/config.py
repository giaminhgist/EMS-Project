"""Typed, strictly validated DINO extraction configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SUPPORTED_INTERPOLATIONS = {"bicubic", "bilinear", "lanczos"}
_SUPPORTED_DEVICES = {"cpu", "cuda"}


class ConfigError(ValueError):
    pass


def _coerce(value: Any, name: str, target: type, allow_none: bool = False) -> Any:
    if allow_none and value is None:
        return None
    if isinstance(value, bool) and target is not bool:
        raise ConfigError(f"config field {name!r} must be {target.__name__}")
    if not isinstance(value, target):
        raise ConfigError(f"config field {name!r} must be {target.__name__}, got {value!r}")
    return value


@dataclass(frozen=True)
class DINOExtractionConfig:
    model_family: str
    model_name: str
    hub_source: str
    pretrained: bool
    frozen: bool
    input_height: int
    input_width: int
    resize_mode: str
    interpolation: str
    antialias: bool
    center_crop: bool
    random_augmentation: bool
    output_layer: str
    include_cls_token: bool
    output_dtype: str
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    expected_patch_size: int
    expected_patch_grid: tuple[int, int]
    expected_token_dim: int
    batch_size: int
    num_workers: int
    device: str
    seed: int
    image_manifest: Path
    image_source_root: Path
    output_root: Path
    output_subdir: str
    _validated: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        # Coerce path fields regardless of construction path (from_dict passes
        # Path objects; direct construction may pass strings).
        for f in ("image_manifest", "image_source_root", "output_root"):
            value = getattr(self, f)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, f, Path(value))
        if not self._validated:
            return
        if self.model_family != "dino":
            raise ConfigError("model_family must be 'dino'")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ConfigError("input dimensions must be positive")
        if self.resize_mode != "exact":
            raise ConfigError("resize_mode must be 'exact'")
        if self.interpolation not in _SUPPORTED_INTERPOLATIONS:
            raise ConfigError(f"interpolation must be one of {sorted(_SUPPORTED_INTERPOLATIONS)}")
        if self.center_crop or self.random_augmentation:
            raise ConfigError("center_crop and random_augmentation must be false (contract)")
        if self.output_layer != "final_normalized_patch_tokens":
            raise ConfigError("only 'final_normalized_patch_tokens' is supported")
        if self.include_cls_token:
            raise ConfigError("include_cls_token must be false (contract)")
        if self.output_dtype != "float32":
            raise ConfigError("output_dtype must be 'float32'")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ConfigError("normalization mean/std must have 3 values")
        if self.expected_patch_size <= 0 or self.expected_token_dim <= 0:
            raise ConfigError("expected patch size/dim must be positive")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ConfigError("batch_size must be positive; num_workers >= 0")
        if self.device not in _SUPPORTED_DEVICES:
            raise ConfigError(f"device must be one of {sorted(_SUPPORTED_DEVICES)}")
        if self.output_subdir.strip() == "":
            raise ConfigError("output_subdir must be non-empty")

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "DINOExtractionConfig":
        known = {
            "model_family", "model_name", "hub_source", "pretrained", "frozen",
            "input_height", "input_width", "resize_mode", "interpolation", "antialias",
            "center_crop", "random_augmentation", "output_layer", "include_cls_token",
            "output_dtype", "normalization_mean", "normalization_std",
            "expected_patch_size", "expected_patch_grid", "expected_token_dim",
            "batch_size", "num_workers", "device", "seed",
            "image_manifest", "image_source_root", "output_root", "output_subdir",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{source}: unknown config fields: {unknown}")
        missing = sorted(
            k for k in known
            if k not in raw and k not in ("normalization_mean", "normalization_std")
        )
        if missing:
            raise ConfigError(f"{source}: missing required fields: {missing}")
        mean = tuple(float(v) for v in raw.get("normalization_mean", [0.485, 0.456, 0.406]))
        std = tuple(float(v) for v in raw.get("normalization_std", [0.229, 0.224, 0.225]))
        grid = raw.get("expected_patch_grid", [24, 32])
        return cls(
            model_family=_coerce(raw["model_family"], "model_family", str),
            model_name=_coerce(raw["model_name"], "model_name", str),
            hub_source=_coerce(raw["hub_source"], "hub_source", str),
            pretrained=_coerce(raw["pretrained"], "pretrained", bool),
            frozen=_coerce(raw["frozen"], "frozen", bool),
            input_height=_coerce(raw["input_height"], "input_height", int),
            input_width=_coerce(raw["input_width"], "input_width", int),
            resize_mode=_coerce(raw["resize_mode"], "resize_mode", str),
            interpolation=_coerce(raw["interpolation"], "interpolation", str),
            antialias=_coerce(raw["antialias"], "antialias", bool),
            center_crop=_coerce(raw["center_crop"], "center_crop", bool),
            random_augmentation=_coerce(raw["random_augmentation"], "random_augmentation", bool),
            output_layer=_coerce(raw["output_layer"], "output_layer", str),
            include_cls_token=_coerce(raw["include_cls_token"], "include_cls_token", bool),
            output_dtype=_coerce(raw["output_dtype"], "output_dtype", str),
            normalization_mean=mean,
            normalization_std=std,
            expected_patch_size=_coerce(raw["expected_patch_size"], "expected_patch_size", int),
            expected_patch_grid=(_coerce(grid[0], "expected_patch_grid[0]", int), _coerce(grid[1], "expected_patch_grid[1]", int)),
            expected_token_dim=_coerce(raw["expected_token_dim"], "expected_token_dim", int),
            batch_size=_coerce(raw["batch_size"], "batch_size", int),
            num_workers=_coerce(raw["num_workers"], "num_workers", int),
            device=_coerce(raw["device"], "device", str),
            seed=_coerce(raw["seed"], "seed", int),
            image_manifest=Path(_coerce(raw["image_manifest"], "image_manifest", str)),
            image_source_root=Path(_coerce(raw["image_source_root"], "image_source_root", str)),
            output_root=Path(_coerce(raw["output_root"], "output_root", str)),
            output_subdir=_coerce(raw["output_subdir"], "output_subdir", str),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DINOExtractionConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: configuration must be a mapping")
        return cls.from_dict(raw, source=str(p))

    @property
    def feature_output_dir(self) -> Path:
        return self.output_root / self.output_subdir

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_validated")
        return {
            k: (str(v) if isinstance(v, Path) else (list(v) if isinstance(v, tuple) else v))
            for k, v in d.items()
        }

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
