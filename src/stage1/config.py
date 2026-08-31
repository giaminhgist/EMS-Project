"""Typed, strictly validated Stage-1 configuration (Phase 5/6 schema)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    heatmap_patch_size: int = 4
    heatmap_residual_blocks: int = 2
    semantic_source: str = "dino"  # dino | none
    semantic_adapter: str = "learned_depthwise_2x2"  # learned_depthwise_2x2 | avgpool_2x2
    fusion: str = "serial_attention_spatial_attention"  # serial_attention_spatial_attention | aligned_add | concat
    single_cross_attention: bool = False
    fusion_residual: bool = True
    learn_fusion_gates: bool = True
    attention_heads: int = 4
    attention_dropout: float = 0.1
    semantic_gamma_total_init: float = 0.1
    share_cross_attention_weights: bool = False
    spatial_bridge: str = "residual_dwconv_ffn"  # residual_dwconv_ffn | token_mlp_ffn | identity
    spatial_bridge_expansion_ratio: float = 2.0
    spatial_bridge_kernel_size: int = 3
    spatial_bridge_dropout: float = 0.1
    spatial_bridge_eta_init: float = 0.1
    pooling: str = "attention"  # attention | mean
    positional_encoding: str = "fixed_2d_sincos"
    input_channels: int = 3  # 3 | 2 | 1 (channel ablations)
    active_channels: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_channels", tuple(self.active_channels))

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.d_model <= 0 or self.heatmap_patch_size <= 0:
            raise ConfigError(f"{where}: d_model and heatmap_patch_size must be positive")
        if self.semantic_source not in ("dino", "none"):
            raise ConfigError(f"{where}: semantic_source must be 'dino' or 'none'")
        if self.semantic_adapter not in ("learned_depthwise_2x2", "avgpool_2x2"):
            raise ConfigError(f"{where}: unsupported semantic_adapter {self.semantic_adapter!r}")
        if self.fusion not in ("serial_attention_spatial_attention", "aligned_add", "concat"):
            raise ConfigError(f"{where}: unsupported fusion {self.fusion!r}")
        if self.attention_heads <= 0 or self.d_model % self.attention_heads != 0:
            raise ConfigError(f"{where}: d_model must be divisible by attention_heads")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ConfigError(f"{where}: attention_dropout must be in [0, 1)")
        if self.semantic_gamma_total_init < 0:
            raise ConfigError(f"{where}: semantic_gamma_total_init must be >= 0")
        if self.share_cross_attention_weights:
            raise ConfigError(
                f"{where}: share_cross_attention_weights=true is not supported "
                "(the two cross-attentions must have independent parameters)"
            )
        if self.spatial_bridge not in ("residual_dwconv_ffn", "token_mlp_ffn", "identity"):
            raise ConfigError(f"{where}: unsupported spatial_bridge {self.spatial_bridge!r}")
        if self.pooling not in ("attention", "mean"):
            raise ConfigError(f"{where}: pooling must be 'attention' or 'mean'")
        if self.positional_encoding != "fixed_2d_sincos":
            raise ConfigError(f"{where}: only fixed_2d_sincos positional encoding is supported")
        if self.input_channels not in (1, 2, 3):
            raise ConfigError(f"{where}: input_channels must be 1, 2, or 3")
        if len(self.active_channels) != self.input_channels:
            raise ConfigError(
                f"{where}: active_channels {list(self.active_channels)} must have "
                f"input_channels={self.input_channels} entries"
            )
        if sorted(set(self.active_channels)) != sorted(self.active_channels):
            raise ConfigError(f"{where}: active_channels must be strictly increasing")
        if not set(self.active_channels) <= {0, 1, 2}:
            raise ConfigError(f"{where}: active_channels must be a subset of {{0, 1, 2}}")

    @property
    def gamma_init(self) -> float:
        """gamma_1_init = gamma_2_init = semantic_gamma_total_init / 2."""
        return self.semantic_gamma_total_init / 2.0


@dataclass(frozen=True)
class MaskingConfig:
    train_mask_ratio: float = 0.35
    validation_mask_ratio: float = 0.35
    reconstruction_scope: str = "masked"  # masked | full

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if not 0.0 <= self.train_mask_ratio < 1.0 or not 0.0 <= self.validation_mask_ratio < 1.0:
            raise ConfigError(f"{where}: mask ratios must be in [0, 1)")
        if self.reconstruction_scope not in ("masked", "full"):
            raise ConfigError(f"{where}: reconstruction_scope must be 'masked' or 'full'")
        if self.reconstruction_scope == "masked" and (
            self.train_mask_ratio <= 0 or self.validation_mask_ratio <= 0
        ):
            raise ConfigError(
                f"{where}: masked-only reconstruction requires positive mask ratios"
            )


@dataclass(frozen=True)
class LossConfig:
    reconstruction: str = "smooth_l1"  # smooth_l1 | l1
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normative_metric: str = "loo_cosine"
    lambda_norm: float = 0.1
    norm_start_epoch: int = 10
    norm_ramp_epochs: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_weights", tuple(self.channel_weights))

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.reconstruction not in ("smooth_l1", "l1"):
            raise ConfigError(f"{where}: reconstruction loss must be 'smooth_l1' or 'l1'")
        if len(self.channel_weights) not in (1, 2, 3) or any(w < 0 for w in self.channel_weights):
            raise ConfigError(
                f"{where}: channel_weights must be 1-3 non-negative values "
                "(matching model.input_channels)"
            )
        if self.normative_metric != "loo_cosine":
            raise ConfigError(f"{where}: only loo_cosine normative metric is supported")
        if self.lambda_norm < 0 or self.norm_start_epoch < 0 or self.norm_ramp_epochs < 0:
            raise ConfigError(f"{where}: lambda_norm / norm_* epochs must be non-negative")


@dataclass(frozen=True)
class SamplerConfig:
    stimuli_per_batch: int = 8
    hc_per_stimulus: int = 8
    replacement: bool = False
    min_hc_per_stimulus: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.stimuli_per_batch <= 0 or self.hc_per_stimulus <= 0:
            raise ConfigError(f"{where}: sampler sizes must be positive")
        if self.min_hc_per_stimulus < 1:
            raise ConfigError(f"{where}: min_hc_per_stimulus must be >= 1")
        if self.replacement:
            raise ConfigError(
                f"{where}: replacement sampling is not supported unless explicitly "
                "approved as an ablation"
            )


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 100
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 5.0e-4
    scheduler: str = "linear_warmup_cosine"
    lr_warmup_epochs: int = 5
    gradient_clip_norm: float = 5.0
    amp: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigError(f"{where}: invalid optimization values")
        if self.optimizer != "adamw" or self.scheduler != "linear_warmup_cosine":
            raise ConfigError(f"{where}: only adamw / linear_warmup_cosine are supported")


@dataclass(frozen=True)
class ValidationConfig:
    selection_metric: str = "val_loss"
    best_eligible_after_norm_ramp: bool = True
    early_stopping_patience: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.selection_metric != "val_loss":
            raise ConfigError(f"{where}: only val_loss model selection is supported")


@dataclass(frozen=True)
class RuntimeConfig:
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    deterministic_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if self.num_workers < 0:
            raise ConfigError(f"{where}: num_workers must be >= 0")


@dataclass(frozen=True)
class PathsConfig:
    processed_root: Path
    dino_root: Path
    cv_fold_dir: Path | None = None  # fold partition dir; required for dataset init
    output_root: Path | None = None

    def __post_init__(self) -> None:
        for f in ("processed_root", "dino_root", "cv_fold_dir", "output_root"):
            value = getattr(self, f)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, f, Path(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }

    def validate(self, where: str) -> None:
        if not self.processed_root.is_dir():
            raise ConfigError(f"{where}: processed_root not found: {self.processed_root}")
        if not self.dino_root.is_dir():
            raise ConfigError(f"{where}: dino_root not found: {self.dino_root}")


@dataclass(frozen=True)
class Stage1Config:
    experiment_name: str = "dino_hc_normative_stage1"
    ablation: str = "base"
    seed: int = 2026
    fold: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig | None = None
    _validated: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        # Coerce dict sections into their dataclass types (direct construction
        # via Stage1Config(**cfg.to_dict()) passes plain dicts).
        for fname, cls_ in [
            ("model", ModelConfig), ("masking", MaskingConfig), ("loss", LossConfig),
            ("sampler", SamplerConfig), ("optimization", OptimizationConfig),
            ("validation", ValidationConfig), ("runtime", RuntimeConfig),
        ]:
            value = getattr(self, fname)
            if isinstance(value, dict):
                object.__setattr__(self, fname, cls_(**value))
        if isinstance(self.paths, dict):
            object.__setattr__(self, "paths", PathsConfig(**self.paths))
        if not self._validated:
            return
        if not 0 <= self.seed <= 2**32 - 1:
            raise ConfigError("seed must be in [0, 2**32-1]")
        if not 0 <= self.fold <= 4:
            raise ConfigError("fold must be in [0, 4]")
        for section, where in [
            (self.model, "model"), (self.masking, "masking"), (self.loss, "loss"),
            (self.sampler, "sampler"), (self.optimization, "optimization"),
            (self.validation, "validation"), (self.runtime, "runtime"),
            (self.paths, "paths"),
        ]:
            if section is not None:
                section.validate(where)
        if len(self.loss.channel_weights) != self.model.input_channels:
            raise ConfigError(
                f"loss.channel_weights has {len(self.loss.channel_weights)} entries "
                f"but model.input_channels={self.model.input_channels}"
            )

    @property
    def d_model(self) -> int:
        return self.model.d_model

    @property
    def gamma_init(self) -> float:
        return self.model.gamma_init

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "Stage1Config":
        known = {
            "experiment_name", "ablation", "seed", "fold", "model", "masking", "loss",
            "sampler", "optimization", "validation", "runtime", "paths",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{source}: unknown config fields: {unknown}")

        def section(name: str, cls_: type, defaults: dict[str, Any]) -> Any:
            data = dict(defaults)
            if name in raw and raw[name] is not None:
                if not isinstance(raw[name], dict):
                    raise ConfigError(f"{source}: {name} must be a mapping")
                extra = set(raw[name]) - set(defaults)
                if extra:
                    raise ConfigError(f"{source}: unknown {name} fields: {sorted(extra)}")
                data.update(raw[name])
            return cls_(**data)

        model = section(
            "model", ModelConfig, asdict(ModelConfig())
        )
        masking = section("masking", MaskingConfig, asdict(MaskingConfig()))
        loss_cfg = section("loss", LossConfig, asdict(LossConfig()))
        sampler = section("sampler", SamplerConfig, asdict(SamplerConfig()))
        optimization = section("optimization", OptimizationConfig, asdict(OptimizationConfig()))
        validation = section("validation", ValidationConfig, asdict(ValidationConfig()))
        runtime = section("runtime", RuntimeConfig, asdict(RuntimeConfig()))

        paths = None
        if raw.get("paths"):
            p = raw["paths"]
            if not isinstance(p, dict):
                raise ConfigError(f"{source}: paths must be a mapping")
            paths = PathsConfig(
                processed_root=Path(p["processed_root"]),
                dino_root=Path(p["dino_root"]),
                cv_fold_dir=Path(p["cv_fold_dir"]) if p.get("cv_fold_dir") else None,
                output_root=Path(p["output_root"]) if p.get("output_root") else None,
            )

        return cls(
            experiment_name=str(raw.get("experiment_name", "dino_hc_normative_stage1")),
            ablation=str(raw.get("ablation", "base")),
            seed=int(raw.get("seed", 2026)),
            fold=int(raw.get("fold", 0)),
            model=model,
            masking=masking,
            loss=loss_cfg,
            sampler=sampler,
            optimization=optimization,
            validation=validation,
            runtime=runtime,
            paths=paths,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Stage1Config":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: configuration must be a mapping")
        return cls.from_dict(raw, source=str(p))

    @classmethod
    def load_base_with_ablation(
        cls, base_path: Path | str, ablation_name: str | None
    ) -> "Stage1Config":
        """Load ``base.yaml`` merged with ``ablations/<name>.yaml`` (deep merge).

        Ablation overlay files contain only the fields they change; the
        resolved configuration always carries the full merged content.
        """
        base = Path(base_path)
        if not base.is_file():
            raise ConfigError(f"base config not found: {base}")
        with open(base, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{base}: configuration must be a mapping")
        if ablation_name and ablation_name != "base":
            overlay_path = base.parent / "ablations" / f"{ablation_name}.yaml"
            if not overlay_path.is_file():
                raise ConfigError(f"ablation overlay not found: {overlay_path}")
            with open(overlay_path, "r", encoding="utf-8") as fh:
                overlay = yaml.safe_load(fh)
            if not isinstance(overlay, dict):
                raise ConfigError(f"{overlay_path}: overlay must be a mapping")
            overlay.pop("_comment", None)
            _deep_merge(raw, overlay)
            raw["ablation"] = ablation_name
        return cls.from_dict(raw, source=str(base))

    @classmethod
    def load_ablation_overlay(cls, overlay_path: Path | str) -> dict[str, Any]:
        p = Path(overlay_path)
        with open(p, "r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh)
        overlay.pop("_comment", None)
        return overlay

    def to_dict(self) -> dict[str, Any]:
        def conv(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: conv(v) for k, v in asdict(value).items()}
            return value

        return {
            "experiment_name": self.experiment_name,
            "ablation": self.ablation,
            "seed": self.seed,
            "fold": self.fold,
            "model": conv(self.model),
            "masking": conv(self.masking),
            "loss": conv(self.loss),
            "sampler": conv(self.sampler),
            "optimization": conv(self.optimization),
            "validation": conv(self.validation),
            "runtime": conv(self.runtime),
            "paths": conv(self.paths) if self.paths else None,
        }

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Recursively merge ``overlay`` into ``base`` (in place)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
