"""Typed validated Stage-2 configuration (Phases 2-4).

Data, model, loss, subset, optimization and validation sections are validated
here with strict types (the string ``"false"`` is rejected, not coerced) and
unknown fields are rejected so overlays cannot silently inject options the
runner ignores. ``load_yaml_dict`` rejects duplicate keys, and ``config_hash``
is invariant to YAML key order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .contracts import EVALUATION_REGIMES


class ConfigError(ValueError):
    pass


# ------------------------------------------------------------------ strict types


def _strict_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(
        f"{where}: expected boolean, got {value!r} (type {type(value).__name__})"
    )


def _strict_int(value: Any, where: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ConfigError(
        f"{where}: expected integer, got {value!r} (type {type(value).__name__})"
    )


def _strict_float(value: Any, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ConfigError(
        f"{where}: expected number, got {value!r} (type {type(value).__name__})"
    )


def _strict_str(value: Any, where: str) -> str:
    if isinstance(value, str):
        return value
    raise ConfigError(
        f"{where}: expected string, got {value!r} (type {type(value).__name__})"
    )


def coerce_cli_value(value: str) -> Any:
    """Parse a command-line override value with YAML scalar semantics.

    ``"0" -> 0``, ``"2.5" -> 2.5``, ``"true" -> True``, ``"1e-4" -> 0.0001``;
    anything else stays a string. Strict type rejection remains in effect for
    YAML *files* (where the quoted string ``"false"`` must not become truthy).
    """
    parsed = yaml.safe_load(value)
    if parsed is None and value != "":
        return value
    return parsed


# --------------------------------------------------------------- YAML loading


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping_unique(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_unique
)


def load_yaml_dict(path: Path | str) -> dict[str, Any]:
    """Load one YAML mapping file, rejecting duplicate keys."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    raw = yaml.load(p.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: configuration must be a mapping")
    return raw


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive copy-merge; the base dict is never mutated."""
    out: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            out[key] = deep_merge(value, {})
        else:
            out[key] = value
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_hash(raw: dict[str, Any]) -> str:
    """Canonical SHA-256 of the resolved configuration dict.

    ``yaml.safe_dump(sort_keys=True)`` makes the hash invariant to YAML key
    order; paths are included so a resolved config is fully identified.
    """
    canonical = yaml.safe_dump(raw, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -------------------------------------------------------------- typed sections


@dataclass(frozen=True)
class BankSectionConfig:
    root: Path | None = None  # resolved normative bank root
    train_mode: str = "crossfit"  # crossfit | full_self_included
    verify_checksums: bool = True
    checkpoint_registry: Path | None = None
    require_token_banks: bool = False  # fused token banks must exist in the bank
    require_heatmap_token_banks: bool = False  # heat-token banks (not Phase-0 approved)

    def validate(self, where: str) -> None:
        if self.train_mode not in ("crossfit", "full_self_included"):
            raise ConfigError(f"{where}: bank.train_mode must be 'crossfit' or 'full_self_included'")
        if self.root is None:
            raise ConfigError(f"{where}: bank.root is required")
        if self.require_heatmap_token_banks:
            raise ConfigError(
                f"{where}: bank.require_heatmap_token_bank is not runnable: heatmap-token "
                f"banks were not built (Phase 0 approved trial + fused-token banks only)"
            )


@dataclass(frozen=True)
class SamplerSectionConfig:
    subject_batch_size: int = 4
    balance_groups: bool = True
    drop_last: bool = False

    def validate(self, where: str) -> None:
        if self.subject_batch_size <= 0:
            raise ConfigError(f"{where}: sampler.subject_batch_size must be positive")
        if self.balance_groups and self.subject_batch_size % 2 != 0:
            raise ConfigError(
                f"{where}: balanced subject batching requires an even subject_batch_size, "
                f"got {self.subject_batch_size}"
            )


@dataclass(frozen=True)
class RuntimeSectionConfig:
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    deterministic_validation: bool = True

    def validate(self, where: str) -> None:
        if self.num_workers < 0:
            raise ConfigError(f"{where}: runtime.num_workers must be >= 0")
        if self.persistent_workers and self.num_workers == 0:
            raise ConfigError(
                f"{where}: runtime.persistent_workers=true requires runtime.num_workers > 0"
            )


@dataclass(frozen=True)
class ModelSectionConfig:
    d_model: int = 128
    encoder_source: str = "stage1_heatmap_encoder"
    freeze_encoder: bool = True
    encoder_random_init: bool = False  # random_encoder ablation
    encoder_unfreeze_last_block: bool = False  # unfreeze_last_block ablation
    query_pooling: str = "attention"  # attention | mean
    relation_hidden: int = 256
    bank_mode: str = "trial"  # trial | trial_and_fused_token
    bank_features_active: bool = True  # no_bank ablation: neutralized when false
    wrong_bank_permutation: bool = False  # wrong_stimulus_bank ablation
    global_bank: bool = False  # global_bank ablation
    attention_heads: int = 4
    token_local_window: int = 3
    token_attention_layers: int = 2  # 1 | 2
    token_spatial_bridge: str = "residual_dwconv_ffn"  # residual_dwconv_ffn | identity
    category_balanced_attention: bool = True  # no_category_balance ablation
    subject_transformer_layers: int = 1  # 0 | 1 (mean_subject_pooling ablation)
    subject_transformer_ffn: int = 256
    dropout: float = 0.25
    auxiliary_evidence_head: bool = True

    def validate(self, where: str) -> None:
        if self.d_model != 128:
            raise ConfigError(f"{where}: model.d_model must be 128 (stable contract)")
        if self.encoder_source != "stage1_heatmap_encoder":
            raise ConfigError(f"{where}: model.encoder_source must be 'stage1_heatmap_encoder'")
        if self.encoder_unfreeze_last_block and not self.freeze_encoder:
            raise ConfigError(
                f"{where}: model.encoder_unfreeze_last_block requires model.freeze_encoder"
            )
        if self.encoder_unfreeze_last_block and self.encoder_random_init:
            raise ConfigError(
                f"{where}: encoder_unfreeze_last_block and encoder_random_init are "
                f"incompatible (anchor requires the Stage-1 reference)"
            )
        if self.query_pooling not in ("attention", "mean"):
            raise ConfigError(
                f"{where}: model.query_pooling must be 'attention' or 'mean'"
            )
        if self.bank_mode not in ("trial", "trial_and_fused_token"):
            raise ConfigError(
                f"{where}: model.bank_mode must be 'trial' or 'trial_and_fused_token' "
                f"(same-space heat-token banks were not built in Phase 0)"
            )
        if self.bank_mode == "trial_and_fused_token" and not self.bank_features_active:
            raise ConfigError(
                f"{where}: a token bank mode requires model.bank_features_active"
            )
        if self.attention_heads != 4:
            raise ConfigError(f"{where}: model.attention_heads must be 4 (stable contract)")
        if self.token_local_window != 3:
            raise ConfigError(f"{where}: model.token_local_window must be 3")
        if self.token_attention_layers not in (1, 2):
            raise ConfigError(f"{where}: model.token_attention_layers must be 1 or 2")
        if self.token_spatial_bridge not in ("residual_dwconv_ffn", "identity"):
            raise ConfigError(
                f"{where}: model.token_spatial_bridge must be 'residual_dwconv_ffn' or 'identity'"
            )
        if self.subject_transformer_layers not in (0, 1):
            raise ConfigError(f"{where}: model.subject_transformer_layers must be 0 or 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"{where}: model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class LossSectionConfig:
    lambda_aux: float = 0.3
    lambda_match: float = 0.1
    lambda_cons: float = 0.1
    lambda_entropy: float = 0.01
    entropy_anneal_epochs: int = 10
    lambda_anchor: float = 0.0
    match_margin: float = 0.2
    bank_rank_margin: float = 0.2
    token_match_margin: float = 0.2
    entropy_floor: float = 0.5

    def validate(self, where: str) -> None:
        for name, value in (
            ("lambda_aux", self.lambda_aux),
            ("lambda_match", self.lambda_match),
            ("lambda_cons", self.lambda_cons),
            ("lambda_entropy", self.lambda_entropy),
            ("lambda_anchor", self.lambda_anchor),
        ):
            if value < 0.0:
                raise ConfigError(f"{where}: loss.{name} must be >= 0")
        if self.entropy_anneal_epochs < 0:
            raise ConfigError(f"{where}: loss.entropy_anneal_epochs must be >= 0")
        if self.entropy_floor <= 0.0:
            raise ConfigError(f"{where}: loss.entropy_floor must be > 0")


@dataclass(frozen=True)
class SubsetsSectionConfig:
    enabled: bool = True
    min_fraction: float = 0.5
    max_fraction: float = 0.8
    category_stratified: bool = True

    def validate(self, where: str) -> None:
        if not 0.0 < self.min_fraction < self.max_fraction <= 1.0:
            raise ConfigError(
                f"{where}: subsets must satisfy 0 < min_fraction < max_fraction <= 1"
            )
        if not self.category_stratified:
            raise ConfigError(f"{where}: subsets.category_stratified must be true")


@dataclass(frozen=True)
class OptimizationSectionConfig:
    alignment_epochs: int = 10
    classification_epochs: int = 50
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-4
    encoder_learning_rate: float = 1.0e-5
    weight_decay: float = 5.0e-4
    scheduler: str = "linear_warmup_cosine"
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    amp: bool = True

    def validate(self, where: str) -> None:
        if self.alignment_epochs < 0 or self.classification_epochs < 1:
            raise ConfigError(f"{where}: optimization epoch counts invalid")
        if self.optimizer != "adamw":
            raise ConfigError(f"{where}: optimization.optimizer must be 'adamw'")
        if self.scheduler != "linear_warmup_cosine":
            raise ConfigError(f"{where}: optimization.scheduler must be 'linear_warmup_cosine'")
        if self.learning_rate <= 0 or self.encoder_learning_rate <= 0:
            raise ConfigError(f"{where}: learning rates must be positive")
        if self.encoder_learning_rate >= self.learning_rate:
            raise ConfigError(
                f"{where}: encoder_learning_rate must be below learning_rate (Stage 2C)"
            )
        if self.gradient_clip_norm <= 0:
            raise ConfigError(f"{where}: gradient_clip_norm must be positive")


@dataclass(frozen=True)
class ValidationSectionConfig:
    selection_metric: str = "val_balanced_accuracy"
    secondary_metric: str = "val_auroc"
    early_stopping_patience: int = 10
    calibrate: bool = True

    def validate(self, where: str) -> None:
        if self.selection_metric not in ("val_balanced_accuracy", "val_auroc", "val_loss"):
            raise ConfigError(f"{where}: unsupported selection_metric {self.selection_metric!r}")
        if self.early_stopping_patience < 1:
            raise ConfigError(f"{where}: early_stopping_patience must be >= 1")


@dataclass(frozen=True)
class PathsSectionConfig:
    processed_root: Path
    cv_root: Path
    normative_bank_root: Path
    output_root: Path | None = None  # resolved Stage-2 output root (defaults in runner)

    def validate(self, where: str) -> None:
        if not self.processed_root.is_dir():
            raise ConfigError(f"{where}: processed_root not found: {self.processed_root}")
        if not self.cv_root.is_dir():
            raise ConfigError(f"{where}: cv_root not found: {self.cv_root}")
        if not self.normative_bank_root.is_dir():
            raise ConfigError(f"{where}: normative_bank_root not found: {self.normative_bank_root}")


# --------------------------------------------------------------------- top level


@dataclass(frozen=True)
class Stage2Config:
    seed: int = 2026
    fold: int = 0
    experiment_name: str = "hc_normative_stage2"
    ablation: str = "base"
    evaluation_regime: str = "pilot_existing_stage1"
    bank: BankSectionConfig = field(default_factory=BankSectionConfig)
    sampler: SamplerSectionConfig = field(default_factory=SamplerSectionConfig)
    runtime: RuntimeSectionConfig = field(default_factory=RuntimeSectionConfig)
    model: ModelSectionConfig = field(default_factory=ModelSectionConfig)
    loss: LossSectionConfig = field(default_factory=LossSectionConfig)
    subsets: SubsetsSectionConfig = field(default_factory=SubsetsSectionConfig)
    optimization: OptimizationSectionConfig = field(default_factory=OptimizationSectionConfig)
    validation: ValidationSectionConfig = field(default_factory=ValidationSectionConfig)
    paths: PathsSectionConfig | None = None

    def __post_init__(self) -> None:
        if self.paths is not None:
            object.__setattr__(self, "bank", _resolve_bank_root(self.bank, self.paths))

    def validate(self) -> None:
        if not 0 <= self.seed <= 2**32 - 1:
            raise ConfigError("seed must be in [0, 2**32-1]")
        if not 0 <= self.fold <= 4:
            raise ConfigError("fold must be in [0, 4]")
        if self.evaluation_regime not in EVALUATION_REGIMES:
            raise ConfigError(
                f"evaluation_regime must be one of {EVALUATION_REGIMES}, "
                f"got {self.evaluation_regime!r}"
            )
        self.bank.validate("bank")
        self.sampler.validate("sampler")
        self.runtime.validate("runtime")
        self.model.validate("model")
        self.loss.validate("loss")
        self.subsets.validate("subsets")
        self.optimization.validate("optimization")
        self.validation.validate("validation")
        if self.paths is None:
            raise ConfigError("paths are required")
        self.paths.validate("paths")

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "Stage2Config":
        known = {
            "seed", "fold", "experiment_name", "ablation", "evaluation_regime",
            "bank", "sampler", "runtime", "model", "loss", "subsets",
            "optimization", "validation", "paths",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{source}: unknown config fields: {unknown}")

        bank_raw = dict(raw.get("bank") or {})
        sampler_raw = dict(raw.get("sampler") or {})
        runtime_raw = dict(raw.get("runtime") or {})
        model_raw = dict(raw.get("model") or {})
        loss_raw = dict(raw.get("loss") or {})
        subsets_raw = dict(raw.get("subsets") or {})
        optimization_raw = dict(raw.get("optimization") or {})
        validation_raw = dict(raw.get("validation") or {})
        paths_raw = raw.get("paths")

        extra = set(bank_raw) - {
            "root", "train_mode", "verify_checksums", "checkpoint_registry",
            "require_fused_token_bank", "require_heatmap_token_bank",
        }
        if extra:
            raise ConfigError(f"{source}: unknown bank fields: {sorted(extra)}")
        extra = set(sampler_raw) - set(SamplerSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown sampler fields: {sorted(extra)}")
        extra = set(runtime_raw) - set(RuntimeSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown runtime fields: {sorted(extra)}")
        extra = set(model_raw) - set(ModelSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown model fields: {sorted(extra)}")
        extra = set(loss_raw) - set(LossSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown loss fields: {sorted(extra)}")
        extra = set(subsets_raw) - set(SubsetsSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown subsets fields: {sorted(extra)}")
        extra = set(optimization_raw) - set(OptimizationSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown optimization fields: {sorted(extra)}")
        extra = set(validation_raw) - set(ValidationSectionConfig.__dataclass_fields__)
        if extra:
            raise ConfigError(f"{source}: unknown validation fields: {sorted(extra)}")

        bank = BankSectionConfig(
            root=Path(bank_raw["root"]) if bank_raw.get("root") else None,
            train_mode=_strict_str(bank_raw.get("train_mode", "crossfit"), f"{source}: bank.train_mode"),
            verify_checksums=_strict_bool(
                bank_raw.get("verify_checksums", True), f"{source}: bank.verify_checksums"
            ),
            checkpoint_registry=(
                Path(bank_raw["checkpoint_registry"]) if bank_raw.get("checkpoint_registry") else None
            ),
            require_token_banks=_strict_bool(
                bank_raw.get("require_fused_token_bank", False),
                f"{source}: bank.require_fused_token_bank",
            ),
            require_heatmap_token_banks=_strict_bool(
                bank_raw.get("require_heatmap_token_bank", False),
                f"{source}: bank.require_heatmap_token_bank",
            ),
        )
        sampler = SamplerSectionConfig(
            subject_batch_size=_strict_int(
                sampler_raw.get("subject_batch_size", 4), f"{source}: sampler.subject_batch_size"
            ),
            balance_groups=_strict_bool(
                sampler_raw.get("balance_groups", True), f"{source}: sampler.balance_groups"
            ),
            drop_last=_strict_bool(sampler_raw.get("drop_last", False), f"{source}: sampler.drop_last"),
        )
        runtime = RuntimeSectionConfig(
            num_workers=_strict_int(
                runtime_raw.get("num_workers", 0), f"{source}: runtime.num_workers"
            ),
            pin_memory=_strict_bool(
                runtime_raw.get("pin_memory", True), f"{source}: runtime.pin_memory"
            ),
            persistent_workers=_strict_bool(
                runtime_raw.get("persistent_workers", False), f"{source}: runtime.persistent_workers"
            ),
            deterministic_validation=_strict_bool(
                runtime_raw.get("deterministic_validation", True),
                f"{source}: runtime.deterministic_validation",
            ),
        )
        model = ModelSectionConfig(
            d_model=_strict_int(model_raw.get("d_model", 128), f"{source}: model.d_model"),
            encoder_source=_strict_str(
                model_raw.get("encoder_source", "stage1_heatmap_encoder"),
                f"{source}: model.encoder_source",
            ),
            freeze_encoder=_strict_bool(
                model_raw.get("freeze_encoder", True), f"{source}: model.freeze_encoder"
            ),
            encoder_random_init=_strict_bool(
                model_raw.get("encoder_random_init", False), f"{source}: model.encoder_random_init"
            ),
            encoder_unfreeze_last_block=_strict_bool(
                model_raw.get("encoder_unfreeze_last_block", False),
                f"{source}: model.encoder_unfreeze_last_block",
            ),
            query_pooling=_strict_str(
                model_raw.get("query_pooling", "attention"), f"{source}: model.query_pooling"
            ),
            relation_hidden=_strict_int(
                model_raw.get("relation_hidden", 256), f"{source}: model.relation_hidden"
            ),
            bank_mode=_strict_str(
                model_raw.get("bank_mode", "trial"), f"{source}: model.bank_mode"
            ),
            bank_features_active=_strict_bool(
                model_raw.get("bank_features_active", True),
                f"{source}: model.bank_features_active",
            ),
            wrong_bank_permutation=_strict_bool(
                model_raw.get("wrong_bank_permutation", False),
                f"{source}: model.wrong_bank_permutation",
            ),
            global_bank=_strict_bool(
                model_raw.get("global_bank", False), f"{source}: model.global_bank"
            ),
            attention_heads=_strict_int(
                model_raw.get("attention_heads", 4), f"{source}: model.attention_heads"
            ),
            token_local_window=_strict_int(
                model_raw.get("token_local_window", 3), f"{source}: model.token_local_window"
            ),
            token_attention_layers=_strict_int(
                model_raw.get("token_attention_layers", 2),
                f"{source}: model.token_attention_layers",
            ),
            token_spatial_bridge=_strict_str(
                model_raw.get("token_spatial_bridge", "residual_dwconv_ffn"),
                f"{source}: model.token_spatial_bridge",
            ),
            category_balanced_attention=_strict_bool(
                model_raw.get("category_balanced_attention", True),
                f"{source}: model.category_balanced_attention",
            ),
            subject_transformer_layers=_strict_int(
                model_raw.get("subject_transformer_layers", 1),
                f"{source}: model.subject_transformer_layers",
            ),
            subject_transformer_ffn=_strict_int(
                model_raw.get("subject_transformer_ffn", 256),
                f"{source}: model.subject_transformer_ffn",
            ),
            dropout=_strict_float(model_raw.get("dropout", 0.25), f"{source}: model.dropout"),
            auxiliary_evidence_head=_strict_bool(
                model_raw.get("auxiliary_evidence_head", True),
                f"{source}: model.auxiliary_evidence_head",
            ),
        )
        loss = LossSectionConfig(
            lambda_aux=_strict_float(loss_raw.get("lambda_aux", 0.3), f"{source}: loss.lambda_aux"),
            lambda_match=_strict_float(
                loss_raw.get("lambda_match", 0.1), f"{source}: loss.lambda_match"
            ),
            lambda_cons=_strict_float(loss_raw.get("lambda_cons", 0.1), f"{source}: loss.lambda_cons"),
            lambda_entropy=_strict_float(
                loss_raw.get("lambda_entropy", 0.01), f"{source}: loss.lambda_entropy"
            ),
            entropy_anneal_epochs=_strict_int(
                loss_raw.get("entropy_anneal_epochs", 10), f"{source}: loss.entropy_anneal_epochs"
            ),
            lambda_anchor=_strict_float(
                loss_raw.get("lambda_anchor", 0.0), f"{source}: loss.lambda_anchor"
            ),
            match_margin=_strict_float(
                loss_raw.get("match_margin", 0.2), f"{source}: loss.match_margin"
            ),
            bank_rank_margin=_strict_float(
                loss_raw.get("bank_rank_margin", 0.2), f"{source}: loss.bank_rank_margin"
            ),
            token_match_margin=_strict_float(
                loss_raw.get("token_match_margin", 0.2), f"{source}: loss.token_match_margin"
            ),
            entropy_floor=_strict_float(
                loss_raw.get("entropy_floor", 0.5), f"{source}: loss.entropy_floor"
            ),
        )
        subsets = SubsetsSectionConfig(
            enabled=_strict_bool(subsets_raw.get("enabled", True), f"{source}: subsets.enabled"),
            min_fraction=_strict_float(
                subsets_raw.get("min_fraction", 0.5), f"{source}: subsets.min_fraction"
            ),
            max_fraction=_strict_float(
                subsets_raw.get("max_fraction", 0.8), f"{source}: subsets.max_fraction"
            ),
            category_stratified=_strict_bool(
                subsets_raw.get("category_stratified", True),
                f"{source}: subsets.category_stratified",
            ),
        )
        optimization = OptimizationSectionConfig(
            alignment_epochs=_strict_int(
                optimization_raw.get("alignment_epochs", 10),
                f"{source}: optimization.alignment_epochs",
            ),
            classification_epochs=_strict_int(
                optimization_raw.get("classification_epochs", 50),
                f"{source}: optimization.classification_epochs",
            ),
            optimizer=_strict_str(
                optimization_raw.get("optimizer", "adamw"), f"{source}: optimization.optimizer"
            ),
            learning_rate=_strict_float(
                optimization_raw.get("learning_rate", 1.0e-4),
                f"{source}: optimization.learning_rate",
            ),
            encoder_learning_rate=_strict_float(
                optimization_raw.get("encoder_learning_rate", 1.0e-5),
                f"{source}: optimization.encoder_learning_rate",
            ),
            weight_decay=_strict_float(
                optimization_raw.get("weight_decay", 5.0e-4),
                f"{source}: optimization.weight_decay",
            ),
            scheduler=_strict_str(
                optimization_raw.get("scheduler", "linear_warmup_cosine"),
                f"{source}: optimization.scheduler",
            ),
            warmup_epochs=_strict_int(
                optimization_raw.get("warmup_epochs", 5), f"{source}: optimization.warmup_epochs"
            ),
            gradient_clip_norm=_strict_float(
                optimization_raw.get("gradient_clip_norm", 1.0),
                f"{source}: optimization.gradient_clip_norm",
            ),
            amp=_strict_bool(optimization_raw.get("amp", True), f"{source}: optimization.amp"),
        )
        validation = ValidationSectionConfig(
            selection_metric=_strict_str(
                validation_raw.get("selection_metric", "val_balanced_accuracy"),
                f"{source}: validation.selection_metric",
            ),
            secondary_metric=_strict_str(
                validation_raw.get("secondary_metric", "val_auroc"),
                f"{source}: validation.secondary_metric",
            ),
            early_stopping_patience=_strict_int(
                validation_raw.get("early_stopping_patience", 10),
                f"{source}: validation.early_stopping_patience",
            ),
            calibrate=_strict_bool(
                validation_raw.get("calibrate", True), f"{source}: validation.calibrate"
            ),
        )
        paths = None
        if paths_raw:
            if not isinstance(paths_raw, dict):
                raise ConfigError(f"{source}: paths must be a mapping")
            paths = PathsSectionConfig(
                processed_root=Path(paths_raw["processed_root"]),
                cv_root=Path(paths_raw["cv_root"]),
                normative_bank_root=Path(paths_raw["normative_bank_root"]),
                output_root=(
                    Path(paths_raw["output_root"]) if paths_raw.get("output_root") else None
                ),
            )
        cfg = cls(
            seed=_strict_int(raw.get("seed", 2026), f"{source}: seed"),
            fold=_strict_int(raw.get("fold", 0), f"{source}: fold"),
            experiment_name=_strict_str(
                raw.get("experiment_name", "hc_normative_stage2"),
                f"{source}: experiment_name",
            ),
            ablation=_strict_str(raw.get("ablation", "base"), f"{source}: ablation"),
            evaluation_regime=_strict_str(
                raw.get("evaluation_regime", "pilot_existing_stage1"),
                f"{source}: evaluation_regime",
            ),
            bank=bank,
            sampler=sampler,
            runtime=runtime,
            model=model,
            loss=loss,
            subsets=subsets,
            optimization=optimization,
            validation=validation,
            paths=paths,
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Stage2Config":
        return cls.from_dict(load_yaml_dict(path), source=str(path))

    def to_dict(self) -> dict[str, Any]:
        def _path(v: Any) -> Any:
            return str(v) if isinstance(v, Path) else v

        return {
            "seed": self.seed,
            "fold": self.fold,
            "experiment_name": self.experiment_name,
            "ablation": self.ablation,
            "evaluation_regime": self.evaluation_regime,
            "bank": {
                "root": _path(self.bank.root),
                "train_mode": self.bank.train_mode,
                "verify_checksums": self.bank.verify_checksums,
                "checkpoint_registry": _path(self.bank.checkpoint_registry),
                "require_fused_token_bank": self.bank.require_token_banks,
                "require_heatmap_token_bank": self.bank.require_heatmap_token_banks,
            },
            "sampler": {
                "subject_batch_size": self.sampler.subject_batch_size,
                "balance_groups": self.sampler.balance_groups,
                "drop_last": self.sampler.drop_last,
            },
            "runtime": {
                "num_workers": self.runtime.num_workers,
                "pin_memory": self.runtime.pin_memory,
                "persistent_workers": self.runtime.persistent_workers,
                "deterministic_validation": self.runtime.deterministic_validation,
            },
            "model": {
                "d_model": self.model.d_model,
                "encoder_source": self.model.encoder_source,
                "freeze_encoder": self.model.freeze_encoder,
                "encoder_random_init": self.model.encoder_random_init,
                "encoder_unfreeze_last_block": self.model.encoder_unfreeze_last_block,
                "query_pooling": self.model.query_pooling,
                "relation_hidden": self.model.relation_hidden,
                "bank_mode": self.model.bank_mode,
                "bank_features_active": self.model.bank_features_active,
                "wrong_bank_permutation": self.model.wrong_bank_permutation,
                "global_bank": self.model.global_bank,
                "attention_heads": self.model.attention_heads,
                "token_local_window": self.model.token_local_window,
                "token_attention_layers": self.model.token_attention_layers,
                "token_spatial_bridge": self.model.token_spatial_bridge,
                "category_balanced_attention": self.model.category_balanced_attention,
                "subject_transformer_layers": self.model.subject_transformer_layers,
                "subject_transformer_ffn": self.model.subject_transformer_ffn,
                "dropout": self.model.dropout,
                "auxiliary_evidence_head": self.model.auxiliary_evidence_head,
            },
            "loss": {
                "lambda_aux": self.loss.lambda_aux,
                "lambda_match": self.loss.lambda_match,
                "lambda_cons": self.loss.lambda_cons,
                "lambda_entropy": self.loss.lambda_entropy,
                "entropy_anneal_epochs": self.loss.entropy_anneal_epochs,
                "lambda_anchor": self.loss.lambda_anchor,
                "match_margin": self.loss.match_margin,
                "bank_rank_margin": self.loss.bank_rank_margin,
                "token_match_margin": self.loss.token_match_margin,
                "entropy_floor": self.loss.entropy_floor,
            },
            "subsets": {
                "enabled": self.subsets.enabled,
                "min_fraction": self.subsets.min_fraction,
                "max_fraction": self.subsets.max_fraction,
                "category_stratified": self.subsets.category_stratified,
            },
            "optimization": {
                "alignment_epochs": self.optimization.alignment_epochs,
                "classification_epochs": self.optimization.classification_epochs,
                "optimizer": self.optimization.optimizer,
                "learning_rate": self.optimization.learning_rate,
                "encoder_learning_rate": self.optimization.encoder_learning_rate,
                "weight_decay": self.optimization.weight_decay,
                "scheduler": self.optimization.scheduler,
                "warmup_epochs": self.optimization.warmup_epochs,
                "gradient_clip_norm": self.optimization.gradient_clip_norm,
                "amp": self.optimization.amp,
            },
            "validation": {
                "selection_metric": self.validation.selection_metric,
                "secondary_metric": self.validation.secondary_metric,
                "early_stopping_patience": self.validation.early_stopping_patience,
                "calibrate": self.validation.calibrate,
            },
            "paths": (
                {
                    "processed_root": _path(self.paths.processed_root),
                    "cv_root": _path(self.paths.cv_root),
                    "normative_bank_root": _path(self.paths.normative_bank_root),
                    "output_root": _path(self.paths.output_root),
                }
                if self.paths
                else None
            ),
        }


def _resolve_bank_root(bank: BankSectionConfig, paths: PathsSectionConfig) -> BankSectionConfig:
    """``bank.root`` wins; otherwise fall back to ``paths.normative_bank_root``."""
    root = bank.root if bank.root is not None else paths.normative_bank_root
    return BankSectionConfig(
        root=root,
        train_mode=bank.train_mode,
        verify_checksums=bank.verify_checksums,
        checkpoint_registry=bank.checkpoint_registry,
        require_token_banks=bank.require_token_banks,
        require_heatmap_token_banks=bank.require_heatmap_token_banks,
    )
