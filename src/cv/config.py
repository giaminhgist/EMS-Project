"""Typed, strictly validated cross-validation configuration."""

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
class CVConfig:
    n_splits: int
    shuffle: bool
    random_state: int
    stratify_column: str
    group_column: str
    subject_manifest: Path
    trial_manifest: Path
    source_inventory: Path
    output_root: Path
    _validated: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        for f in ("subject_manifest", "trial_manifest", "source_inventory", "output_root"):
            value = getattr(self, f)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, f, Path(value))
        if not self._validated:
            return
        if self.n_splits < 2:
            raise ConfigError("n_splits must be >= 2")
        if not 0 <= self.random_state <= 2**32 - 1:
            raise ConfigError("random_state must be in [0, 2**32-1]")
        if self.stratify_column != "label":
            raise ConfigError("stratify_column must be 'label'")
        if self.group_column != "subject_id":
            raise ConfigError("group_column must be 'subject_id'")

    @property
    def versioned_dir_name(self) -> str:
        """Versioned directory encodes the split identity (splits + seed)."""
        return f"{self.n_splits}fold_seed{self.random_state}"

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.versioned_dir_name

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "CVConfig":
        known = {
            "n_splits", "shuffle", "random_state", "stratify_column", "group_column",
            "subject_manifest", "trial_manifest", "source_inventory", "output_root",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{source}: unknown config fields: {unknown}")
        missing = sorted(known - set(raw))
        if missing:
            raise ConfigError(f"{source}: missing required fields: {missing}")
        return cls(
            n_splits=int(raw["n_splits"]),
            shuffle=bool(raw["shuffle"]),
            random_state=int(raw["random_state"]),
            stratify_column=str(raw["stratify_column"]),
            group_column=str(raw["group_column"]),
            subject_manifest=Path(raw["subject_manifest"]),
            trial_manifest=Path(raw["trial_manifest"]),
            source_inventory=Path(raw["source_inventory"]),
            output_root=Path(raw["output_root"]),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CVConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: configuration must be a mapping")
        return cls.from_dict(raw, source=str(p))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_validated")
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
