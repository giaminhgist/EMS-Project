"""Typed, strictly validated preprocessing configuration.

The schema accepts exactly the fields of ``configs/preprocessing.yaml`` plus
their resolved defaults. Unknown fields and invalid values are rejected. The
resolved configuration is written to ``preprocessing_config.json`` inside the
output root as the canonical record of how the dataset was produced.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_SUPPORTED_OFF_CANVAS_POLICIES = {"drop"}
_SUPPORTED_ZERO_SPATIAL_POLICIES = {"exclude_no_spatial"}
_SUPPORTED_LABEL_RULES = {"numeric_id_below_split_is_hc"}
_SUPPORTED_DTYPES = {"float32"}

# Field metadata: (required, default, expected type, validator name)
_SCHEMA: dict[str, dict[str, Any]] = {
    "raw_root": {"type": "path", "required": True},
    "fixation_root": {"type": "path", "required": True},
    "image_root": {"type": "path", "required": True},
    "output_root": {"type": "path", "required": True},
    "subject_glob": {"type": "str", "required": False, "default": "*.xlsx"},
    "subject_filename_regex": {"type": "str", "required": False, "default": r"^[0-9]+[.]xlsx$"},
    "sheet_name": {"type": "str", "required": False, "default": "Free_viewing"},
    "columns": {"type": "columns", "required": False, "default": None},
    "label_rule": {"type": "str", "required": False, "default": "numeric_id_below_split_is_hc"},
    "hc_sz_split": {"type": "int", "required": False, "default": 200},
    "source_width": {"type": "int", "required": False, "default": 1024},
    "source_height": {"type": "int", "required": False, "default": 768},
    "heatmap_height": {"type": "int", "required": False, "default": 48},
    "heatmap_width": {"type": "int", "required": False, "default": 64},
    "gaussian_sigma_cells": {"type": "float", "required": False, "default": 2.0},
    "gaussian_truncate_sigma": {"type": "float", "required": False, "default": 4.0},
    "transition_sample_step_cells": {"type": "float", "required": False, "default": 0.5},
    "temporal_epsilon": {"type": "float", "required": False, "default": 1.0e-8},
    "off_canvas_policy": {"type": "str", "required": False, "default": "drop"},
    "zero_spatial_policy": {"type": "str", "required": False, "default": "exclude_no_spatial"},
    "min_fix_duration_ms": {"type": "float", "required": False, "default": 0},
    "drop_nonpositive_duration": {"type": "bool", "required": False, "default": True},
    "dtype": {"type": "str", "required": False, "default": "float32"},
    "num_workers": {"type": "int", "required": False, "default": 0},
    "seed": {"type": "int", "required": False, "default": 2026},
    "write_trial_manifest_csv": {"type": "bool", "required": False, "default": True},
}

_DEFAULT_COLUMNS = {
    "image": "IMAGE",
    "fix_index": "FIX_INDEX",
    "fix_duration": "FIX_DURATION",
    "fix_x": "FIX_X",
    "fix_y": "FIX_Y",
    "fix_pupil": "FIX_PUPIL",
}


class ConfigError(ValueError):
    """Raised when a configuration file is invalid or unsupported."""


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"config field {name!r} must be a boolean, got {value!r}")


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"config field {name!r} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    raise ConfigError(f"config field {name!r} must be an integer, got {value!r}")


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"config field {name!r} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    raise ConfigError(f"config field {name!r} must be a number, got {value!r}")


def _coerce_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"config field {name!r} must be a non-empty path string")
    return Path(str(value))


@dataclass(frozen=True)
class PreprocessingConfig:
    raw_root: Path
    fixation_root: Path
    image_root: Path
    output_root: Path
    subject_glob: str
    subject_filename_regex: str
    sheet_name: str
    columns: dict[str, str]
    label_rule: str
    hc_sz_split: int
    source_width: int
    source_height: int
    heatmap_height: int
    heatmap_width: int
    gaussian_sigma_cells: float
    gaussian_truncate_sigma: float
    transition_sample_step_cells: float
    temporal_epsilon: float
    off_canvas_policy: str
    zero_spatial_policy: str
    min_fix_duration_ms: float
    drop_nonpositive_duration: bool
    dtype: str
    num_workers: int
    seed: int
    write_trial_manifest_csv: bool
    # Resolved artifacts (not user-supplied):
    _validated: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if not self._validated:
            return  # rehydration from a previously resolved dict
        if self.hc_sz_split <= 0:
            raise ConfigError("hc_sz_split must be positive")
        if self.label_rule not in _SUPPORTED_LABEL_RULES:
            raise ConfigError(f"unsupported label_rule {self.label_rule!r}")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ConfigError("source dimensions must be positive")
        if self.heatmap_height <= 0 or self.heatmap_width <= 0:
            raise ConfigError("heatmap dimensions must be positive")
        if not self.gaussian_sigma_cells > 0:
            raise ConfigError("gaussian_sigma_cells must be positive")
        if not self.gaussian_truncate_sigma > 0:
            raise ConfigError("gaussian_truncate_sigma must be positive")
        if not self.transition_sample_step_cells > 0:
            raise ConfigError("transition_sample_step_cells must be positive")
        if not self.temporal_epsilon > 0:
            raise ConfigError("temporal_epsilon must be positive")
        if self.off_canvas_policy not in _SUPPORTED_OFF_CANVAS_POLICIES:
            raise ConfigError(
                f"off_canvas_policy must be one of {sorted(_SUPPORTED_OFF_CANVAS_POLICIES)}"
            )
        if self.zero_spatial_policy not in _SUPPORTED_ZERO_SPATIAL_POLICIES:
            raise ConfigError(
                f"zero_spatial_policy must be one of {sorted(_SUPPORTED_ZERO_SPATIAL_POLICIES)}"
            )
        if self.min_fix_duration_ms < 0:
            raise ConfigError("min_fix_duration_ms must be >= 0")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ConfigError(f"unsupported dtype {self.dtype!r}")
        if self.num_workers < 0:
            raise ConfigError("num_workers must be >= 0")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ConfigError("seed must be in [0, 2**32-1]")
        if self.sheet_name.strip() == "":
            raise ConfigError("sheet_name must be non-empty")
        try:
            re.compile(self.subject_filename_regex)
        except re.error as exc:
            raise ConfigError(f"subject_filename_regex is invalid: {exc}") from exc
        required_cols = set(_DEFAULT_COLUMNS)
        if set(self.columns) != required_cols:
            raise ConfigError(
                f"columns must define exactly {sorted(required_cols)}, got {sorted(self.columns)}"
            )
        if len(set(self.columns.values())) != len(self.columns):
            raise ConfigError("column names must be distinct")
        if not self.drop_nonpositive_duration and self.min_fix_duration_ms > 0:
            raise ConfigError(
                "min_fix_duration_ms > 0 requires drop_nonpositive_duration=true"
            )

    # ------------------------------------------------------------------ load

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "PreprocessingConfig":
        unknown = sorted(set(raw) - set(_SCHEMA))
        if unknown:
            raise ConfigError(f"{source}: unknown config fields: {unknown}")
        missing = [k for k, meta in _SCHEMA.items() if meta["required"] and k not in raw]
        if missing:
            raise ConfigError(f"{source}: missing required fields: {missing}")

        def value(name: str) -> Any:
            meta = _SCHEMA[name]
            if name not in raw or raw[name] is None:
                if meta.get("default") is None and name == "columns":
                    return dict(_DEFAULT_COLUMNS)
                if not meta["required"]:
                    return meta["default"]
                raise ConfigError(f"{source}: field {name!r} is required")
            return raw[name]

        kind = lambda name: _SCHEMA[name]["type"]
        data: dict[str, Any] = {}
        for name in _SCHEMA:
            v = value(name)
            k = kind(name)
            if k == "path":
                data[name] = _coerce_path(v, name)
            elif k == "str":
                if not isinstance(v, str):
                    raise ConfigError(f"config field {name!r} must be a string, got {v!r}")
                data[name] = v
            elif k == "int":
                data[name] = _coerce_int(v, name)
            elif k == "float":
                data[name] = _coerce_float(v, name)
            elif k == "bool":
                data[name] = _coerce_bool(v, name)
            elif k == "columns":
                if not isinstance(v, dict) or not all(
                    isinstance(kk, str) and isinstance(vv, str) for kk, vv in v.items()
                ):
                    raise ConfigError(f"config field {name!r} must map strings to strings")
                data[name] = dict(v)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PreprocessingConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: configuration must be a mapping")
        return cls.from_dict(raw, source=str(p))

    # ----------------------------------------------------------------- write

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_validated")
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
        return d

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
