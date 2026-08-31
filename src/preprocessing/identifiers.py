"""Identifier contracts for EMS preprocessing.

Rules from ``01_DATA_MODEL_CONTRACTS.md``:

- ``subject_id`` is a canonical string taken from the workbook stem, preserving
  leading zeros (``"000"``). It must never be converted to an array offset.
- ``subject_numeric_id`` is a separate integer used only for validation and
  initial label derivation.
- ``stimulus_id`` is the exact workbook ``IMAGE`` value when it uniquely
  identifies one disk image.
- ``stimulus_index`` is an explicit contiguous integer ``0..N-1`` created only
  through the image manifest.
- The trial key is ``(subject_id, stimulus_id)``; ``trial_uid`` is derived with
  SHA-256 (never Python's process-randomized ``hash()``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def canonical_subject_id(workbook_stem: str) -> str:
    """Return the canonical subject ID string, preserving leading zeros."""
    if not isinstance(workbook_stem, str) or not workbook_stem:
        raise ValueError(f"invalid workbook stem: {workbook_stem!r}")
    return workbook_stem


def subject_numeric_id(workbook_stem: str) -> int:
    """Derive the validation-only numeric subject ID."""
    sid = canonical_subject_id(workbook_stem)
    try:
        return int(sid)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"workbook stem {sid!r} is not numeric; cannot derive subject_numeric_id"
        ) from exc


def resolve_group_label(subject_numeric_id: int, split: int = 200) -> tuple[str, int]:
    """Apply the dataset README label rule: < split is HC, >= split is SZ.

    Returns ``(group, label)`` with ``label=0`` for HC and ``label=1`` for SZ.
    The result is stored explicitly in the subject manifest; downstream code
    must consume these fields and never re-derive diagnosis from the ID.
    """
    if subject_numeric_id < split:
        return ("HC", 0)
    return ("SZ", 1)


def trial_uid(subject_id: str, stimulus_id: str) -> str:
    """Deterministic trial key: first 20 hex chars of SHA256(subject \\0 stimulus)."""
    digest = hashlib.sha256(f"{subject_id}\0{stimulus_id}".encode("utf-8")).hexdigest()
    return digest[:20]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SubjectIdentity:
    subject_id: str
    subject_numeric_id: int
    group: str
    label: int
    source_workbook: str

    @classmethod
    def from_stem(cls, stem: str, split: int = 200) -> "SubjectIdentity":
        numeric = subject_numeric_id(stem)
        group, label = resolve_group_label(numeric, split=split)
        return cls(
            subject_id=canonical_subject_id(stem),
            subject_numeric_id=numeric,
            group=group,
            label=label,
            source_workbook=f"{stem}.xlsx",
        )
