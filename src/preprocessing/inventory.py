"""Source inventory: stimuli, subject workbooks, validation, and sizing."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .config import ConfigError
from .identifiers import SubjectIdentity, sha256_hex

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class InventoryError(ValueError):
    """Raised when the source layout is ambiguous or invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StimulusInfo:
    stimulus_index: int
    stimulus_id: str
    source_image_name: str
    category: str
    relative_image_path: str  # normalized POSIX path relative to image_root
    width: int
    height: int
    sha256: str

    def to_manifest_row(self) -> dict[str, object]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class SubjectFileInfo:
    path: Path
    identity: SubjectIdentity
    sha256: str
    size_bytes: int


def inventory_stimuli(image_root: Path) -> list[StimulusInfo]:
    """Build the deterministic stimulus manifest.

    Top-level directories are categories. Basenames must be unique across all
    categories (ambiguity is refused). ``stimulus_index`` is assigned in
    ``(category, basename)`` order; the numeric part of an image name plays no
    role in indexing.
    """
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise InventoryError(f"image root not found: {image_root}")

    category_dirs = sorted(d for d in image_root.iterdir() if d.is_dir())
    if not category_dirs:
        raise InventoryError(f"no category directories under {image_root}")

    seen_basenames: dict[str, Path] = {}
    entries: list[tuple[str, str, Path]] = []  # (category, basename, path)
    for cat_dir in category_dirs:
        files = sorted(
            p
            for p in cat_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        for p in files:
            if p.name in seen_basenames:
                raise InventoryError(
                    f"ambiguous image basename {p.name!r}: found in "
                    f"{seen_basenames[p.name].parent} and {p.parent}"
                )
            seen_basenames[p.name] = p
            entries.append((cat_dir.name, p.name, p))

    manifest: list[StimulusInfo] = []
    for index, (category, basename, path) in enumerate(sorted(entries)):
        with Image.open(path) as img:
            width, height = img.size
        manifest.append(
            StimulusInfo(
                stimulus_index=index,
                stimulus_id=basename,
                source_image_name=basename,
                category=category,
                relative_image_path=f"{category}/{basename}",
                width=width,
                height=height,
                sha256=_sha256_file(path),
            )
        )
    return manifest


def inventory_subjects(
    fixation_root: Path, subject_glob: str, subject_filename_regex: str, split: int
) -> tuple[list[SubjectFileInfo], list[str]]:
    """Inventory subject workbooks matching the configured glob and regex.

    Non-matching files (e.g. metadata workbooks) are excluded from the subject
    set and returned separately as ``other_files``.
    """
    fixation_root = Path(fixation_root)
    if not fixation_root.is_dir():
        raise InventoryError(f"fixation root not found: {fixation_root}")
    pattern = re.compile(subject_filename_regex)
    candidates = sorted(fixation_root.glob(subject_glob))
    if not candidates:
        raise InventoryError(f"no files match {subject_glob!r} under {fixation_root}")

    subjects: list[SubjectFileInfo] = []
    other_files: list[str] = []
    for p in candidates:
        if not p.is_file():
            other_files.append(p.name)
            continue
        if not pattern.fullmatch(p.name):
            other_files.append(p.name)
            continue
        identity = SubjectIdentity.from_stem(p.stem, split=split)
        subjects.append(
            SubjectFileInfo(
                path=p,
                identity=identity,
                sha256=_sha256_file(p),
                size_bytes=p.stat().st_size,
            )
        )
    if not subjects:
        raise InventoryError(
            f"no subject workbooks match {subject_filename_regex!r} under {fixation_root}"
        )
    # Deterministic subject order: ascending numeric ID (string form preserved
    # elsewhere; ordering is by numeric value for a stable artifact layout).
    subjects.sort(key=lambda s: s.identity.subject_numeric_id)
    return subjects, other_files


def estimate_output_size(
    n_trials_ok: int, heatmap_height: int, heatmap_width: int
) -> int:
    """Estimate the per-subject heatmap payload bytes (3 channels, float32)."""
    return n_trials_ok * 3 * heatmap_height * heatmap_width * 4


def verify_free_space(output_root: Path, required_bytes: int) -> int:
    """Return free bytes on the output filesystem; fail if insufficient."""
    output_root = Path(output_root)
    parent = output_root if output_root.exists() else output_root.parent
    usage = shutil.disk_usage(parent)
    # Require the estimate plus a 512 MiB working margin.
    needed = required_bytes + 512 * (1 << 20)
    if usage.free < needed:
        raise InventoryError(
            f"insufficient free space for output at {parent}: "
            f"need ~{needed / 1e9:.2f} GB, have {usage.free / 1e9:.2f} GB"
        )
    return usage.free


def validate_image_references(
    trials_by_image: dict[str, object], stimuli: list[StimulusInfo]
) -> None:
    """Refuse to continue on workbook image references missing from disk."""
    known = {s.stimulus_id for s in stimuli}
    missing = sorted(set(trials_by_image) - known)
    if missing:
        raise InventoryError(
            f"workbooks reference images missing on disk: {missing}"
        )


def source_inventory_json(
    stimuli: list[StimulusInfo],
    subjects: list[SubjectFileInfo],
    other_files: list[str],
    anomalies: dict[str, object],
) -> str:
    """Render the reproducible ``source_inventory.json`` (no timestamps)."""
    payload = {
        "images": [s.to_manifest_row() for s in stimuli],
        "subjects": [
            {
                "file": s.path.name,
                "subject_id": s.identity.subject_id,
                "subject_numeric_id": s.identity.subject_numeric_id,
                "group": s.identity.group,
                "label": s.identity.label,
                "sha256": s.sha256,
                "size_bytes": s.size_bytes,
            }
            for s in subjects
        ],
        "other_files": other_files,
        "anomaly_measurements": anomalies,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
