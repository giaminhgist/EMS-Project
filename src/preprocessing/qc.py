"""Per-trial quality control and policy application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedEvent:
    """One fixation row after explicit coercion."""

    row_number: int  # 1-based row in the source sheet (header is row 1)
    image: str
    fix_index: int | None
    fix_index_status: str  # ok | empty | malformed
    duration: float | None
    duration_status: str  # ok | empty | nonfinite | malformed
    x: float | None
    y: float | None
    pupil: float | None
    coord_status: str  # ok | empty | nonfinite | malformed
    pupil_status: str  # ok | empty | nonfinite | malformed


def coerce_float(value: Any) -> tuple[str, float | None]:
    """Explicitly coerce a cell to float and classify the outcome."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return ("empty", None)
    try:
        f = float(value)
    except (ValueError, TypeError):
        return ("malformed", None)
    import math

    if math.isnan(f):
        return ("nonfinite", None)
    if math.isinf(f):
        return ("nonfinite", None)
    return ("ok", f)


def coerce_int(value: Any) -> tuple[str, int | None]:
    """Coerce a FIX_INDEX cell to int; fractional values are rejected."""
    status, f = coerce_float(value)
    if status != "ok" or f is None:
        return (status, None)
    if f != int(f):
        return ("malformed", None)
    return ("ok", int(f))


@dataclass
class TrialEvents:
    """One observed ``(subject_id, stimulus_id)`` trial, sorted by FIX_INDEX."""

    subject_id: str
    stimulus_id: str
    events: list[ParsedEvent] = field(default_factory=list)

    def sorted_by_fix_index(self) -> list[ParsedEvent]:
        """Stable-sort by FIX_INDEX, then original row number.

        Malformed indices sort last in their original row order.
        """
        def key(e: ParsedEvent) -> tuple[int, int, int]:
            if e.fix_index is None:
                return (1, 0, e.row_number)
            return (0, e.fix_index, e.row_number)

        return sorted(self.events, key=key)


@dataclass
class TrialQC:
    """Measured QC record for one trial (see trial-manifest contract)."""

    n_fixations_raw: int
    n_fixations_used: int
    n_transitions_used: int
    total_duration_ms_raw: float
    total_duration_ms_used: float
    median_fixation_duration_ms: float | None
    n_off_canvas: int
    n_nonfinite: int
    n_nonpositive_duration: int
    n_below_duration_threshold: int
    fix_index_has_gap: bool
    fix_index_has_duplicates: bool
    fix_index_non_monotonic: bool
    temporal_undefined: bool
    n_malformed_fix_index: int
    n_malformed_duration: int
    n_malformed_pupil: int
    qc_status: str  # "ok" | "excluded_no_spatial_fixations"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_fixations_raw": self.n_fixations_raw,
            "n_fixations_used": self.n_fixations_used,
            "n_transitions_used": self.n_transitions_used,
            "total_duration_ms_raw": self.total_duration_ms_raw,
            "total_duration_ms_used": self.total_duration_ms_used,
            "median_fixation_duration_ms": self.median_fixation_duration_ms,
            "n_off_canvas": self.n_off_canvas,
            "n_nonfinite": self.n_nonfinite,
            "n_nonpositive_duration": self.n_nonpositive_duration,
            "n_below_duration_threshold": self.n_below_duration_threshold,
            "fix_index_has_gap": self.fix_index_has_gap,
            "fix_index_has_duplicates": self.fix_index_has_duplicates,
            "fix_index_non_monotonic": self.fix_index_non_monotonic,
            "temporal_undefined": self.temporal_undefined,
            "n_malformed_fix_index": self.n_malformed_fix_index,
            "n_malformed_duration": self.n_malformed_duration,
            "n_malformed_pupil": self.n_malformed_pupil,
            "qc_status": self.qc_status,
        }


def compute_trial_qc(
    subject_id: str,
    stimulus_id: str,
    sorted_events: list[ParsedEvent],
    *,
    source_width: int,
    source_height: int,
    min_fix_duration_ms: float,
    drop_nonpositive_duration: bool,
    zero_spatial_policy: str,
) -> TrialQC:
    """Apply the approved duration/coordinate policies and measure QC counts.

    Spatial validity: finite coordinates inside the inclusive canvas
    ``[0, W] x [0, H]``. Duration validity: finite duration strictly greater
    than ``min_fix_duration_ms`` (default 0, so only nonpositive durations are
    dropped).
    """
    n_off_canvas = 0
    n_nonfinite = 0
    n_nonpositive_duration = 0
    n_below_threshold = 0
    n_malformed_fix_index = 0
    n_malformed_duration = 0
    n_malformed_pupil = 0

    spatial_valid: list[bool] = []
    duration_valid: list[bool] = []
    total_duration_raw = 0.0
    total_duration_used = 0.0
    used_durations: list[float] = []

    for e in sorted_events:
        if e.fix_index_status in ("empty", "malformed"):
            n_malformed_fix_index += 1
        if e.coord_status in ("nonfinite", "malformed"):
            n_nonfinite += 1
        if e.duration_status == "malformed":
            n_malformed_duration += 1
        if e.pupil_status == "malformed":
            n_malformed_pupil += 1

        if e.coord_status == "ok" and e.x is not None and e.y is not None:
            in_canvas = 0.0 <= e.x <= float(source_width) and 0.0 <= e.y <= float(source_height)
            spatial_valid.append(in_canvas)
            if not in_canvas:
                n_off_canvas += 1
        else:
            spatial_valid.append(False)

        if e.duration_status == "ok" and e.duration is not None:
            d = e.duration
            if d > 0:
                total_duration_raw += d
            if d <= 0:
                n_nonpositive_duration += 1
            if 0 < d <= min_fix_duration_ms:
                n_below_threshold += 1
            dur_ok = d > min_fix_duration_ms
            if dur_ok and (not drop_nonpositive_duration or d > 0):
                total_duration_used += d
                used_durations.append(d)
            duration_valid.append(bool(dur_ok))
        else:
            duration_valid.append(False)

    n_used = sum(sp and dv for sp, dv in zip(spatial_valid, duration_valid))

    # FIX_INDEX hygiene flags.
    idxs = [e.fix_index for e in sorted_events if e.fix_index is not None]
    has_duplicates = len(idxs) != len(set(idxs))
    has_gap = bool(idxs) and sorted(idxs) != list(range(min(idxs), max(idxs) + 1))
    # Monotonicity in the original sheet row order (before sorting).
    row_order_idxs = [e.fix_index for e in sorted(sorted_events, key=lambda e: e.row_number) if e.fix_index is not None]
    non_monotonic = row_order_idxs != sorted(row_order_idxs)

    temporal_undefined = total_duration_used <= 0
    n_transitions_used = 0
    for i in range(len(spatial_valid) - 1):
        if spatial_valid[i] and spatial_valid[i + 1]:
            n_transitions_used += 1

    if n_used == 0:
        qc_status = "excluded_no_spatial_fixations"
        if zero_spatial_policy != "exclude_no_spatial":
            raise ValueError(f"unsupported zero_spatial_policy {zero_spatial_policy!r}")
    else:
        qc_status = "ok"

    if used_durations:
        used_durations.sort()
        median_fixation_duration_ms = used_durations[len(used_durations) // 2]
    else:
        median_fixation_duration_ms = None

    return TrialQC(
        n_fixations_raw=len(sorted_events),
        n_fixations_used=n_used,
        n_transitions_used=n_transitions_used,
        total_duration_ms_raw=total_duration_raw,
        total_duration_ms_used=total_duration_used,
        median_fixation_duration_ms=median_fixation_duration_ms,
        n_off_canvas=n_off_canvas,
        n_nonfinite=n_nonfinite,
        n_nonpositive_duration=n_nonpositive_duration,
        n_below_duration_threshold=n_below_threshold,
        fix_index_has_gap=has_gap,
        fix_index_has_duplicates=has_duplicates,
        fix_index_non_monotonic=non_monotonic,
        temporal_undefined=temporal_undefined,
        n_malformed_fix_index=n_malformed_fix_index,
        n_malformed_duration=n_malformed_duration,
        n_malformed_pupil=n_malformed_pupil,
        qc_status=qc_status,
    )
