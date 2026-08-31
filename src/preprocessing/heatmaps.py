"""Three-channel heatmap construction per ``01_DATA_MODEL_CONTRACTS.md``.

Channels are stored in this immutable order:

```text
0 = fixation_density
1 = consecutive_fixation_transition_density
2 = temporal_progression
```

All computations work in float64 and are cast to float32 only when the final
tensor is validated. No per-subject, per-group, or population normalization is
applied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class HeatmapError(ValueError):
    """Raised when a heatmap violates the channel contract."""


@dataclass(frozen=True)
class HeatmapParams:
    """Grid and kernel parameters shared by all channel builders."""

    height: int  # 48
    width: int  # 64
    sigma: float  # grid cells, default 2.0
    truncate: float  # truncation in sigma units, default 4.0
    transition_step: float  # max line sampling spacing in cells, default 0.5
    temporal_epsilon: float  # default 1e-8

    @property
    def radius(self) -> int:
        return int(math.ceil(self.truncate * self.sigma))


def grid_position(
    x: float, y: float, source_width: int, source_height: int, height: int, width: int
) -> tuple[float, float]:
    """Raw-to-grid coordinate contract (floating-point, never pre-rounded)."""
    u = x * (width - 1) / (source_width - 1)
    v = y * (height - 1) / (source_height - 1)
    return u, v


def _kernel1d(center: float, length: int, sigma: float, radius: int) -> tuple[np.ndarray, int]:
    """Unit-sum truncated Gaussian kernel along one axis.

    The window covers all cells within ``radius`` of ``center``
    (``ceil(center - r) .. floor(center + r)``) so the window is symmetric
    about the true (possibly fractional) center. Returns ``(kernel, lo)``
    where ``lo`` is the first index covered, so the caller can place the
    kernel with ``field[lo : lo + len(kernel)] += w * kernel``.
    """
    lo = int(math.ceil(center - radius))
    hi = int(math.floor(center + radius))
    lo = max(0, lo)
    hi = min(length - 1, hi)
    if hi < lo:
        # Center is outside the grid (cannot happen for on-canvas coordinates,
        # but guard for safety): deposit at the nearest boundary cell.
        lo = hi = min(length - 1, max(0, int(math.floor(center))))
    pos = np.arange(lo, hi + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * ((pos - center) / sigma) ** 2)
    total = kernel.sum()
    if total <= 0 or not np.isfinite(total):
        raise HeatmapError("degenerate Gaussian kernel")
    return kernel / total, lo


def deposit_gaussian(
    field: np.ndarray,
    u: float,
    v: float,
    params: HeatmapParams,
    weight: float = 1.0,
) -> None:
    """Add one unit-mass (times ``weight``) truncated Gaussian at ``(u, v)``.

    Each truncated kernel is renormalized so its discrete sum is one, exactly as
    required for border fixations.
    """
    kx, lo_x = _kernel1d(u, params.width, params.sigma, params.radius)
    ky, lo_y = _kernel1d(v, params.height, params.sigma, params.radius)
    field[lo_y : lo_y + ky.size, lo_x : lo_x + kx.size] += weight * np.outer(ky, kx)


def fixation_density(
    us: np.ndarray, vs: np.ndarray, params: HeatmapParams
) -> np.ndarray:
    """Channel 0: H_F = sum of unit-mass Gaussians over used fixations.

    Total mass is deliberately approximately the number of valid fixations.
    """
    field = np.zeros((params.height, params.width), dtype=np.float64)
    for u, v in zip(us, vs):
        deposit_gaussian(field, float(u), float(v), params)
    return field


def transition_density(
    us: np.ndarray,
    vs: np.ndarray,
    endpoint_valid: np.ndarray,
    params: HeatmapParams,
) -> np.ndarray:
    """Channel 1: consecutive-fixation transition density.

    Pairs are formed between adjacent events in original ``FIX_INDEX`` order
    (``us``/``vs`` must already be sorted). A pair is used only when both
    endpoints are spatially valid; rejected intermediate events are never
    bridged. Each accepted segment is rasterized with sub-cell sampling,
    normalized to unit mass, and Gaussian-smoothed, so each segment contributes
    exactly unit mass.
    """
    field = np.zeros((params.height, params.width), dtype=np.float64)
    raster = np.zeros((params.height, params.width), dtype=np.float64)
    for i in range(len(us) - 1):
        if not (endpoint_valid[i] and endpoint_valid[i + 1]):
            continue
        u0, v0 = float(us[i]), float(vs[i])
        u1, v1 = float(us[i + 1]), float(vs[i + 1])
        du, dv = u1 - u0, v1 - v0
        dist = math.hypot(du, dv)
        if dist < 1e-9:
            # Zero-length transition: unit mass at its endpoint.
            deposit_gaussian(field, u1, v1, params)
            continue
        n_samples = max(1, int(math.ceil(dist / params.transition_step)))
        raster.fill(0.0)
        for k in range(n_samples):
            t = (k + 0.5) / n_samples
            su = u0 + t * du
            sv = v0 + t * dv
            x0 = int(math.floor(su))
            y0 = int(math.floor(sv))
            fx = su - x0
            fy = sv - y0
            x1 = min(x0 + 1, params.width - 1)
            y1 = min(y0 + 1, params.height - 1)
            x0 = max(0, min(x0, params.width - 1))
            y0 = max(0, min(y0, params.height - 1))
            w00 = (1.0 - fx) * (1.0 - fy)
            w10 = fx * (1.0 - fy)
            w01 = (1.0 - fx) * fy
            w11 = fx * fy
            raster[y0, x0] += w00
            raster[y0, x1] += w10
            raster[y1, x0] += w01
            raster[y1, x1] += w11
        total = raster.sum()
        if total <= 0 or not np.isfinite(total):
            raise HeatmapError("transition rasterization produced empty segment")
        # Normalize the segment to unit mass, then smooth with the Gaussian.
        nonzero = np.nonzero(raster)
        for y, x in zip(*nonzero):
            deposit_gaussian(field, float(x), float(y), params, weight=raster[y, x] / total)
    return field


def temporal_progression(
    us: np.ndarray,
    vs: np.ndarray,
    taus: np.ndarray,
    density: np.ndarray,
    params: HeatmapParams,
) -> tuple[np.ndarray, int]:
    """Channel 2: density-conditioned local temporal mean.

    ``taus[i] = 2 * t_mid_i - 1`` in ``[-1, 1]``, with ``t_mid`` computed from
    the cumulative positive-duration clock. Unvisited locations are set to
    exactly zero. Returns ``(field, n_clipped)`` where ``n_clipped`` counts
    small floating-point overshoots outside ``[-1, 1]``.
    """
    numerator = np.zeros((params.height, params.width), dtype=np.float64)
    for u, v, tau in zip(us, vs, taus):
        deposit_gaussian(numerator, float(u), float(v), params, weight=float(tau))
    visited = density > params.temporal_epsilon
    field = np.zeros_like(density)
    np.divide(numerator, density + params.temporal_epsilon, out=field, where=visited)
    # Clip only tiny floating-point overshoots; larger violations are errors.
    overshoot = np.abs(field) - 1.0
    big = int(np.count_nonzero(overshoot > 1e-6))
    if big > 0:
        raise HeatmapError(
            f"temporal progression exceeded [-1, 1] by more than 1e-6 in {big} cells"
        )
    clipped = int(np.count_nonzero(overshoot > 0.0))
    field = np.clip(field, -1.0, 1.0)
    field[~visited] = 0.0
    return field, clipped


@dataclass(frozen=True)
class TrialHeatmapStats:
    n_fixations_used: int
    n_transitions_used: int
    mass_density: float
    mass_transition: float
    mass_density_expected: int
    mass_transition_expected: int
    n_temporal_clipped: int
    max_abs_temporal: float


def build_trial_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    spatial_valid: np.ndarray,
    durations: np.ndarray,
    duration_valid: np.ndarray,
    source_width: int,
    source_height: int,
    params: HeatmapParams,
) -> tuple[np.ndarray, TrialHeatmapStats]:
    """Build the float32 ``[3, H, W]`` heatmap for one trial.

    ``xs``/``ys`` are raw canvas coordinates in original ``FIX_INDEX`` order;
    ``spatial_valid`` marks finite on-canvas events; ``durations`` are raw
    fixation durations (non-finite where invalid); ``duration_valid`` marks
    temporally valid positive-duration events (which may still be off-canvas
    and therefore contribute elapsed time only).
    """
    if not (len(xs) == len(ys) == len(spatial_valid) == len(durations) == len(duration_valid)):
        raise HeatmapError("event arrays must have equal length")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    spatial_valid = np.asarray(spatial_valid, dtype=bool)
    duration_valid = np.asarray(duration_valid, dtype=bool)
    durations = np.asarray(durations, dtype=np.float64)

    used = spatial_valid & duration_valid
    n_used = int(np.count_nonzero(used))
    if n_used == 0:
        raise HeatmapError("trial has no usable spatial fixation")

    # Floating-point grid positions for every event.
    us = xs * (params.width - 1) / (source_width - 1)
    vs = ys * (params.height - 1) / (source_height - 1)

    # Channel 0: fixation density (unit-mass Gaussian per used fixation).
    density = fixation_density(us[used], vs[used], params)
    mass_density = float(density.sum())

    # Channel 1: consecutive transition density.
    transitions = transition_density(us, vs, spatial_valid, params)
    n_transitions_used = int(np.count_nonzero(spatial_valid[:-1] & spatial_valid[1:]))
    mass_transition = float(transitions.sum())

    # Channel 2: temporal progression.
    # Cumulative clock over all temporally valid positive-duration events in
    # original order, including off-canvas events (elapsed time only).
    total_duration = float(np.sum(durations[duration_valid]))
    taus = np.zeros(len(xs), dtype=np.float64)
    if total_duration > 0 and np.isfinite(total_duration):
        clock = np.cumsum(np.where(duration_valid, durations, 0.0))
        t_mid = np.zeros(len(xs), dtype=np.float64)
        np.divide(
            clock - 0.5 * np.where(duration_valid, durations, 0.0),
            total_duration,
            out=t_mid,
            where=duration_valid,
        )
        taus = 2.0 * t_mid - 1.0
    # else: total_duration == 0 -> temporal clock undefined; channel stays zero
    # (recorded as temporal_undefined in QC).

    progression, n_clipped = temporal_progression(us[used], vs[used], taus[used], density, params)

    heatmap = np.stack([density, transitions, progression], axis=0).astype(np.float32)
    if not np.all(np.isfinite(heatmap)):
        raise HeatmapError("non-finite values in trial heatmap")
    if np.any(heatmap[:2] < 0):
        raise HeatmapError("negative values in density/transition channels")
    if np.any(np.abs(heatmap[2]) > 1.0 + 1e-6):
        raise HeatmapError("temporal channel exceeds [-1, 1] beyond tolerance")

    # Contract mass checks (approximate, tolerance for float accumulation).
    expected_density = n_used
    expected_transition = n_transitions_used
    if abs(mass_density - expected_density) > 1e-3 * max(1, expected_density):
        raise HeatmapError(
            f"fixation density mass {mass_density} != expected {expected_density}"
        )
    if abs(mass_transition - expected_transition) > 1e-3 * max(1, expected_transition):
        raise HeatmapError(
            f"transition density mass {mass_transition} != expected {expected_transition}"
        )

    stats = TrialHeatmapStats(
        n_fixations_used=n_used,
        n_transitions_used=n_transitions_used,
        mass_density=mass_density,
        mass_transition=mass_transition,
        mass_density_expected=expected_density,
        mass_transition_expected=expected_transition,
        n_temporal_clipped=n_clipped,
        max_abs_temporal=float(np.max(np.abs(progression))) if n_used else 0.0,
    )
    return heatmap, stats
