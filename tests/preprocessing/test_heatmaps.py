"""Heatmap channel-contract tests on synthetic events."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.heatmaps import (
    HeatmapParams,
    build_trial_heatmap,
    fixation_density,
    grid_position,
    temporal_progression,
    transition_density,
)

PARAMS = HeatmapParams(
    height=48, width=64, sigma=2.0, truncate=4.0, transition_step=0.5, temporal_epsilon=1e-8
)
SW, SH = 1024, 768


def _build(xs, ys, spatial=None, durations=None, duration_valid=None):
    n = len(xs)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if spatial is None:
        spatial = (0 <= xs) & (xs <= SW) & (0 <= ys) & (ys <= SH)
    if durations is None:
        durations = np.full(n, 300.0)
    if duration_valid is None:
        duration_valid = np.ones(n, dtype=bool)
    return build_trial_heatmap(
        xs, ys, spatial, durations, duration_valid,
        source_width=SW, source_height=SH, params=PARAMS,
    )


def test_grid_position_contract():
    u, v = grid_position(0.0, 0.0, SW, SH, PARAMS.height, PARAMS.width)
    assert u == 0.0 and v == 0.0
    # Last pixel center maps exactly to the last grid cell center.
    u, v = grid_position(1023.0, 767.0, SW, SH, PARAMS.height, PARAMS.width)
    assert u == PARAMS.width - 1 and v == PARAMS.height - 1
    # Canvas center maps to the grid center: 512*63/1023 and 384*47/767.
    u, v = grid_position(512.0, 384.0, SW, SH, PARAMS.height, PARAMS.width)
    assert u == pytest.approx(512.0 * 63 / 1023)
    assert v == pytest.approx(384.0 * 47 / 767)


def test_center_fixation_symmetric_mass_one():
    # x=511.5 and y=383.5 map exactly onto the grid center (31.5, 23.5), so
    # the kernel is symmetric about the cell-center boundary.
    heatmap, stats = _build([511.5], [383.5])
    ch0 = heatmap[0]
    assert stats.mass_density == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(ch0, ch0[::-1], atol=0.0)  # vertical symmetry
    assert np.allclose(ch0, ch0[:, ::-1], atol=0.0)  # horizontal symmetry
    assert heatmap.shape == (3, 48, 64)
    assert heatmap.dtype == np.float32
    assert np.all(np.isfinite(heatmap))


def test_border_fixation_mass_one_after_truncation():
    for x, y in [(0.0, 0.0), (1024.0, 768.0), (0.0, 768.0), (1024.0, 0.0)]:
        heatmap, stats = _build([x], [y])
        assert stats.mass_density == pytest.approx(1.0, abs=1e-6), (x, y)
        assert np.all(np.isfinite(heatmap))


def test_two_fixations_mass_two():
    heatmap, stats = _build([100.0, 900.0], [100.0, 700.0])
    assert stats.mass_density == pytest.approx(2.0, abs=1e-6)
    assert stats.n_fixations_used == 2


def test_one_transition_mass_one():
    heatmap, stats = _build([100.0, 900.0], [100.0, 700.0])
    assert stats.n_transitions_used == 1
    assert stats.mass_transition == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(heatmap[1]))
    assert np.all(heatmap[1] >= 0)


def test_zero_length_transition_finite_mass_one():
    heatmap, stats = _build([400.0, 400.0], [300.0, 300.0])
    assert stats.n_transitions_used == 1
    assert stats.mass_transition == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(heatmap[1]))


def test_invalid_middle_fixation_prevents_bridging():
    # Middle event is off-canvas: pair (0,1) and (1,2) are both rejected, so
    # the valid endpoints are NOT connected -> zero transitions.
    heatmap, stats = _build(
        [100.0, 2000.0, 900.0],
        [100.0, 500.0, 700.0],
        spatial=[True, False, True],
    )
    assert stats.n_transitions_used == 0
    assert stats.mass_transition == 0.0
    # Fixation density only counts the two spatially valid events.
    assert stats.mass_density == pytest.approx(2.0, abs=1e-6)


def test_off_canvas_contributes_time_but_not_density():
    # Off-canvas event in the middle of the sequence still advances the
    # temporal clock for later events.
    xs = [200.0, 1800.0, 800.0]
    ys = [200.0, 900.0, 500.0]
    spatial = [True, False, True]
    durations = [100.0, 500.0, 100.0]
    heatmap, stats = _build(xs, ys, spatial=spatial, durations=durations)
    assert stats.n_fixations_used == 2
    assert stats.mass_density == pytest.approx(2.0, abs=1e-6)
    # Total clock = 700; first fixation mid = 50/700 -> tau ~ -0.857 (early),
    # last fixation mid = (600 + 50)/700 -> tau ~ +0.857 (late).
    ch2 = heatmap[2]
    u0, v0 = grid_position(200.0, 200.0, SW, SH, 48, 64)
    u2, v2 = grid_position(800.0, 500.0, SW, SH, 48, 64)
    early = ch2[int(round(v0)), int(round(u0))]
    late = ch2[int(round(v2)), int(round(u2))]
    assert early < 0 < late
    assert np.all(np.isfinite(ch2))
    assert np.all(np.abs(ch2) <= 1.0)


def test_temporal_progression_range_and_zeros():
    # Unvisited locations are exactly zero; visited values in [-1, 1].
    xs = [200.0, 800.0, 500.0]
    ys = [200.0, 500.0, 700.0]
    durations = [10.0, 500.0, 400.0]
    heatmap, stats = _build(xs, ys, durations=durations)
    ch2 = heatmap[2]
    ch0 = heatmap[0]
    assert np.all(np.isfinite(ch2))
    assert np.all(np.abs(ch2) <= 1.0 + 1e-6)
    # Cells with no fixation density are exactly zero.
    unvisited = ch0 <= PARAMS.temporal_epsilon
    assert np.all(ch2[unvisited] == 0.0)
    assert stats.n_temporal_clipped >= 0


def test_temporal_undefined_when_no_positive_duration():
    # Under the approved policy, nonpositive durations are dropped, so a trial
    # whose every event has a nonpositive duration has no usable spatial
    # fixation and must raise (the QC layer marks it excluded).
    with pytest.raises(Exception, match="no usable spatial fixation"):
        _build(
            [200.0, 800.0],
            [200.0, 500.0],
            durations=[0.0, -3.0],
            duration_valid=[False, False],
        )


def test_heatmap_determinism_and_mass_consistency():
    xs = [100.0, 300.0, 900.0, 500.0]
    ys = [100.0, 600.0, 700.0, 200.0]
    durations = [120.0, 340.0, 210.0, 500.0]
    h1, s1 = _build(xs, ys, durations=durations)
    h2, s2 = _build(xs, ys, durations=durations)
    assert np.array_equal(h1, h2)
    assert s1.mass_density == pytest.approx(s1.mass_density_expected, abs=1e-3)
    assert s1.mass_transition == pytest.approx(s1.mass_transition_expected, abs=1e-3)


def test_no_usable_spatial_fixation_raises():
    with pytest.raises(Exception, match="no usable spatial fixation"):
        _build([1500.0], [900.0])


def test_channel_order_and_nonnegativity():
    xs = [100.0, 300.0, 900.0]
    ys = [100.0, 600.0, 700.0]
    heatmap, _ = _build(xs, ys)
    # Channel 0: fixation density (sum = 3 fixations); channel 1: transitions
    # (sum = 2); channel 2: temporal progression in [-1, 1].
    assert heatmap[0].sum() == pytest.approx(3.0, abs=1e-4)
    assert heatmap[1].sum() == pytest.approx(2.0, abs=1e-4)
    assert np.all(heatmap[0] >= 0)
    assert np.all(heatmap[1] >= 0)
    assert np.all(np.abs(heatmap[2]) <= 1.0 + 1e-6)
