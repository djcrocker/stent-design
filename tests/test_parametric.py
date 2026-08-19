"""The parametric crown family has to be valid across its sweep and labelled."""

import numpy as np
import pytest

import config
from geom import validity as V
from geom.parametric import (DEFAULT_AMPLITUDES, DEFAULT_PERIODS, achievable_widths,
                             crown, snap_width_mm, sweep)

# WIDTH SNAPPING: THE LABEL HAS TO EQUAL THE GEOMETRY #
def test_snapped_width_is_realisable_exactly():
    px_mm = config.mm_per_px()[0]
    for w in (0.10, 0.13, 0.17, 0.21, 0.25):
        snapped = snap_width_mm(w)
        assert snapped / px_mm == pytest.approx(round(snapped / px_mm), abs=1e-9)

def test_snapping_never_goes_below_the_manufacturing_floor():
    """A width is a minimum; snapping must not round through MIN_FEATURE_MM."""
    for w in (0.01, 0.05, 0.09, 0.10):
        assert snap_width_mm(w) >= config.MIN_FEATURE_MM - 1e-12

def test_achievable_widths_are_distinct_and_ordered():
    widths = achievable_widths()
    assert len(widths) == len(set(widths))
    assert widths == sorted(widths)
    assert all(w >= config.MIN_FEATURE_MM - 1e-12 for w in widths)

def test_snapping_is_idempotent():
    for w in achievable_widths():
        assert snap_width_mm(w) == pytest.approx(w)

# GEOMETRY #
def test_default_crown_is_valid():
    ok, reasons = V.is_valid(crown())
    assert ok, reasons

@pytest.mark.parametrize('n_periods', DEFAULT_PERIODS)
def test_crown_wraps_both_ways(n_periods):
    """Rings carry the hoop; links carry the axial path. Both must close."""
    cell = crown(strut_width_mm=0.1473, crown_amplitude=0.25, n_periods=n_periods)
    assert V.wraps(cell.to_array()) == (True, True)

def test_crown_is_a_single_component():
    cell = crown(strut_width_mm=0.1473, crown_amplitude=0.25, n_periods=2)
    _, count = __import__('geom.periodic', fromlist=['periodic']).label(cell.to_array())
    assert count == 1

def test_wider_struts_mean_more_metal():
    previous = 0.0
    for w in achievable_widths():
        f = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=2).f_metal
        assert f > previous
        previous = f

def test_link_length_is_derived_not_free():
    """2(A + L) = n is forced by the tiling, so L = 0.5 - amplitude."""
    valid, _ = sweep(widths_mm=[0.1473], amplitudes=(0.2, 0.3), periods=(2,))
    for params, _ in valid:
        assert params['link_length'] == pytest.approx(0.5 - params['crown_amplitude'])

# PARAMETER VALIDATION #
@pytest.mark.parametrize('amplitude', [0.0, 0.5, 0.7, -0.1])
def test_amplitude_outside_the_geometric_limit_is_refused(amplitude):
    with pytest.raises(ValueError):
        crown(crown_amplitude=amplitude)

def test_zero_periods_is_refused():
    with pytest.raises(ValueError):
        crown(n_periods=0)

# THE SWEEP #
def test_sweep_returns_only_valid_cells():
    valid, _ = sweep()
    assert valid
    for params, cell in valid:
        ok, reasons = V.is_valid(cell)
        assert ok, f'{params} -> {reasons}'

def test_sweep_rejections_are_reported_with_reasons():
    """Rejection at the grid corners is expected; it must be visible, not trimmed away."""
    _, rejected = sweep()
    assert rejected
    for params, reasons in rejected:
        assert reasons

def test_sweep_spans_a_useful_coverage_range():
    """The family must reach conventional stent coverage (~19-26%), or it is no baseline."""
    valid, _ = sweep()
    coverage = np.array([p['f_metal'] for p, _ in valid])
    assert coverage.min() < 0.26
    assert coverage.max() > 0.40

def test_sweep_levels_are_distinct_geometry():
    """Snapping means distinct parameters give distinct cells, not relabelled duplicates."""
    valid, _ = sweep(amplitudes=(0.2, 0.3), periods=(2,))
    arrays = {cell.to_array().tobytes() for _, cell in valid}
    assert len(arrays) == len(valid)
