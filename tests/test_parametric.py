"""The parametric crown family has to be valid across its sweep and labeled."""

import numpy as np
import pytest

import config
from geom import validity as V
from geom.parametric import (DEFAULT_AMPLITUDES, DEFAULT_PERIODS, achievable_widths,
                             crown, extended_sweep, snap_width_mm, sweep)

# WIDTH SNAPPING: THE LABEL HAS TO EQUAL THE GEOMETRY #
def test_snapped_width_is_realizable_exactly():
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
    """Snapping means distinct parameters give distinct cells, not relabeled duplicates."""
    valid, _ = sweep(amplitudes=(0.2, 0.3), periods=(2,))
    arrays = {cell.to_array().tobytes() for _, cell in valid}
    assert len(arrays) == len(valid)

# FAIR-BASELINE EXTENSION (S10.1 Part 2) #
def _two_descriptors(arr):
    """`thickness_cv` and `straight_run_circ` only, off the same helpers `describe` uses."""
    from geom import periodic
    from screen import motifs

    thick = periodic.distance_transform(arr)[motifs.skeleton(arr)]
    return {'thickness_cv': float(thick.std() / thick.mean()),
            'straight_run_circ': motifs.longest_run(arr, 1)}

@pytest.fixture(scope='module')
def extended():
    """The extended family and the two descriptors these tests assert on."""
    valid, _ = extended_sweep()
    return valid, [_two_descriptors(cell.to_array()) for _, cell in valid]

def test_the_fast_descriptors_match_describe():
    from screen import motifs

    arr = crown(strut_width_mm=0.196, crown_amplitude=0.25, n_periods=2, band_width_mm=0.10, width_taper=0.45).to_array()
    full, fast = motifs.describe(arr), _two_descriptors(arr)
    for key, value in fast.items():
        assert value == pytest.approx(full[key])

def test_the_extension_defaults_to_the_published_family():
    assert np.array_equal(crown().to_array(), crown(band_width_mm=0.0, width_taper=0.0).to_array())

def test_the_default_sweep_is_the_population_s10_3_measured():
    """76 valid / 32 rejected."""
    valid, rejected = sweep()
    assert (len(valid), len(rejected)) == (76, 32)
    assert not any('band_width_mm' in p or 'width_taper' in p for p, _ in valid)

def test_a_band_closes_a_complete_circumferential_hoop():
    """`straight_run_circ` is the descriptor Finding 2 rests on; 1.0 means unbroken."""
    from screen import motifs

    plain = crown(strut_width_mm=0.10, crown_amplitude=0.30, n_periods=2)
    banded = crown(strut_width_mm=0.10, crown_amplitude=0.30, n_periods=2, band_width_mm=0.15)
    assert motifs.describe(plain.to_array())['straight_run_circ'] < 1.0
    assert motifs.describe(banded.to_array())['straight_run_circ'] == pytest.approx(1.0)

def test_the_band_wraps_instead_of_sitting_inside_the_cell():
    """A hoop has to cross the seam, or tiling gives two half-bands and no ring."""
    arr = crown(band_width_mm=0.15).to_array()
    assert arr[0].all() and arr[-1].all()

def test_a_taper_breaks_the_uniform_width_the_family_was_stuck_at():
    """Past 0.131 is the point of this parameter."""
    from screen import motifs

    uniform = crown(strut_width_mm=0.245, crown_amplitude=0.25, n_periods=2)
    tapered = crown(strut_width_mm=0.245, crown_amplitude=0.25, n_periods=2, width_taper=0.45)
    assert motifs.describe(uniform.to_array())['thickness_cv'] < 0.131
    assert motifs.describe(tapered.to_array())['thickness_cv'] > 0.131

def test_the_taper_is_symmetric_about_both_turns_of_the_wave():
    """
    Ring 1 spans axial [0, 16] at this amplitude, with its trough ridge at row 0 and its
    peak ridge at row 16, so rows 0 and 15 cut through the two apexes of the same ring and
    are comparable by construction.
    """
    flat = crown(strut_width_mm=0.147, crown_amplitude=0.25, n_periods=2).to_array()
    arr = crown(strut_width_mm=0.147, crown_amplitude=0.25, n_periods=2,
                width_taper=0.45).to_array()
    assert not np.array_equal(arr, flat)             # the taper did something
    assert arr[0].sum() == arr[15].sum()             # and it did it at both apexes

@pytest.mark.parametrize('taper', [1.0, -1.0, 1.5])
def test_a_taper_that_reaches_zero_width_is_refused(taper):
    with pytest.raises(ValueError, match='width_taper'):
        crown(width_taper=taper)

def test_a_negative_band_is_refused():
    with pytest.raises(ValueError, match='band_width_mm'):
        crown(band_width_mm=-0.1)

def test_the_extended_sweep_reaches_both_motifs_the_old_family_could_not_aim_at(extended):
    valid, rows = extended
    assert len(valid) > len(sweep()[0])
    assert sum(1 for d in rows if d['thickness_cv'] > 0.131) > 400
    assert sum(1 for d in rows if d['straight_run_circ'] >= 0.999) > 300

def test_the_extended_sweep_still_cannot_reach_the_winner_median(extended):
    assert 0.131 < max(d['thickness_cv'] for d in extended[1]) < 0.427

def test_every_extended_cell_is_still_valid_and_recorded(extended):
    for params, cell in extended[0]:
        assert V.check(cell).ok
        if params.get('band_width_mm') or params.get('width_taper'):
            assert 'f_metal' in params
