"""
Tests for the gauge audit.
"""

import numpy as np
import pytest

import config
from geom import periodic, reference
from geom.cell import UnitCell
from screen import gauge, motifs
from sim2d import label

def band(width_px, n=32):
    """A circumferential band `width_px` deep."""
    arr = np.zeros((n, n), dtype=bool)
    arr[:width_px, :] = True
    return arr

# OFFSET #
def test_narrowing_takes_the_asked_pixels_off_each_side():
    arr = band(10)
    for offset in (1, 2, 3):
        thin = gauge.narrow(arr, offset)
        assert thin.sum(axis=0).max() == 10 - 2 * offset
        assert thin.sum(axis=1).max() == 32     # Untouched around the circumference

def test_narrowing_only_removes_material():
    arr = band(10)
    assert np.all(gauge.narrow(arr, 2) <= arr)

def test_narrowing_wraps():
    """A design is the same design shifted, so the offset has to be shift-equivariant."""
    rng = np.random.default_rng(0)
    arr = periodic.dilation(rng.random((32, 32)) > 0.9, iterations=3)
    assert periodic.is_shift_equivariant(lambda x: gauge.narrow(x, 2), arr)

def test_narrowing_severs_a_band_thinner_than_the_offset():
    """A one-pixel circumferential band can't survive any offset."""
    assert not gauge.narrow(band(1), 1).any()
    assert gauge.narrow(band(5), 2).any()

def test_the_measured_width_matches_a_known_band():
    """`member_width_mm` is a descriptor, so we pin it against a cell whose width is exact."""
    assert gauge.member_width_mm(band(8)) == pytest.approx(8 * config.mm_per_px()[0], rel=0.2)

# THE BAND #
@pytest.mark.parametrize('depth', [1, 5])
def test_a_full_circumferential_run_says_nothing_about_depth(depth):
    """`straight_run_circ` is 1.0 at any band depth."""
    assert motifs.longest_run(band(depth), 1) == pytest.approx(1.0)
    assert gauge.band_depth_px(band(depth)) == depth

def test_band_depth_counts_consecutive_full_rows():
    assert gauge.band_depth_px(np.ones((8, 8), dtype=bool)) == 8
    assert gauge.band_depth_px(np.zeros((8, 8), dtype=bool)) == 0
    assert gauge.band_depth_px(band(3)) == 3

def test_band_depth_wraps():
    assert gauge.band_depth_px(np.roll(band(4), -2, axis=0)) == 4

# GAUGE LADDER #
def test_the_gauge_ladder_is_the_pixel_arithmetic_and_keeps_its_aspect_ratio():
    assert gauge.gauge_mm(0) == gauge.REFERENCE_MM
    for offset in gauge.OFFSETS_PX:
        width, thickness = gauge.gauge_mm(offset)
        assert width == pytest.approx(config.STRUT_WIDTH_MM - 2 * offset * config.mm_per_px()[0])
        assert thickness / width == pytest.approx(config.STRUT_THICKNESS_MM / config.STRUT_WIDTH_MM)
        assert 0.0 < width < config.STRUT_WIDTH_MM

# WHAT THE 2D TIER CAN SEE #
def test_a_thickness_change_only_rescales_k_radial():
    """
    Plane stress has no out-of-plane thickness in it, so t multiplies K_radial and touches nothing else.
    """
    arr = reference.load().to_array()
    thick = gauge.relabel(arr, 0.22)
    thin = gauge.relabel(arr, 0.11)
    assert thin['K_radial'] / thick['K_radial'] == pytest.approx(0.5, rel=1e-12)
    for key in ('eps_a_max', 'eps_a_p99', 'A_over_lim', 'f_metal', 'E_circ'):
        assert thin[key] == thick[key]

def test_relabel_reproduces_the_projects_own_labeler():
    cell = reference.load()
    got = gauge.relabel(cell.to_array(), config.STRUT_THICKNESS_MM)
    want = label.label(cell)
    for key in ('K_radial', 'eps_a_max', 'A_over_lim', 'f_metal'):
        assert got[key] == pytest.approx(want.as_dict()[key], rel=1e-12)

def test_narrowing_makes_a_cell_lighter_and_more_compliant():
    arr = reference.load().to_array()
    thin = gauge.narrow(arr, 1)
    assert thin.mean() < arr.mean()
    assert (gauge.relabel(thin, config.STRUT_THICKNESS_MM)['E_circ']
            < gauge.relabel(arr, config.STRUT_THICKNESS_MM)['E_circ'])

# TWO STATISTICS #
def test_identical_arms_agree_perfectly():
    rows = [{'K_radial': k, 'eps_a_max': 1.0 / k, 'eps_a_p99': 1.0 / k,
             'A_over_lim': 0.01 * k, 'f_metal': 0.2 + 0.01 * k} for k in range(1, 13)]
    for d in gauge.spearman(rows, rows).values():
        assert d['spearman'] == pytest.approx(1.0)
    for d in gauge.top_k_overlap(rows, rows, ks=(3, 6)).values():
        assert d['overlap'] == pytest.approx(1.0)
    assert gauge.front_overlap(rows, rows)['jaccard'] == pytest.approx(1.0)

def test_a_reversed_ranking_is_caught():
    rows = [{'K_radial': k, 'eps_a_max': 1.0 / k, 'eps_a_p99': 1.0 / k,
             'A_over_lim': 0.01 * k, 'f_metal': 0.2 + 0.01 * k} for k in range(1, 13)]
    assert gauge.spearman(rows, rows[::-1])['K_radial']['spearman'] == pytest.approx(-1.0)

def test_top_k_overlap_skips_a_k_larger_than_the_pool():
    rows = [{'K_radial': k, 'A_over_lim': 0.01 * k, 'f_metal': 0.2 + 0.01 * k}
            for k in range(1, 5)]
    assert list(gauge.top_k_overlap(rows, rows, ks=(2, 50))) == ['2']
