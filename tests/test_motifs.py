"""Motif descriptors."""

import numpy as np
import pytest

import config
from screen import motifs

GRID = 64

def straight_axial(width=6, col=10, n=GRID):
    """One dead-straight strut spanning the cell in the axial direction."""
    a = np.zeros((n, n), bool)
    a[:, col:col + width] = True
    return a

def straight_circ(width=6, row=10, n=GRID):
    a = np.zeros((n, n), bool)
    a[row:row + width, :] = True
    return a

def cross(width=6, n=GRID):
    return straight_axial(width, 10, n) | straight_circ(width, 30, n)

def zigzag(width=5, n=GRID, amp=12):
    """A strut that serpentines as it crosses axially, so it has to be longer than the span."""
    a = np.zeros((n, n), bool)
    for r in range(n):
        # Triangle wave in the column coordinate: folds back and forth while advancing.
        phase = (r * 4.0 / n) % 2.0
        off = phase if phase <= 1.0 else 2.0 - phase
        c = int(round(amp * off)) + 20
        a[r, c:c + width] = True
        a[r, (c - 1) % n] = True       # keep it 8-connected across the folds
    return a

SHIFTS = ((1, 0), (0, 1), (7, 13), (-5, 3), (32, 32))

@pytest.fixture(scope='module')
def cells():
    return {'axial': straight_axial(), 'circ': straight_circ(),
            'cross': cross(), 'zigzag': zigzag()}

# INVARIANCE #

@pytest.mark.parametrize('name', ['axial', 'circ', 'cross', 'zigzag'])
def test_every_descriptor_is_shift_invariant(cells, name):
    base = motifs.describe(cells[name])
    for shift in SHIFTS:
        rolled = motifs.describe(np.roll(cells[name], shift, axis=(0, 1)))
        assert set(rolled) == set(base)
        for key, value in base.items():
            if np.isnan(value):
                assert np.isnan(rolled[key]), f'{key} moved under shift {shift}'
                continue
            assert rolled[key] == pytest.approx(value, rel=1e-9, abs=1e-9), \
                f'{key} moved under shift {shift}'

def test_skeleton_is_periodic_not_cut_at_the_seam(cells):
    """A strut crossing the seam has to skeletonize to one unbroken line, not two stubs."""
    for col in (0, 31, 61):
        skel = motifs.skeleton(straight_axial(width=6, col=col))
        # One skeleton pixel per row, and no endpoints.
        assert skel.sum() >= GRID
        nb = motifs._skeleton_neighbors(skel)
        assert (nb == 1).sum() == 0

# KNOWN ANSWERS #

def test_a_straight_strut_has_tortuosity_one(cells):
    """A path that runs straight costs the span."""
    assert motifs.tortuosity(cells['axial'], 0)['min'] == pytest.approx(1.0, abs=1e-6)
    assert motifs.tortuosity(cells['circ'], 1)['min'] == pytest.approx(1.0, abs=1e-6)

def test_a_folding_strut_costs_more_than_the_span_it_bridges(cells):
    straight = motifs.tortuosity(cells['axial'], 0)['mean']
    folded = motifs.tortuosity(cells['zigzag'], 0)['mean']
    assert straight == pytest.approx(1.0, abs=1e-6)
    assert folded > 1.1, f'zigzag read {folded:.3f}, not distinguishable from straight'

def test_longest_run_finds_a_strut_that_crosses_the_seam():
    """A run wrapping the boundary is one run, not two."""
    a = np.zeros((GRID, GRID), bool)
    a[:, 5] = True                       # spans axially
    assert motifs.longest_run(a, 0) == pytest.approx(1.0)
    # Nothing spans circumferentially
    assert motifs.longest_run(a, 1) == pytest.approx(1.0 / GRID)

def test_orientation_separates_axial_from_circumferential(cells):
    """The descriptor that names the parametric family's motif."""
    ax = motifs.orientation(motifs.skeleton(cells['axial']))
    ci = motifs.orientation(motifs.skeleton(cells['circ']))
    assert ax['orient_axial'] > 0.9
    assert ci['orient_circ'] > 0.9
    assert ax['orient_axial'] + ax['orient_circ'] + ax['orient_diag'] == pytest.approx(1.0)

def test_junctions_appear_only_where_struts_meet(cells):
    """A lone strut has no junction; a cross has them where the two cross."""
    assert motifs.describe(cells['axial'])['n_junctions'] == 0
    assert motifs.describe(cells['cross'])['n_junctions'] > 0

def test_pores_are_counted_on_the_torus(cells):
    """One axial strut leaves one void region: the void wraps around the seam."""
    assert motifs.describe(cells['axial'])['n_pores'] == 1
    # The cross cuts the void into a single wrapped region too.
    assert motifs.describe(cells['cross'])['n_pores'] >= 1

def test_thickness_reads_the_strut_half_width():
    """Inscribed radius along the centerline, so a strut of width w reads about w/2."""
    d = motifs.describe(straight_axial(width=8, col=20))
    assert d['thickness_mean'] == pytest.approx(4.0, abs=0.6)
    assert d['thickness_cv'] < 0.1

def test_describe_returns_every_declared_key(cells):
    d = motifs.describe(cells['cross'])
    for key in motifs.SHAPE_KEYS:
        assert key in d, f'{key} declared in SHAPE_KEYS but not produced'
    assert 'f_metal' in d

def test_a_direction_with_no_wrapping_path_reads_nan_not_zero():
    """
    `geom.validity` requires both wraps, so this never happens in the pool, but a silent
    0.0 would read as being perfectly straight, not having no path at all.
    """
    t = motifs.tortuosity(straight_axial(), 1)
    assert np.isnan(t['min']) and np.isnan(t['mean'])
