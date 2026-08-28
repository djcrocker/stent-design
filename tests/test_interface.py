"""Stacking two different cells into one stent."""

import numpy as np
import pytest

import config
from diffusion import interface

def _ring(rows, cols, band, cols_on):
    """A minimal stent-like cell: axial struts at `cols_on`, tied by one circumferential ring."""
    a = np.zeros((rows, cols), bool)
    a[:, cols_on] = True                 # Axial struts: the stack wraps axially
    a[band] = True                       # One full ring: connects them, wraps circumferentially
    return a

def test_module_does_not_import_torch():
    """The analysis half stays torch-free."""
    import sys
    assert 'torch' not in sys.modules

def test_band_is_the_top_rows():
    a = np.zeros((8, 8), bool)
    a[0:2] = True
    assert interface.band_of(a, 2).all()
    assert not interface.band_of(a, 3)[2].any()

def test_stack_puts_the_first_cell_on_top():
    a = np.zeros((4, 4), bool); a[0] = True
    b = np.zeros((4, 4), bool); b[3] = True
    s = interface.stack([a, b])
    assert s.shape == (8, 4)
    assert s[0].all() and s[7].all()

def test_a_shared_signature_makes_two_different_cells_join():
    """
    Two cells that differ everywhere except a shared top band should still carry load across
    the interface, because each cell's own periodicity already connects its bottom to that band.
    """
    n = 16
    a = _ring(n, n, 3, [2, 7, 12])
    b = _ring(n, n, 3, [2, 7, 12])
    b[3:9, 4] = True                           # b differs inside, but stays attached to
                                               # the ring at row 3
    assert not np.array_equal(a, b)
    assert np.array_equal(interface.band_of(a, 3), interface.band_of(b, 3))

    rep = interface.interface_report([a, b])
    assert rep['joined']
    assert rep['min_crossings'] >= 3

def test_mismatched_signatures_do_not_join():
    """Metal on both sides at different angles is two tubes touching, not one stent."""
    n = 16
    a = _ring(n, n, 3, [1, 5])
    b = _ring(n, n, 3, [8, 12])
    rep = interface.interface_report([a, b])
    assert rep['min_crossings'] == 0
    assert not rep['joined']

def test_the_report_sees_the_stack_as_a_torus():
    """A stack is periodic axially too."""
    n = 16
    a = _ring(n, n, 3, [4, 11])
    rep = interface.interface_report([a, a])
    assert rep['wrap_axial'] and rep['wrap_circ']
    assert len(rep['crossings']) == 2          # one interface per cell boundary

def test_choose_sigma_rejects_an_empty_band():
    """
    A band with no metal has no axial load path, and every cell built on it fails `no_wrap_axial`.
    """
    n = config.GRID_N
    empty = np.zeros((n, n), bool)
    dense = _ring(n, n, 4, list(range(0, n, 3)))
    sigma, info = interface.choose_sigma([empty, dense], rows=4, valid_only=False)
    assert sigma.any()
    assert info['band_density'] > 0

def test_choose_sigma_prefers_the_population_density():
    n = config.GRID_N
    sparse = _ring(n, n, 4, [1])
    mid = _ring(n, n, 4, list(range(0, n, 4)))
    heavy = np.ones((n, n), bool)
    sigma, info = interface.choose_sigma([sparse, mid, heavy], rows=4, valid_only=False)
    assert info['band_density'] == pytest.approx(interface.band_of(mid, 4).mean())

def test_choose_sigma_filters_bands_that_cannot_carry_a_load_path():
    """Coverage below MIN_SIGMA_COVERAGE is excluded before scoring, not penalized."""
    n = config.GRID_N
    thin = _ring(n, n, 4, [3])
    thin[:4] = False
    thin[:4, 3] = True
    wide = _ring(n, n, 4, list(range(0, n, 5)))
    sigma, info = interface.choose_sigma([thin, wide], rows=4, valid_only=False)
    assert info['column_coverage'] >= interface.MIN_SIGMA_COVERAGE
