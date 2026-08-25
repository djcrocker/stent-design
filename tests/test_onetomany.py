"""The translation-invariant distinctness measure."""

import numpy as np
import pytest

from screen import onetomany

def _bars(axis, n=64, width=6, count=2):
    """Straight bars along one axis."""
    arr = np.zeros((n, n), bool)
    for j in range(count):
        start = int(j * n / count)
        sl = slice(start, start + width)
        if axis == 0:
            arr[sl, :] = True
        else:
            arr[:, sl] = True
    return arr

def test_module_does_not_import_torch():
    import sys
    assert 'torch' not in sys.modules

def test_descriptor_is_exactly_translation_invariant():
    """The descriptor should be invariant to translation."""
    arr = _bars(0)
    shifted = np.roll(np.roll(arr, 23, axis=0), 41, axis=1)
    d = onetomany.descriptor(np.stack([arr, shifted]))
    assert onetomany.pairwise(d)[0, 1] == pytest.approx(0.0, abs=1e-5)

def test_shift_floor_is_zero():
    arrs = np.stack([_bars(0), _bars(1), _bars(0, width=10)])
    mean, mx = onetomany.shift_floor(arrs, n_shifts=6)
    assert mean == pytest.approx(0.0, abs=1e-5)
    assert mx == pytest.approx(0.0, abs=1e-5)

def test_different_orientations_are_far_apart():
    """Bars along one axis versus the other are different designs."""
    d = onetomany.descriptor(np.stack([_bars(0), _bars(1)]))
    assert onetomany.pairwise(d)[0, 1] > 0.5

def test_identical_designs_have_zero_distance():
    arr = _bars(0)
    d = onetomany.descriptor(np.stack([arr, arr.copy()]))
    assert onetomany.pairwise(d)[0, 1] == pytest.approx(0.0, abs=1e-6)

def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    rng = np.random.default_rng(0)
    arrs = rng.random((5, 64, 64)) > 0.6
    d = onetomany.pairwise(onetomany.descriptor(arrs))
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6)

def test_descriptor_ignores_density_alone():
    thin = _bars(0, width=4)
    thick = _bars(0, width=5)
    d_struct = onetomany.pairwise(onetomany.descriptor(np.stack([thin, thick])))[0, 1]
    flipped = _bars(1, width=4)
    d_orient = onetomany.pairwise(onetomany.descriptor(np.stack([thin, flipped])))[0, 1]
    assert d_struct < d_orient

def test_farthest_first_returns_the_requested_count_without_repeats():
    rng = np.random.default_rng(1)
    desc = onetomany.descriptor(rng.random((20, 64, 64)) > 0.6)
    picked = onetomany.farthest_first(desc, 6)
    assert len(picked) == 6
    assert len(set(picked.tolist())) == 6

def test_farthest_first_prefers_spread_over_clustering():
    """
    A gallery of near-copies would misrepresent the variety, so exemplars are chosen to
    span the set rather than taken in order.
    """
    base = _bars(0)
    cluster = [np.roll(base, i, axis=1) for i in range(6)]     # all the same design
    odd = _bars(1)
    desc = onetomany.descriptor(np.stack(cluster + [odd]))
    picked = onetomany.farthest_first(desc, 2)
    # The one different design must be picked over another copy.
    assert 6 in picked.tolist()

def test_farthest_first_caps_at_the_pool_size():
    desc = onetomany.descriptor(np.stack([_bars(0), _bars(1)]))
    assert len(onetomany.farthest_first(desc, 10)) == 2
