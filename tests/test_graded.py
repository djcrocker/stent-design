"""Assembling cells into a stent that varies along its length."""

import numpy as np
import pytest

import config
from diffusion import graded, interface

W = 6   # Strut width in px; MIN_FEATURE_MM is 4.07 px

def _cell(starts, band=6, rung=None):
    """A minimal stent-like cell: axial struts tied by one circumferential ring."""
    n = config.GRID_N
    a = np.zeros((n, n), bool)
    for c in starts:
        a[:, c:c + W] = True
    a[n // 2:n // 2 + band] = True
    if rung is not None:
        row, c0, c1 = rung
        a[row:row + W, c0:c1] = True
    return a

def test_module_does_not_import_torch():
    import sys
    assert 'torch' not in sys.modules

def test_gradient_is_monotone_and_spans_the_range():
    """The ramp is what the whole step is testing, so it has to actually ramp."""
    import pandas as pd

    n = 400
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'K_radial': 10 ** rng.uniform(1.5, 2.8, n),
        'eps_a_max': rng.uniform(0.01, 0.09, n),
        'A_over_lim': rng.uniform(0.02, 0.6, n),
        'f_metal': rng.uniform(0.15, 0.48, n),
    })
    ramp = graded.gradient(frame, n_cells=5, key='K_radial', lo=120.0, hi=380.0)
    asked = [t['asked']['K_radial'] for t in ramp]
    assert asked == sorted(asked)
    assert asked[0] == pytest.approx(120.0) and asked[-1] == pytest.approx(380.0)
    # Every completed target carries all four y components, as S9.1 requires.
    for t in ramp:
        assert set(t['target']) == set(frame.columns)

def test_pick_next_requires_a_join_not_just_validity():
    """A valid cell that doesn't connect to the one below is useless here."""
    below = _cell([4, 24, 44])
    mismatched = _cell([14, 34, 54])        # Valid, but no metal at the same angles
    matched = _cell([4, 24, 44], rung=(14, 4, 30))
    idx, rep = graded.pick_next([mismatched, matched], below)
    assert idx == 1
    assert rep['joined']

def test_pick_next_only_needs_validity_at_the_first_position():
    a = _cell([4, 24, 44])
    idx, rep = graded.pick_next([a], None)
    assert idx == 0 and rep is None

def test_pick_next_reports_failure_rather_than_guessing():
    """No candidate joins: the caller has to know, not be handed a broken chain."""
    below = _cell([4, 24, 44])
    idx, rep = graded.pick_next([_cell([14, 34, 54])], below)
    assert idx is None and rep is None

def test_a_chain_of_compatible_cells_is_one_connected_stent():
    """Every interface joined and the whole stack one component, which are not the same."""
    chain = [_cell([4, 24, 44]),
             _cell([4, 24, 44], rung=(14, 4, 30)),
             _cell([4, 24, 44], rung=(44, 24, 50))]
    rep = interface.interface_report(chain)
    assert rep['connected']
    assert rep['wrap_circ'] and rep['wrap_axial']
    assert min(rep['crossings']) > 0

def test_one_bad_interface_shows_up_in_the_stack():
    """A break can't be averaged away by the other interfaces holding."""
    chain = [_cell([4, 24, 44]), _cell([4, 24, 44]), _cell([14, 34, 54])]
    rep = interface.interface_report(chain)
    assert min(rep['crossings']) == 0
    assert not rep['joined']
