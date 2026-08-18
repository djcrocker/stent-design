"""
is_valid must accept good cells and reject each broken kind for the right reason.

Cells come from geom.handmade so this suite and the S2.3 contact sheet cannot drift apart.
"""

import pytest

import config
from geom import handmade as H
from geom import validity as V


@pytest.mark.parametrize('name', list(H.VALID_CELLS), ids=list(H.VALID_CELLS))
def test_valid_cells_pass(name):
    ok, reasons = V.is_valid(H.VALID_CELLS[name]())
    assert ok, f'{name} rejected for {reasons}'

@pytest.mark.parametrize('name', list(H.BROKEN_CELLS), ids=list(H.BROKEN_CELLS))
def test_broken_cells_fail_for_the_right_reason(name):
    build, expected = H.BROKEN_CELLS[name]
    ok, reasons = V.is_valid(build())
    assert not ok
    assert expected in reasons, f'{name}: expected {expected}, got {reasons}'

def test_reason_codes_match_the_validity_module():
    """The registry's expected codes must be real codes, not typos."""
    known = {V.EMPTY, V.TOO_SPARSE, V.TOO_DENSE, V.THIN_FEATURE,
             V.DISCONNECTED, V.NO_WRAP_CIRC, V.NO_WRAP_AXIAL}
    assert {code for _, code in H.BROKEN_CELLS.values()} <= known

# CHECKER BEHAVIOR #
def test_wraps_both_ways_on_a_good_cell():
    assert V.wraps(H.diamond().to_array()) == (True, True)

def test_wrapping_is_directional():
    """Circumferential bands wrap one way only; axial bars the other."""
    assert V.wraps(H.bands_circ_only().to_array()) == (True, False)
    assert V.wraps(H.bars_axial_only().to_array()) == (False, True)

def test_metrics_are_reported_even_when_valid():
    m = V.check(H.diamond()).metrics
    assert m['n_components'] == 1
    assert m['thin_fraction'] == pytest.approx(0.0)
    assert config.F_METAL_MIN < m['f_metal'] < config.F_METAL_MAX

def test_result_unpacks_as_bool_and_reasons():
    ok, reasons = V.check(H.diamond())
    assert ok is True and reasons == []

def test_reasons_accumulate():
    """A cell can break several rules at once; all of them are reported."""
    _, reasons = V.is_valid(H.tiny_blob())
    assert {V.NO_WRAP_CIRC, V.NO_WRAP_AXIAL, V.TOO_SPARSE} <= set(reasons)

def test_wrap_isolating_cells_stay_above_the_sparse_guard():
    """bands/bars must fail only on wrapping, or they would pass for the wrong reason."""
    for build in (H.bands_circ_only, H.bars_axial_only):
        _, reasons = V.is_valid(build())
        assert V.TOO_SPARSE not in reasons
        assert len(reasons) == 1
