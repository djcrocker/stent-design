"""Cleanup must repair what is repairable and refuse otherwise."""

import numpy as np
import pytest

from geom import cleanup, handmade as H, validity as V

REPAIRABLE = ['island']
UNFIXABLE = ['empty', 'tiny blob', 'circ bands only', 'axial bars only',
             'thin grid (2 px)', 'nearly solid']

@pytest.mark.parametrize('name', list(H.VALID_CELLS), ids=list(H.VALID_CELLS))
def test_valid_cells_pass_through_untouched(name):
    """A repair op has nothing to do on a cell that needs no repair."""
    cell = H.VALID_CELLS[name]()
    result = cleanup.clean(cell)
    assert result.fixed
    assert result.change_fraction == 0.0
    assert np.array_equal(result.cell.to_array(), cell.to_array())
    assert result.actions == []

@pytest.mark.parametrize('name', REPAIRABLE)
def test_repairable_cells_come_out_valid(name):
    build, _ = H.BROKEN_CELLS[name]
    result = cleanup.clean(build())
    assert result.fixed
    ok, reasons = V.is_valid(result.cell)
    assert ok, reasons
    assert result.change_fraction > 0

@pytest.mark.parametrize('name', UNFIXABLE)
def test_unfixable_cells_are_refused_honestly(name):
    """No cell, fixed=False, and a verdict saying why."""
    build, _ = H.BROKEN_CELLS[name]
    result = cleanup.clean(build())
    assert not result.fixed
    assert result.cell is None
    assert result.validity.reasons

def test_cleanup_never_returns_an_invalid_cell():
    """If a cell comes back, it passes validity."""
    for build in list(H.VALID_CELLS.values()) + [b for b, _ in H.BROKEN_CELLS.values()]:
        result = cleanup.clean(build())
        if result.cell is not None:
            assert V.is_valid(result.cell)[0]

def test_idempotent():
    """Cleaning a repaired cell changes nothing further."""
    first = cleanup.clean(H.island())
    second = cleanup.clean(first.cell)
    assert second.fixed
    assert second.change_fraction == 0.0
    assert np.array_equal(first.cell.to_array(), second.cell.to_array())

def test_wrapping_is_never_manufactured():
    """
    Cleanup shouldn't dilate a non-wrapping cell into a wrapping one.

    Doing so would invent a load path the generator never produced and inflate the valid-generation rate.
    """
    for name in ('circ bands only', 'axial bars only'):
        build, _ = H.BROKEN_CELLS[name]
        assert not cleanup.clean(build()).fixed

def test_result_unpacks_as_cell_and_fixed():
    cell, fixed = cleanup.clean(H.diamond())
    assert fixed and cell is not None

def test_actions_are_recorded_for_real_repairs():
    result = cleanup.clean(H.island())
    assert any('island' in a for a in result.actions)

# BATCH REPORT #
def test_summarize_reports_rate_and_effort_together():
    results = ([cleanup.clean(b()) for b in H.VALID_CELLS.values()]
               + [cleanup.clean(b()) for b, _ in H.BROKEN_CELLS.values()])
    s = cleanup.summarize(results)
    assert s['n'] == len(results)
    assert s['n_fixed'] == 6                      # 5 valid + the repaired island
    assert s['valid_rate'] == pytest.approx(0.5)
    assert 0.0 < s['mean_change'] < s['max_change']
    assert s['failure_reasons']

def test_summarize_change_stats_cover_fixed_results_only():
    """Effort spent on hopeless samples says nothing about the ones that survived."""
    only_unfixable = [cleanup.clean(H.BROKEN_CELLS[n][0]()) for n in UNFIXABLE]
    s = cleanup.summarize(only_unfixable)
    assert s['n_fixed'] == 0
    assert s['mean_change'] == 0.0
    assert s['valid_rate'] == 0.0

def test_summarize_untouched_rate_separates_clean_from_repaired():
    """Samples needing zero repair are the honest measure of generator quality."""
    s = cleanup.summarize([cleanup.clean(b()) for b in H.VALID_CELLS.values()])
    assert s['untouched_rate'] == 1.0
    s2 = cleanup.summarize([cleanup.clean(H.island())])
    assert s2['untouched_rate'] == 0.0
