"""The Pareto plane."""

import numpy as np
import pytest

from screen import pareto

def rows(pairs, key='A_over_lim'):
    return [{'K_radial': k, key: v} for k, v in pairs]

# THE CONDITIONAL FRONTIER #

def test_each_band_reports_its_own_best_and_where_it_sits():
    got = pareto.band_best(rows([(10, 0.5), (40, 0.2), (120, 0.9), (140, 0.3)]), bands=(0.0, 50.0, 150.0))
    assert [b['n'] for b in got] == [2, 2]
    assert got[0]['A_over_lim'] == pytest.approx(0.2)
    assert got[0]['K_radial'] == pytest.approx(40)
    assert got[1]['A_over_lim'] == pytest.approx(0.3)

def test_an_empty_band_reports_no_value_rather_than_a_nan():
    got = pareto.band_best(rows([(10, 0.5)]), bands=(0.0, 50.0, 150.0))
    assert got[1]['n'] == 0 and 'A_over_lim' not in got[1]

def test_a_band_is_half_open_so_no_cell_is_counted_twice():
    got = pareto.band_best(rows([(50, 0.4)]), bands=(0.0, 50.0, 150.0))
    assert got[0]['n'] == 0 and got[1]['n'] == 1

# THE FRONT #

def test_the_front_keeps_only_what_nothing_dominates():
    population = rows([(100, 0.2), (150, 0.3), (200, 0.1), (300, 0.4)])
    assert pareto.front(population) == [2, 3]

def test_the_front_comes_back_ordered_by_stiffness():
    got = pareto.front(rows([(300, 0.4), (200, 0.1)]))
    assert got == [1, 0]

def test_an_empty_population_has_an_empty_front():
    assert pareto.front([]) == []

# THE SHARED MEASUREMENT #

def test_every_population_is_measured_on_the_binary_cell():
    """
    The campaign reports K_radial on the delivered density field, while
    every other population is a binary cell.
    """
    import json

    from geom.cell import UnitCell
    from screen import motifs
    from sim2d import label as label_mod

    simp = [c for c in motifs.simp_cells() if c['name'] == 'simp_k200']
    assert len(simp) == 1
    got = label_mod.label(UnitCell(simp[0]['field']), check_validity=False)
    k = dict(zip(__import__('config').OBJECTIVE_KEYS, got.y))['K_radial']
    record = json.loads((pareto.RESULTS_DIR.parent.parent / 'baseline' / 'results' / 's10_1_prescribed_screened.json').read_text(encoding='utf-8'))
    row = next(r for r in record['targets'] if r['k_target'] == 200.0)
    assert k == pytest.approx(row['binary']['K_radial'], rel=1e-6)
    assert k != pytest.approx(row['K_radial'], rel=1e-3)

def test_an_invalid_cell_is_labeled_rather_than_dropped():
    from geom.cell import UnitCell
    from screen import motifs

    unscreened = [c for c in motifs.simp_cells() if not c['screened']]
    assert unscreened
    got = pareto.label_rows([UnitCell(c['field']) for c in unscreened[:1]], progress=False)
    assert got[0]['valid'] is False and got[0]['reasons']
    assert np.isfinite(got[0]['K_radial']) and got[0]['K_radial'] > 0
