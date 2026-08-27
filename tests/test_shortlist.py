"""Non-dominated sorting and crowding-distance selection."""

import numpy as np
import pytest

from screen import shortlist

def _obj(pairs):
    """(K_radial to maximize, A_over_lim to minimize)."""
    return np.array(pairs, float)

def test_module_does_not_import_torch():
    """The screening tier stays torch-free."""
    import sys
    assert 'torch' not in sys.modules

def test_a_strictly_better_design_dominates():
    # B has more stiffness and less over-limit area, so A is dominated.
    layers = shortlist.nondominated_layers(_obj([[100, 0.30], [200, 0.10]]))
    assert list(layers[0]) == [1]
    assert list(layers[1]) == [0]

def test_a_trade_off_is_not_dominated():
    """Stiffer but worse on fatigue is a genuine trade, so both stay on the front."""
    layers = shortlist.nondominated_layers(_obj([[100, 0.10], [200, 0.30]]))
    assert len(layers) == 1
    assert sorted(layers[0]) == [0, 1]

def test_layers_partition_every_design_exactly_once():
    rng = np.random.default_rng(0)
    obj = np.column_stack([10 ** rng.uniform(1, 2.5, 40), rng.uniform(0.02, 0.6, 40)])
    layers = shortlist.nondominated_layers(obj)
    flat = np.concatenate(layers)
    assert sorted(flat.tolist()) == list(range(40))

def test_later_layers_are_dominated_by_earlier_ones():
    rng = np.random.default_rng(1)
    obj = np.column_stack([10 ** rng.uniform(1, 2.5, 30), rng.uniform(0.02, 0.6, 30)])
    layers = shortlist.nondominated_layers(obj)
    for i in layers[1]:
        # Every second-layer design must be dominated by something in the first.
        dominated = any(obj[j, 0] >= obj[i, 0] and obj[j, 1] <= obj[i, 1]
                        and (obj[j, 0] > obj[i, 0] or obj[j, 1] < obj[i, 1])
                        for j in layers[0])
        assert dominated

def test_crowding_keeps_the_extremes():
    """Boundary designs score infinity so the ends of the trade-off are never trimmed."""
    obj = _obj([[100, 0.10], [200, 0.20], [300, 0.30], [400, 0.40]])
    d = shortlist.crowding_distance(obj, np.arange(4))
    assert np.isinf(d[0]) and np.isinf(d[3])
    assert np.all(np.isfinite(d[1:3]))

def test_crowding_prefers_isolated_designs():
    """Two clustered designs and one isolated: the isolated one must score higher."""
    obj = _obj([[100, 0.10], [101, 0.101], [400, 0.40], [1000, 0.90]])
    d = shortlist.crowding_distance(obj, np.arange(4))
    assert d[2] > d[1]

def test_select_returns_exactly_k():
    rng = np.random.default_rng(2)
    obj = np.column_stack([10 ** rng.uniform(1, 2.5, 60), rng.uniform(0.02, 0.6, 60)])
    chosen, layers = shortlist.select(obj, k=17)
    assert len(chosen) == 17
    assert len(layers) == 17
    assert len(set(chosen.tolist())) == 17

def test_select_takes_whole_early_layers_before_later_ones():
    rng = np.random.default_rng(3)
    obj = np.column_stack([10 ** rng.uniform(1, 2.5, 50), rng.uniform(0.02, 0.6, 50)])
    all_layers = shortlist.nondominated_layers(obj)
    k = len(all_layers[0]) + 2
    chosen, ranks = shortlist.select(obj, k=k)
    # The entire first layer survives; only the overflow comes from the second.
    assert set(all_layers[0].tolist()) <= set(chosen.tolist())
    assert (ranks == 1).sum() == len(all_layers[0])
    assert (ranks == 2).sum() == 2

def test_select_returns_everything_when_k_exceeds_the_pool():
    obj = _obj([[100, 0.30], [200, 0.10], [150, 0.20]])
    chosen, ranks = shortlist.select(obj, k=99)
    assert len(chosen) == 3
    assert ranks.min() >= 1

def test_k_radial_is_compared_on_a_log_scale():
    """
    Two decades of range. On a linear scale crowding distance would be almost entirely
    about the high end and the shortlist would cluster there.
    """
    obj = _obj([[10, 0.1], [100, 0.2], [1000, 0.3]])
    f = shortlist._as_minimization(obj)
    gaps = np.diff(f[:, 0])
    assert gaps[0] == pytest.approx(gaps[1])

def test_maximizing_k_radial_flips_its_sign():
    f = shortlist._as_minimization(_obj([[100, 0.1], [200, 0.1]]))
    # Higher stiffness must become a SMALLER minimization value.
    assert f[1, 0] < f[0, 0]

def _obj3(triples):
    """(K_radial max, A_over_lim min, f_metal min)."""
    return np.array(triples, float)

def test_a_lighter_design_at_equal_performance_dominates():
    """Less metal for the same numbers is better."""
    layers = shortlist.nondominated_layers(_obj3([[200, 0.10, 0.48], [200, 0.10, 0.30]]))
    assert list(layers[0]) == [1]
    assert list(layers[1]) == [0]

def test_a_lightweight_design_survives_a_stiffer_heavier_one():
    dense = [500, 0.05, 0.48]
    light = [300, 0.05, 0.30]
    two = shortlist.nondominated_layers(np.array([dense, light], float)[:, :2])
    assert list(two[0]) == [0] and list(two[1]) == [1]      # dense wins on two
    three = shortlist.nondominated_layers(_obj3([dense, light]))
    assert sorted(three[0].tolist()) == [0, 1]              # both survive on three

def test_f_metal_is_passed_through_unscaled():
    """Only K_radial is log-scaled; the fractions are already comparable."""
    f = shortlist._as_minimization(_obj3([[100, 0.20, 0.30], [100, 0.20, 0.45]]))
    assert f[0, 2] == pytest.approx(0.30)
    assert f[1, 2] == pytest.approx(0.45)

def test_selection_still_returns_k_with_three_objectives():
    rng = np.random.default_rng(11)
    obj = np.column_stack([10 ** rng.uniform(1, 2.5, 80), rng.uniform(0.02, 0.6, 80), rng.uniform(0.10, 0.50, 80)])
    chosen, ranks = shortlist.select(obj, k=24)
    assert len(chosen) == 24 and len(set(chosen.tolist())) == 24
    assert ranks.min() >= 1

def test_saved_cells_are_in_the_same_order_as_the_shortlist_entries():
    """Row i of the cells file should be slot i+1 of the report."""
    import json

    report = shortlist.RESULTS_DIR / 's9_2_shortlist.json'
    cells = shortlist.RESULTS_DIR / 's9_2_shortlist_cells.npz'
    if not (report.exists() and cells.exists()):
        pytest.skip('shortlist has not been built')

    rows = json.loads(report.read_text(encoding='utf-8'))['shortlist']
    blob = np.load(cells)
    assert [int(p) for p in blob['pool_index']] == [r['pool_index'] for r in rows]

    fields = np.unpackbits(blob['fields'], count=int(np.prod(blob['shape']))).reshape(tuple(blob['shape'])).astype(bool)
    for i, r in enumerate(rows):
        assert fields[i].mean() == pytest.approx(r['f_metal'], abs=1e-9)
