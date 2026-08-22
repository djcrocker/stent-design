"""Rank correlation, bootstrap CIs, and top-K retention."""

import numpy as np
import pytest

from sim3d import correlate

def test_spearman_is_one_for_a_monotone_but_nonlinear_map():
    """Spearman shouldn't care about shape, only order."""
    x = np.arange(1, 21, dtype=float)
    rho, p = correlate.spearman(x, np.exp(x / 4))
    assert rho == pytest.approx(1.0)
    assert p < 1e-6

def test_spearman_is_minus_one_when_order_reverses():
    x = np.arange(1, 21, dtype=float)
    rho, _ = correlate.spearman(x, -x)
    assert rho == pytest.approx(-1.0)

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x + rng.normal(scale=0.6, size=60)
    rho, _ = correlate.spearman(x, y)
    lo, hi = correlate.bootstrap_ci(x, y, n_boot=800, seed=1)
    assert lo < rho < hi
    assert -1.0 <= lo < hi <= 1.0

def test_bootstrap_ci_narrows_as_n_grows():
    """n=60 was chosen over 30."""
    rng = np.random.default_rng(2)
    def width(n):
        x = rng.normal(size=n)
        y = x + rng.normal(scale=0.8, size=n)
        lo, hi = correlate.bootstrap_ci(x, y, n_boot=600, seed=3)
        return hi - lo
    assert width(200) < width(25)

def test_retention_is_perfect_when_the_ranking_is_perfect():
    pred = np.arange(20, dtype=float)
    assert correlate.topk_retention(pred, pred * 3.0, 5, higher_is_better=True) == 1.0

def test_retention_respects_the_direction_of_good():
    """eps_a_max wants the smallest values; K_radial wants the largest."""
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert correlate.topk_retention(pred, truth, 2, higher_is_better=False) == 1.0
    assert correlate.topk_retention(pred, -truth, 2, higher_is_better=False) == 0.0

def test_retention_is_one_when_k_is_everything():
    rng = np.random.default_rng(4)
    pred, truth = rng.normal(size=12), rng.normal(size=12)
    assert correlate.topk_retention(pred, truth, 12) == 1.0

def test_retention_rejects_an_out_of_range_k():
    pred = np.arange(5, dtype=float)
    for bad in (0, -1, 6):
        with pytest.raises(ValueError):
            correlate.topk_retention(pred, pred, bad)

def test_retention_curve_covers_every_k():
    pred = np.arange(8, dtype=float)
    curve = correlate.retention_curve(pred, pred)
    assert [k for k, _ in curve] == list(range(1, 9))
    assert curve[-1][1] == 1.0

def test_only_gate_metrics_can_pass_or_fail_the_gate():
    rows = [{'K_radial_2D': i, 'K_radial_3D': i,
             'A_over_lim_2D': i, 'A_over_lim_3D': i} for i in range(1, 21)]
    gated = correlate.analyse_metric(rows, 'K_radial', n_boot=200)
    reported = correlate.analyse_metric(rows, 'A_over_lim', n_boot=200)
    assert gated['gates'] and gated['passes']
    # A_over_lim correlates perfectly here and still shouldn't count toward the gate.
    assert reported['rho'] == pytest.approx(1.0)
    assert not reported['gates']
    assert not reported['passes']

def test_a_weak_correlation_fails_the_gate():
    rng = np.random.default_rng(5)
    rows = [{'K_radial_2D': float(v), 'K_radial_3D': float(w)}
            for v, w in zip(rng.normal(size=40), rng.normal(size=40))]
    assert not correlate.analyse_metric(rows, 'K_radial', n_boot=200)['passes']

def test_load_labels_drops_unconverged_cells(tmp_path):
    import json
    p = tmp_path / 'labels.json'
    p.write_text(json.dumps([{'name': 'a', 'converged': True},
                             {'name': 'b', 'converged': False}]))
    rows, all_rows = correlate.load_labels(p)
    assert [r['name'] for r in rows] == ['a']
    assert len(all_rows) == 2
