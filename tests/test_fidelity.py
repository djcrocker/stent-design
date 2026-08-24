"""S8.4: conditioning-fidelity statistics.

These run in the DEFAULT suite, unlike tests/test_diffusion.py: nothing here imports torch,
which is the whole point of splitting fidelity into a sampling phase and an analysis phase.
"""

import numpy as np
import pandas as pd
import pytest

from diffusion import fidelity


def test_module_does_not_import_torch():
    """The analysis phase must stay torch-free or it aborts on the OpenMP clash."""
    import sys
    assert 'torch' not in sys.modules


def test_rankdata_matches_scipy_including_ties():
    scipy_stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.integers(0, 6, 40).astype(float)      # deliberately tie-heavy
        assert np.allclose(fidelity._rankdata(v), scipy_stats.rankdata(v))


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(1)
    a = rng.integers(0, 8, 50).astype(float)
    b = a * 0.5 + rng.normal(size=50)
    assert fidelity._spearman(a, b) == pytest.approx(
        scipy_stats.spearmanr(a, b).statistic, abs=1e-9)


def test_spearman_is_one_for_a_monotone_map():
    x = np.arange(1.0, 21.0)
    assert fidelity._spearman(x, np.exp(x / 3)) == pytest.approx(1.0)


def test_correlation_of_a_constant_is_nan_not_a_crash():
    x = np.arange(10.0)
    assert np.isnan(fidelity._pearson(x, np.ones(10)))


def _pairs(n=30, noise=0.0, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    targets, achieved = [], []
    for _ in range(n):
        t = {'K_radial': float(10 ** rng.uniform(0.5, 2.2)),
             'eps_a_max': float(rng.uniform(0.02, 0.12)),
             'A_over_lim': float(rng.uniform(0.05, 0.9)),
             'f_metal': float(rng.uniform(0.2, 0.5))}
        a = {k: v * (1 + noise * rng.normal()) + offset * (k == 'f_metal')
             for k, v in t.items()}
        targets.append(t)
        achieved.append(a)
    return targets, achieved


def test_perfect_conditioning_scores_perfectly():
    targets, achieved = _pairs(noise=0.0)
    stats = fidelity.fidelity_stats(targets, achieved, np.ones(4))
    for key in fidelity.Y_KEYS:
        assert stats[key]['spearman'] == pytest.approx(1.0)
        assert stats[key]['median_abs_err_sd'] == pytest.approx(0.0, abs=1e-9)


def test_bias_is_separated_from_tracking():
    """
    A model can follow the target's ordering perfectly and still sit at a constant offset,
    which is why landing error and bias are reported alongside the correlation.
    """
    targets, achieved = _pairs(noise=0.0, offset=0.1)
    stats = fidelity.fidelity_stats(targets, achieved, np.ones(4))
    assert stats['f_metal']['spearman'] == pytest.approx(1.0)
    assert stats['f_metal']['bias_sd'] == pytest.approx(0.1, abs=1e-6)


def test_dropped_samples_are_skipped_not_counted():
    targets, achieved = _pairs(n=20)
    achieved[3] = None
    achieved[7] = None
    stats = fidelity.fidelity_stats(targets, achieved, np.ones(4))
    assert stats['K_radial']['n'] == 18


def test_too_few_labeled_samples_reports_rather_than_crashing():
    targets, achieved = _pairs(n=3)
    stats = fidelity.fidelity_stats(targets, achieved, np.ones(4))
    assert 'note' in stats['K_radial']


def test_diversity_is_zero_when_every_sample_is_identical():
    """Guidance strong enough to collapse a target group would break the one-to-many claim."""
    field = np.random.default_rng(0).random((64, 64)) > 0.5
    fields = np.stack([field] * 8)
    which = np.zeros(8, dtype=int)
    assert fidelity.diversity(fields, which, 1) == pytest.approx(0.0, abs=1e-9)


def test_diversity_is_high_for_independent_samples():
    rng = np.random.default_rng(1)
    fields = rng.random((64, 64, 64)) > 0.5
    which = np.zeros(64, dtype=int)
    # Two independent fair coins disagree half the time.
    assert fidelity.diversity(fields, which, 1) == pytest.approx(0.5, abs=0.02)


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'K_radial': 10 ** rng.uniform(0, 2, n),
        'eps_a_max': rng.uniform(0.02, 0.12, n),
        'A_over_lim': rng.uniform(0.05, 0.9, n),
        'f_metal': rng.uniform(0.2, 0.5, n),
        'split': ['test'] * n,
    })


def test_targets_come_from_the_requested_split():
    frame = _frame()
    frame.loc[:99, 'split'] = 'train'
    targets = fidelity.pick_targets(frame, n_targets=10, split='test')
    assert len(targets) == 10
    for t in targets:
        assert set(t) == set(fidelity.Y_KEYS)


def test_probes_push_one_component_outside_the_training_range():
    frame = _frame()
    probes = fidelity.out_of_range_targets(frame)
    assert len(probes) == 3
    for probe in probes:
        key = probe['component']
        value = probe['target'][key]
        assert value > frame[key].max() or value < frame[key].min()
        # Every other component sits at the median, so a miss attributes to one axis.
        for other in fidelity.Y_KEYS:
            if other != key:
                assert probe['target'][other] == pytest.approx(frame[other].median())
