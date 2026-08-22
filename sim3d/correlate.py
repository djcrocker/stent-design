"""
Does the cheap 2D screen rank designs like the expensive 3D truth?

What we're looking for here is ordering: later we'll rank candidates by the 2D surrogate 
and keep the top ~10-30 for 3D validation. Our gate is rank correlation.

Reported with rho:
  - Bootstrap CIs, because n was chosen for statistical power and a bare point estimate
    would waste that;
  - Per-family rho, because the crown family is the screen's home ground and the handmade
    cells are the only test of generalization;
  - Top-K retention, because rho can look healthy while the errors concentrate exactly at
    the top of the ranking.
"""

import json
import pathlib

import numpy as np
from scipy import stats

import config

# Both metrics have to clear this
GATE_RHO = 0.7
GATE_METRICS = ('K_radial', 'eps_a_max')

# Radial support wants a stiff stent; fatigue wants low strain amplitude.
HIGHER_IS_BETTER = {'K_radial': True, 'eps_a_max': False, 'A_over_lim': False}

def load_labels(path=None):
    """Converged rows only, as a cell with no 3D result can't enter a correlation."""
    path = (config.PROJECT_ROOT / 'sim3d' / 'results' / 's6_1_labels.json') \
        if path is None else pathlib.Path(path)
    rows = json.loads(path.read_text(encoding='utf-8'))
    return [r for r in rows if r.get('converged')], rows

def spearman(x, y):
    """Rank correlation. Returns rho and the two-sided p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)

def bootstrap_ci(x, y, n_boot=10000, alpha=0.05, seed=0):
    """
    Percentile bootstrap CI for Spearman rho, resampling cells with replacement.

    Percentile rather than Fisher-z: rho is bounded and its sampling distribution is skewed
    near the ends, where the gate sits.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 4:
        return (float('nan'), float('nan'))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            draws[i] = np.nan
            continue
        draws[i] = stats.spearmanr(x[idx], y[idx]).statistic
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def topk_retention(pred, truth, k, higher_is_better=True):
    """Fraction of the true top-k that the surrogate's own top-k captures."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if k <= 0 or k > len(pred):
        raise ValueError(f'k={k} outside 1..{len(pred)}')
    order = -1 if higher_is_better else 1
    best_pred = set(np.argsort(order * pred, kind='stable')[:k].tolist())
    best_truth = set(np.argsort(order * truth, kind='stable')[:k].tolist())
    return len(best_pred & best_truth) / k

def retention_curve(pred, truth, higher_is_better=True):
    """Retention at every k."""
    return [(k, topk_retention(pred, truth, k, higher_is_better))
            for k in range(1, len(pred) + 1)]

def analyse_metric(rows, metric, n_boot=10000, seed=0):
    """rho, CI, p, n and retention for one metric across the given rows."""
    x = [r[f'{metric}_2D'] for r in rows]
    y = [r[f'{metric}_3D'] for r in rows]
    rho, p = spearman(x, y)
    lo, hi = bootstrap_ci(x, y, n_boot=n_boot, seed=seed)
    hib = HIGHER_IS_BETTER[metric]
    return {'metric': metric, 'n': len(rows), 'rho': rho, 'p': p,
            'ci95': [lo, hi], 'higher_is_better': hib,
            'retention': {str(k): topk_retention(x, y, k, hib)
                          for k in (5, 10, 20, 30) if k <= len(rows)},
            'gates': metric in GATE_METRICS,
            'passes': (metric in GATE_METRICS) and rho >= GATE_RHO}

def analyze(path=None, metrics=('K_radial', 'eps_a_max', 'A_over_lim'), n_boot=10000, seed=0):
    """Overall, per family, and gate verdict."""
    rows, all_rows = load_labels(path)
    out = {'n_converged': len(rows), 'n_total': len(all_rows),
           'gate_rho': GATE_RHO, 'gate_metrics': list(GATE_METRICS),
           'overall': {}, 'by_family': {}}
    for m in metrics:
        out['overall'][m] = analyse_metric(rows, m, n_boot, seed)

    families = sorted({r['family'] for r in rows})
    for fam in families:
        sub = [r for r in rows if r['family'] == fam]
        if len(sub) < 4:
            out['by_family'][fam] = {'n': len(sub), 'note': 'too few cells for a rho'}
            continue
        out['by_family'][fam] = {m: analyse_metric(sub, m, n_boot, seed) for m in metrics}

    gated = [out['overall'][m] for m in GATE_METRICS if m in out['overall']]
    out['gate_passed'] = bool(gated) and all(g['passes'] for g in gated)
    # A pass whose CI reaches below the threshold is not a settled pass.
    out['gate_ci_clears'] = bool(gated) and all(g['ci95'][0] >= GATE_RHO for g in gated)
    return out

def write(result, path=None):
    path = (config.PROJECT_ROOT / 'sim3d' / 'results' / 's6_2_correlation.json') \
        if path is None else pathlib.Path(path)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return path
