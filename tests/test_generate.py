"""Target construction and support checking."""

import numpy as np
import pandas as pd
import pytest

from diffusion import generate

def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = 10 ** rng.uniform(0.5, 2.6, n)
    return pd.DataFrame({
        'K_radial': k,
        # A_over_lim falls as K_radial rises, so conditional and global medians differ.
        'A_over_lim': np.clip(0.9 - 0.25 * np.log10(k) + rng.normal(0, 0.05, n), 0.01, 0.95),
        'eps_a_max': rng.uniform(0.02, 0.12, n),
        'f_metal': rng.uniform(0.2, 0.5, n),
    })

def test_module_does_not_import_torch():
    """The screening phase has to stay torch-free."""
    import sys
    assert 'torch' not in sys.modules

def test_support_counts_a_neighborhood_not_exact_matches():
    frame = _frame()
    n, mask = generate.support(frame, {'K_radial': 100.0})
    assert n > 1
    assert mask.sum() == n
    # Everything counted really is near the target, on a log scale.
    near = frame.loc[mask, 'K_radial'].to_numpy()
    assert np.all(np.abs(np.log10(near) - 2.0) <= np.log10(1.25) + 1e-9)

def test_support_shrinks_as_the_ask_gets_harder():
    frame = _frame()
    easy, _ = generate.support(frame, {'K_radial': 100.0})
    hard, _ = generate.support(frame, {'K_radial': 300.0, 'A_over_lim': 0.10})
    assert hard < easy

def test_unspecified_components_come_from_the_conditional_median():
    """
    Pinning them to the global median would ask for a combination the data doesn't contain:
    high K_radial co-occurs with particular strain levels, and the model would then be asked
    to satisfy a vector no design realizes.
    """
    frame = _frame()
    high, _ = generate.complete_target(frame, {'K_radial': 300.0})
    low, _ = generate.complete_target(frame, {'K_radial': 10.0})
    assert high['A_over_lim'] < low['A_over_lim']

def test_completion_preserves_what_was_asked():
    frame = _frame()
    full, _ = generate.complete_target(frame, {'K_radial': 200.0, 'A_over_lim': 0.15})
    assert full['K_radial'] == pytest.approx(200.0)
    assert full['A_over_lim'] == pytest.approx(0.15)
    assert set(full) == set(generate.Y_KEYS)

def test_completion_falls_back_to_the_whole_frame_when_support_is_tiny():
    frame = _frame()
    full, n = generate.complete_target(frame, {'K_radial': 1e6})
    assert n < 8
    assert all(np.isfinite(v) for v in full.values())

def test_build_targets_flags_thin_support():
    frame = _frame()
    ladder = ({'name': 'ok', 'K_radial': 100.0},
              {'name': 'impossible', 'K_radial': 1e9})
    out = generate.build_targets(frame, ladder=ladder)
    assert out[0]['well_supported'] is True
    assert out[1]['well_supported'] is False

def test_build_targets_records_asked_separately_from_completed():
    """
    The distinction matters when reading results, as a component we never asked for can't be
    called a conditioning miss.
    """
    frame = _frame()
    out = generate.build_targets(frame, ladder=({'name': 't', 'K_radial': 150.0},))
    assert set(out[0]['asked']) == {'K_radial'}
    assert set(out[0]['target']) == set(generate.Y_KEYS)

def test_default_ladder_spans_the_desirable_direction():
    """High K_radial with low A_over_lim is the corner Stage 10 cares about."""
    names = [t['name'] for t in generate.TARGET_LADDER]
    assert 'K300_A10' in names
    assert any(n.startswith('control') for n in names)
    ks = [t['K_radial'] for t in generate.TARGET_LADDER if 'K_radial' in t]
    assert max(ks) / min(ks) >= 10
