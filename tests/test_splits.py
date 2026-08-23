"""Splits by design group."""

import numpy as np
import pytest

from diffusion import splits
from geom import handmade, reference

def _cells(n=6):
    base = [f() for f in handmade.VALID_CELLS.values()] + [reference.build()]
    return np.stack([c.to_array() for c in base[:n]])

def test_signature_is_unchanged_by_translation():
    """
    The cell is a torus, so a shifted copy is the same design and can't be allowed to
    land in a different split. |FFT| is translation invariant, which is why it is the key.
    """
    arr = reference.build().to_array()
    shifted = np.roll(np.roll(arr, 17, axis=0), 29, axis=1)
    a, b = splits.signature(np.stack([arr, shifted]))
    assert (a == b).all()

def test_identical_cells_share_a_group():
    arr = reference.build().to_array()
    ids = splits.group_ids(np.stack([arr, arr.copy(), np.roll(arr, 5, axis=1)]))
    assert len(set(ids.tolist())) == 1

def test_different_designs_get_different_groups():
    arrs = _cells()
    ids = splits.group_ids(arrs)
    assert len(set(ids.tolist())) == len(arrs)

def test_no_design_group_straddles_a_split():
    """A duplicate in train and val makes val loss meaningless."""
    arrs = np.concatenate([_cells(), _cells(), _cells()])
    split, groups = splits.make_splits(arrs, seed=1)
    assert splits.leakage_check(split, groups) == []

def test_leakage_check_catches_a_deliberate_leak():
    groups = np.array([0, 0, 1, 1])
    split = np.array(['train', 'val', 'test', 'test'])
    assert splits.leakage_check(split, groups) == [0]

def test_fractions_are_respected_within_tolerance():
    rng = np.random.default_rng(0)
    # Many distinct designs so group granularity doesn't dominate the proportions.
    arrs = rng.random((400, 16, 16)) > 0.5
    split, _ = splits.make_splits(arrs, fractions=(0.8, 0.1, 0.1), seed=2)
    for name, want in zip(splits.SPLIT_NAMES, (0.8, 0.1, 0.1)):
        assert (split == name).mean() == pytest.approx(want, abs=0.03)

def test_every_cell_lands_in_exactly_one_split():
    arrs = np.concatenate([_cells(), _cells()])
    split, _ = splits.make_splits(arrs, seed=3)
    assert len(split) == len(arrs)
    assert set(split.tolist()) <= set(splits.SPLIT_NAMES)

def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        splits.make_splits(_cells(), fractions=(0.5, 0.2, 0.2))

def test_splits_are_deterministic_for_a_seed():
    arrs = np.concatenate([_cells(), _cells()])
    a, _ = splits.make_splits(arrs, seed=4)
    b, _ = splits.make_splits(arrs, seed=4)
    assert (a == b).all()
