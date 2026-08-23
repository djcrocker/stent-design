"""Rebuilding the same designs at another GRID_N."""

import pytest

import config
from geom import handmade
from sim3d import resolution

def test_crown_cells_rebuild_at_the_requested_resolution():
    entry = {'family': 'crown', 'params': {'strut_width_mm': 0.1473, 'crown_amplitude': 0.25, 'n_periods': 1}}
    assert resolution.rebuild(entry, 64).n == 64
    assert resolution.rebuild(entry, 128).n == 128

def test_reference_family_rebuilds_like_a_crown_cell():
    """The reference cell is a crown geometry."""
    params = {'strut_width_mm': 0.1473, 'crown_amplitude': 0.25, 'n_periods': 1}
    a = resolution.rebuild({'family': 'crown', 'params': params}, 128)
    b = resolution.rebuild({'family': 'reference', 'params': params}, 128)
    assert (a.to_array() == b.to_array()).all()

def test_handmade_cells_rebuild_at_the_requested_resolution():
    entry = {'family': 'handmade', 'params': {'label': 'diamond w=0.25'}}
    assert resolution.rebuild(entry, 128).n == 128

def test_handmade_factories_still_default_to_config():
    """n=None should keep the old behavior so existing callers are unaffected."""
    for factory in handmade.VALID_CELLS.values():
        assert factory().n == config.GRID_N

def test_rebuild_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        resolution.rebuild({'family': 'nonsense', 'params': {}}, 64)

def test_a_finer_grid_preserves_physical_size_not_pixel_counts():
    """
    The design is the same physical object at both resolutions, so f_metal should be close
    while the array doubles. If f_metal moved a lot, we'd be comparing different designs.
    """
    entry = {'family': 'crown', 'params': {'strut_width_mm': 0.1473, 'crown_amplitude': 0.25, 'n_periods': 1}}
    coarse, fine = resolution.rebuild(entry, 64), resolution.rebuild(entry, 128)
    assert fine.to_array().shape[0] == 2 * coarse.to_array().shape[0]
    assert fine.f_metal == pytest.approx(coarse.f_metal, abs=0.06)

def test_materially_better_requires_clearing_the_coarse_upper_bound():
    metrics = {'K_radial': {'64': {'rho': 0.80, 'ci95': [0.70, 0.88]},
                            '128': {'rho': 0.83, 'ci95': [0.74, 0.90]}}}
    entry = metrics['K_radial']
    assert not (entry['128']['rho'] > entry['64']['ci95'][1])
    entry['128']['rho'] = 0.93
    assert entry['128']['rho'] > entry['64']['ci95'][1]
