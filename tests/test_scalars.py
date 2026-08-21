"""The three 3D scalars, and the volume weighting behind A_over_lim_3D."""

import numpy as np
import pytest

import config
from geom import reference
from sim3d import loadsteps, mesh3d

def unit_cube_stack(n=4):
    """n unit cubes stacked along z: volumes are exactly 1, so answers are checkable."""
    pts = []
    for k in range(n + 1):
        pts += [[0., 0., k], [1., 0., k], [1., 1., k], [0., 1., k]]
    points = np.array(pts, dtype=float)
    hexes = np.array([[4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3,
                       4 * (k + 1), 4 * (k + 1) + 1, 4 * (k + 1) + 2, 4 * (k + 1) + 3]
                      for k in range(n)])
    return points, hexes

def test_element_volumes_of_unit_cubes_are_one():
    points, hexes = unit_cube_stack()
    assert np.allclose(loadsteps.element_volumes(points, hexes), 1.0)

def test_lumped_volumes_sum_to_total_volume():
    points, hexes = unit_cube_stack()
    lumped = loadsteps.lumped_node_volumes(points, hexes)
    assert lumped.sum() == pytest.approx(loadsteps.element_volumes(points, hexes).sum())

def test_lumped_volumes_sum_to_total_on_the_real_mesh():
    cell = reference.build()
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=1, n_axial=2, layers=4)
    lumped = loadsteps.lumped_node_volumes(points, hexes)
    total = loadsteps.element_volumes(points, hexes).sum()
    assert lumped.sum() == pytest.approx(total, rel=1e-12)

def test_a_over_lim_is_zero_and_one_at_the_extremes():
    points, hexes = unit_cube_stack()
    below = np.full(points.shape[0], config.EPS_A_LIM / 2)
    above = np.full(points.shape[0], config.EPS_A_LIM * 2)
    assert loadsteps.a_over_lim_3d(below, points, hexes) == 0.0
    assert loadsteps.a_over_lim_3d(above, points, hexes) == 1.0

def test_a_over_lim_matches_a_hand_computed_fraction():
    """
    Top face of a 4-cube stack over the limit. That face's nodes carry 1/8 of the top
    element each, so 4 * 1/8 = 0.5 of one cube, out of 4 -> 0.125.
    """
    points, hexes = unit_cube_stack(4)
    amp = np.full(points.shape[0], config.EPS_A_LIM / 2)
    amp[points[:, 2] == 4.0] = config.EPS_A_LIM * 2
    assert loadsteps.a_over_lim_3d(amp, points, hexes) == pytest.approx(0.125)

def test_a_over_lim_rejects_a_mismatched_field():
    points, hexes = unit_cube_stack()
    with pytest.raises(ValueError):
        loadsteps.a_over_lim_3d(np.zeros(3), points, hexes)

def test_read_node_amplitudes_halves_the_range(tmp_path):
    """The deck writes a RANGE; the amplitude is half of it."""
    f = tmp_path / 'amp.txt'
    f.write_text(' 0.20000000E-01\n 0.40000000E-01\n banner line\n 0.00000000E+00\n')
    amp = loadsteps.read_node_amplitudes(f)
    assert np.allclose(amp, [0.01, 0.02, 0.0])

def test_read_node_amplitudes_can_skip_halving(tmp_path):
    f = tmp_path / 'amp.txt'
    f.write_text(' 0.20000000E-01\n')
    assert loadsteps.read_node_amplitudes(f, halve=False)[0] == pytest.approx(0.02)
