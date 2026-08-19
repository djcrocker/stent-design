"""The tube mesh must be solvable, with no orphans, no inverted elements, right volume."""

import numpy as np
import pytest

import config
from geom import reference, tube
from sim3d import mesh3d

N_AXIAL = 1

@pytest.fixture(scope='module')
def cell():
    return reference.build()

@pytest.fixture(scope='module')
def ring(cell):
    points, hexes, info = mesh3d.tube_hex_mesh(cell, n_axial=N_AXIAL, layers=4)
    return points, hexes, info, mesh3d.quality(points, hexes)

def test_no_orphan_nodes(ring):
    points, hexes, _, _ = ring
    assert mesh3d.orphan_nodes(points, hexes) == 0

def test_no_inverted_elements(ring):
    """Ansys refuses to solve on non-positive Jacobians, so this is pass/fail for usability."""
    *_, q = ring
    assert q['n_nonpositive'] == 0
    assert q['scaled_jacobian_min'] > 0

def test_quality_is_reported_and_usable(ring):
    *_, q = ring
    assert q['scaled_jacobian_mean'] > 0.9
    assert q['volume_min_mm3'] > 0

# GEOMETRY #
def test_volume_matches_the_analytic_value(cell, ring):
    _, _, _, q = ring
    expected = tube.expected_volume_mm3(cell, n_axial=N_AXIAL)
    assert q['volume_total_mm3'] == pytest.approx(expected, rel=2e-3)

def test_radii_span_the_wall_thickness(ring):
    points, *_ = ring
    radial = np.hypot(points[:, 0], points[:, 1])
    assert radial.min() == pytest.approx(config.D_DEPLOYED_MM / 2, abs=1e-6)
    assert radial.max() == pytest.approx(
        config.D_DEPLOYED_MM / 2 + config.STRUT_THICKNESS_MM, abs=1e-6)

def test_axial_extent_is_not_exceeded(ring):
    """Boundary projection can't pull nodes past the cut planes at the tube ends."""
    points, *_ = ring
    length = config.cell_extent_mm()[1] * N_AXIAL
    assert points[:, 2].min() >= -1e-9
    assert points[:, 2].max() <= length + 1e-9

def test_one_hex_per_material_pixel_per_layer(cell, ring):
    _, hexes, info, _ = ring
    assert info['n_quads_2d'] == int(cell.tile(config.N_CIRC, N_AXIAL).sum())
    assert hexes.shape[0] == info['n_quads_2d'] * info['layers']

# THE SECTOR CASE #

def test_sector_is_not_folded_into_a_ring(cell):
    """A 1/12 sector has two free cyclic-symmetry faces. Folding it wraps a 30-degree wedge
    into a closed ring, which inverts elements and drives the total volume negative."""
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=1, n_axial=2, layers=4)
    q = mesh3d.quality(points, hexes)
    assert q['n_nonpositive'] == 0
    assert q['volume_total_mm3'] > 0

def test_sector_volume_is_the_ring_fraction(cell):
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=1, n_axial=2, layers=4)
    q = mesh3d.quality(points, hexes)
    expected = tube.expected_volume_mm3(cell, n_axial=2) / config.N_CIRC
    assert q['volume_total_mm3'] == pytest.approx(expected, rel=2e-3)

def test_sector_fits_a_student_licence(cell):
    """Runs locally on Ansys Student while the Research VM is pending."""
    _, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=1, n_axial=2, layers=4)
    assert hexes.shape[0] < 32000

# PROJECTION #

def test_projection_moves_only_boundary_nodes(cell):
    nodes, quads, boundary, _ = mesh3d.quad_mesh_2d(cell, n_circ=config.N_CIRC, n_axial=1)
    moved = mesh3d.project_to_level_set(nodes, boundary, cell, n_axial=1)
    assert np.allclose(moved[~boundary], nodes[~boundary])
    assert not np.allclose(moved[boundary], nodes[boundary])

def test_unprojected_mesh_is_the_voxel_mesh(cell):
    """
    When project=False, the mesh should reproduce the stair-stepped geometry. 
    Quality is near but not exactly 1.0: wrapping voxels onto a cylinder makes them slightly trapezoidal.
    """
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_axial=1, layers=4, project=False)
    q = mesh3d.quality(points, hexes)
    assert q['scaled_jacobian_min'] > 0.999
    assert q['scaled_jacobian_min'] < 1.0

def test_written_deck_has_a_node_and_element_for_each(cell, tmp_path):
    out = tmp_path / 'mesh.inp'
    info, _ = mesh3d.build_and_write(cell, out, n_circ=1, n_axial=1, layers=2)
    text = out.read_text(encoding='utf-8')
    assert text.count('\nN,') == info['n_points']
    assert text.count('\nEN,') == info['n_hexes']
    assert 'SOLID185' in text

def test_no_element_exceeds_the_ansys_face_angle_warning(ring):
    """Ansys warns on brick faces past 155 degrees."""
    points, hexes, _, q = ring
    assert q['n_over_155deg'] == 0
    assert q['max_face_angle_deg'] < 155.0

def test_projection_clamp_keeps_quality_solvable(ring):
    """The clamp is what trades smoothing against element shape."""
    *_, q = ring
    assert q['scaled_jacobian_min'] > 0.4
