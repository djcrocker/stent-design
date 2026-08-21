"""Contact surface extraction for the crimp."""

import numpy as np
import pytest

import config
from geom import reference
from sim3d import contact, mesh3d

def single_hex():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                       [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
    return points, np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

@pytest.fixture(scope='module')
def ring():
    cell = reference.build()
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=config.N_CIRC, n_axial=1, layers=2)
    return points, hexes

def test_lone_hex_has_six_exterior_faces():
    _, hexes = single_hex()
    owners, faces = contact.exterior_faces(hexes)
    assert len(faces) == 6
    assert set(owners) == {0}

def test_shared_face_is_not_exterior():
    """Two stacked hexes share one face, so 12 faces minus 2 shared."""
    points, _ = single_hex()
    upper = points + np.array([0., 0., 1.])
    all_points = np.vstack([points, upper])
    hexes = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                      [4, 5, 6, 7, 8, 9, 10, 11]])
    _, faces = contact.exterior_faces(hexes)
    assert len(faces) == 10

def test_skins_are_equal_and_side_walls_scale_with_layers(ring):
    points, hexes = ring
    owners, faces = contact.exterior_faces(hexes)
    kinds = contact.classify_faces(points, owners, faces, hexes)
    # Inner and outer skins are the same 2D surface, so they have to match exactly.
    assert len(kinds['inner']) == len(kinds['outer'])
    assert len(kinds['side']) > 0
    assert len(kinds['inner']) + len(kinds['outer']) + len(kinds['side']) == len(faces)

def test_skins_do_not_depend_on_layer_count():
    """Radial skins come from the 2D mesh; only side walls grow with layers."""
    cell = reference.build()
    counts = {}
    for layers in (1, 2):
        points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=config.N_CIRC,
                                                n_axial=1, layers=layers)
        owners, faces = contact.exterior_faces(hexes)
        k = contact.classify_faces(points, owners, faces, hexes)
        counts[layers] = (len(k['outer']), len(k['side']))
    assert counts[1][0] == counts[2][0]
    assert counts[2][1] == 2 * counts[1][1]

def test_outward_faces_point_away_from_their_element(ring):
    points, hexes = ring
    owners, faces = contact.exterior_faces(hexes)
    kinds = contact.classify_faces(points, owners, faces, hexes)
    idx = kinds['side'][:400]
    out = contact.outward_faces(points, owners, faces, hexes, idx)
    for i, nodes in zip(idx, out):
        p = points[nodes]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        ecen = points[hexes[owners[i]]].mean(axis=0)
        # A contact surface facing into its own solid detects nothing.
        assert np.dot(nrm, p.mean(axis=0) - ecen) > 0

def test_crimper_cylinder_is_closed_and_clear_of_the_stent(ring):
    points, _ = ring
    nodes, quads, r = contact.crimper_cylinder(points, n_theta=24, n_z=4)
    r_outer = config.D_DEPLOYED_MM / 2.0 + config.STRUT_THICKNESS_MM
    assert r > r_outer                       # Starts out of contact
    assert len(nodes) == 24 * 5
    assert len(quads) == 24 * 4
    # Every node sits on the cylinder.
    assert np.allclose(np.hypot(nodes[:, 0], nodes[:, 1]), r)
    # Closed in theta: every node is used by four quads.
    counts = np.bincount(quads.ravel(), minlength=len(nodes))
    assert set(np.unique(counts[24:-24])) == {4}

def test_crimp_radius_reaches_the_configured_diameter():
    assert contact.crimp_radius_mm(config.D_CRIMPED_MM) * 2 == config.D_CRIMPED_MM

def test_budget_totals_its_parts(ring):
    points, hexes = ring
    owners, faces = contact.exterior_faces(hexes)
    kinds = contact.classify_faces(points, owners, faces, hexes)
    b = contact.budget(points, hexes, kinds, 768)
    assert b['total'] == (b['nodes'] + b['hexes'] + b['self_contact']
                          + b['crimper_contact'] + b['crimper_target'])
    assert b['self_contact'] == 2 * len(kinds['side'])

def test_base_anchors_are_distinct_and_on_the_base_edge(ring):
    points, _ = ring
    anchors = contact.base_anchor_nodes(points)
    assert len(set(anchors)) == 3
    z0 = points[:, 2].min()
    for n in anchors:
        assert points[n - 1, 2] == pytest.approx(z0, abs=1e-9)

def test_base_anchors_kill_all_three_rigid_modes(ring):
    """
    The requirement is non-degeneracy, not 120 deg spacing.

    The ring is perforated, so the z = 0 edge only has material at some angles (here
    11.25-348.75 deg) and the nominal 0/120/240 is not reachable. What must hold is that
    pinning UY at these three nodes leaves no rigid mode free: rotation contributes a
    constant, x-translation -sin(theta), y-translation cos(theta), so the 3x3 matrix of
    those columns has to stay well conditioned.
    """
    points, _ = ring
    anchors = contact.base_anchor_nodes(points)
    theta = np.radians([np.degrees(np.arctan2(points[n - 1, 1], points[n - 1, 0])) % 360.0
                        for n in anchors])
    M = np.column_stack([np.ones(3), -np.sin(theta), np.cos(theta)])
    assert abs(np.linalg.det(M)) > 0.5
    assert np.linalg.cond(M) < 10.0

def test_base_anchors_are_one_based_for_apdl(ring):
    points, _ = ring
    assert min(contact.base_anchor_nodes(points)) >= 1
    assert max(contact.base_anchor_nodes(points)) <= points.shape[0]
