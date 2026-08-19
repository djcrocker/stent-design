"""The wrapped tube must close exactly and match arithmetic."""

import numpy as np
import pytest

import config
from geom import tube
from geom.handmade import diamond

N_AXIAL = 3

@pytest.fixture(scope='module')
def cell():
    return diamond()

@pytest.fixture(scope='module')
def mesh(cell):
    return tube.tube_mesh(cell, n_axial=N_AXIAL)

def test_one_hexahedron_per_material_pixel(cell, mesh):
    assert mesh.n_cells == int(cell.tile(config.N_CIRC, N_AXIAL).sum())

def test_ring_closes_watertight(mesh):
    """The seam is index arithmetic, not duplicated vertices."""
    surface = mesh.extract_surface(algorithm='dataset_surface')
    assert surface.n_open_edges == 0

def test_radii_are_exact(mesh):
    radial = np.hypot(mesh.points[:, 0], mesh.points[:, 1])
    assert radial.min() == pytest.approx(config.D_DEPLOYED_MM / 2)
    assert radial.max() == pytest.approx(config.D_DEPLOYED_MM / 2
                                         + config.STRUT_THICKNESS_MM)

def test_length_is_n_axial_cells(mesh):
    length = mesh.bounds[5] - mesh.bounds[4]
    assert length == pytest.approx(config.cell_extent_mm()[1] * N_AXIAL)

def test_volume_matches_the_analytic_value(cell, mesh):
    volume = float(mesh.compute_cell_sizes()['Volume'].sum())
    assert volume == pytest.approx(tube.expected_volume_mm3(cell, n_axial=N_AXIAL),
                                   rel=1e-3)

def test_every_hexahedron_has_positive_volume(mesh):
    """Negative volume means inverted node ordering, which an FEA solver would reject."""
    assert float(mesh.compute_cell_sizes()['Volume'].min()) > 0

def test_axial_repeats_scale_the_mesh(cell):
    one = tube.tube_mesh(cell, n_axial=1)
    three = tube.tube_mesh(cell, n_axial=3)
    assert three.n_cells == 3 * one.n_cells

def test_n_circ_closes_the_ring_without_overlap(cell, mesh):
    """N_CIRC cells must span exactly 2*pi*r."""
    circumference = 2 * np.pi * (config.D_DEPLOYED_MM / 2)
    assert config.cell_extent_mm()[0] * config.N_CIRC == pytest.approx(circumference)
