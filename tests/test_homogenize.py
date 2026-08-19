"""Effective stiffness has to reproduce exact solutions before it's trusted on cells."""

import numpy as np
import pytest
from skfem import Basis, ElementQuad1, ElementVector

import config
from geom.cell import UnitCell
from geom import reference
from geom.parametric import achievable_widths, crown
from sim2d.homogenize import homogenize
from sim2d.mesh import cell_to_mesh

N = config.GRID_N
E_BULK = config.NITINOL['E_austenite_MPa']
NU = config.NITINOL['poisson_austenite']

def exact_plane_stress():
    return E_BULK / (1 - NU ** 2) * np.array([[1, NU, 0],
                                              [NU, 1, 0],
                                              [0, 0, (1 - NU) / 2]])


# MESH #
def test_one_element_per_material_pixel():
    cell = reference.build()
    assert cell_to_mesh(cell).t.shape[1] == int(cell.to_array().sum())

def test_wrap_merges_nodes():
    """Periodic identification must actually remove DOFs, or the BCs are not periodic."""
    cell = reference.build()
    periodic = Basis(cell_to_mesh(cell), ElementVector(ElementQuad1())).N
    solid_free = 2 * (int(cell.to_array().sum()) * 4)   # loose upper bound
    assert periodic < solid_free

def test_corner_folding_does_not_break_a_solid_cell():
    """
    A node at (circ, axial) folds twice. A single-pass edge map leaves a dangling
    reference and a singular system.
    """
    result = homogenize(UnitCell(np.ones((N, N), dtype=bool)))
    assert np.all(np.isfinite(result.C_eff))

# EXACT SOLUTIONS #
def test_solid_cell_returns_the_base_material():
    """The sharpest available check: no holes means no homogenisation."""
    C = homogenize(UnitCell(np.ones((N, N), dtype=bool))).C_eff
    assert np.allclose(C, exact_plane_stress(), rtol=1e-9, atol=1e-6)

def test_solid_cell_recovers_youngs_modulus():
    assert homogenize(UnitCell(np.ones((N, N), dtype=bool))).E_circ == pytest.approx(
        E_BULK, rel=1e-9)

@pytest.mark.parametrize('fill', [0.25, 0.5, 0.75])
def test_aligned_bars_obey_the_rule_of_mixtures(fill):
    """Load along continuous bars is stretch-dominated, so E scales with area fraction."""
    a = np.zeros((N, N), dtype=bool)
    a[:, :int(fill * N)] = True
    assert homogenize(UnitCell(a)).E_axial == pytest.approx(E_BULK * fill, rel=1e-6)

def test_disconnected_bars_carry_no_load_across_them():
    a = np.zeros((N, N), dtype=bool)
    a[:, :N // 4] = True
    assert abs(homogenize(UnitCell(a)).E_circ) < 1e-6

# PROPERTIES OF C_eff #
def test_stiffness_is_symmetric():
    C = homogenize(reference.build()).C_eff
    assert np.allclose(C, C.T, rtol=1e-6, atol=1e-6)

def test_stiffness_is_positive_definite():
    """Anything else means the cell would release energy when deformed."""
    assert np.all(np.linalg.eigvalsh(homogenize(reference.build()).C_eff) > 0)

def test_cellular_stiffness_is_below_the_bulk():
    """A perforated sheet can't be stiffer than the solid it is cut from."""
    assert homogenize(reference.build()).E_circ < E_BULK

def test_thicker_struts_are_stiffer():
    previous = 0.0
    for w in achievable_widths():
        k = homogenize(crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)).K_radial
        assert k > previous
        previous = k

def test_k_radial_follows_its_definition():
    """K_radial = E_circ * t / R^2."""
    h = homogenize(reference.build())
    R = config.D_DEPLOYED_MM / 2
    assert h.K_radial == pytest.approx(
        h.E_circ * config.STRUT_THICKNESS_MM / R ** 2, rel=1e-12)

def test_labelling_is_fast_enough():
    """~5 s per cell for the whole vector; stiffness is three solves of it."""
    assert homogenize(reference.build()).seconds < 2.0
