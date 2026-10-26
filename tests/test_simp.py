"""Setup and tests for the SIMP baseline."""

import numpy as np
import pytest

import config
from baseline import simp
from geom import reference
from geom.cell import UnitCell
from sim2d.homogenize import homogenize

N = config.GRID_N
E_BULK = config.NITINOL['E_austenite_MPa']
NU = config.NITINOL['poisson_austenite']

def exact_plane_stress():
    return E_BULK / (1 - NU ** 2) * np.array([[1, NU, 0],
                                              [NU, 1, 0],
                                              [0, 0, (1 - NU) / 2]])

@pytest.fixture(scope='module')
def gray():
    """A smooth interior density field: no 0s or 1s, so the gradient is differentiable."""
    rng = np.random.default_rng(7)
    return 0.35 + 0.3 * rng.random((N, N))

@pytest.fixture(scope='module')
def gray_result(gray):
    """Solved once."""
    return simp.homogenize_density(gray)

# FORWARD SOLVE #
def test_full_density_returns_the_base_material():
    """rho = 1 is a solid cell, so the density solve has to reproduce the closed form."""
    C = simp.homogenize_density(np.ones((N, N))).C_eff
    assert np.allclose(C, exact_plane_stress(), rtol=1e-9, atol=1e-6)

def test_the_energy_form_agrees_with_the_published_labeler():
    """
    The energy assembly here and the stress average in `sim2d.homogenize` are the same
    quantity.

    Measured agreement is 1.2e-08 relative, so the tolerance is set near that rather than
    somewhere comfortable - a loose tolerance here would pass whatever the two forms did.
    """
    cell = reference.build()
    rho = cell.to_array().astype(float)
    mine = simp.homogenize_density(rho).C_eff
    theirs = homogenize(cell).C_eff
    assert np.allclose(mine, theirs, rtol=1e-6, atol=1e-6 * abs(theirs).max())

def test_a_transposed_density_field_is_a_different_answer():
    """
    Guards the check above: if the two agreed for any layout, that test would prove nothing
    about the element order.
    """
    rho = reference.build().to_array().astype(float)
    assert not np.allclose(simp.homogenize_density(rho).C_eff,
                           simp.homogenize_density(rho.T).C_eff, rtol=1e-6)

def test_aligned_bars_obey_the_rule_of_mixtures():
    rho = np.zeros((N, N))
    rho[:, :N // 2] = 1.0
    assert simp.homogenize_density(rho).E_axial == pytest.approx(E_BULK * 0.5, rel=1e-3)

def test_stiffness_is_symmetric_and_positive_definite(gray):
    C = simp.homogenize_density(gray).C_eff
    assert np.allclose(C, C.T, rtol=1e-6, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(C) > 0)

def test_void_is_soft_but_not_singular():
    """E_min keeps K invertible; it can't contribute real stiffness."""
    C = simp.homogenize_density(np.full((N, N), 1e-4)).C_eff
    assert np.all(np.isfinite(C))
    assert abs(C).max() < 1e-6 * E_BULK

# THE GRADIENT #
@pytest.mark.parametrize('ij', [(0, 0), (1, 1), (0, 1), (2, 2)])
def test_dC_matches_a_finite_difference(gray, gray_result, ij):
    i, j = ij
    analytic = gray_result.dC(i, j)
    h = 1e-5
    rng = np.random.default_rng(11)
    for flat in rng.choice(N * N, size=3, replace=False):
        plus, minus = gray.copy(), gray.copy()
        plus.ravel()[flat] += h
        minus.ravel()[flat] -= h
        fd = (simp.homogenize_density(plus).C_eff[i, j]
            - simp.homogenize_density(minus).C_eff[i, j]) / (2 * h)
        assert analytic.ravel()[flat] == pytest.approx(fd, rel=2e-4, abs=1e-9)

def test_dK_radial_matches_a_finite_difference(gray, gray_result):
    """
    K_radial runs through the compliance, so its gradient is the one most likely to be
    wrong in a way the C gradient would not catch.
    """
    analytic = gray_result.dK_radial()
    h = 1e-5
    rng = np.random.default_rng(13)
    for flat in rng.choice(N * N, size=3, replace=False):
        plus, minus = gray.copy(), gray.copy()
        plus.ravel()[flat] += h
        minus.ravel()[flat] -= h
        fd = (simp.homogenize_density(plus).K_radial
            - simp.homogenize_density(minus).K_radial) / (2 * h)
        assert analytic.ravel()[flat] == pytest.approx(fd, rel=2e-4, abs=1e-12)

def test_the_gradient_costs_no_extra_solves(gray_result):
    """The point of self-adjointness: the gradient is free once C_eff is computed."""
    assert len(gray_result.fluctuations) == 3
    assert gray_result.energies.shape == (3, 3, N * N)

def test_adding_material_anywhere_stiffens_the_hoop(gray_result):
    """
    dC[0,0]/drho >= 0 at every element. A sign error in the mutual energy would show up
    here as a region where filling metal in made the cell softer circumferentially.
    """
    assert np.all(gray_result.dC(0, 0) >= -1e-12)

# LENGTH SCALE #
def test_the_filter_conserves_material_on_a_torus():
    """A normalized kernel on a periodic domain adds and removes nothing."""
    rng = np.random.default_rng(3)
    x = rng.random((N, N))
    assert simp.density_filter(x, 3.0).sum() == pytest.approx(x.sum(), rel=1e-12)

def test_the_filter_wraps_across_the_seam():
    """A spot on the last row must bleed onto the first, or the seam is an edge."""
    x = np.zeros((N, N))
    x[-1, N // 2] = 1.0
    assert simp.density_filter(x, 3.0)[0, N // 2] > 0

def test_the_filter_spreads_material_at_least_its_radius():
    x = np.zeros((N, N))
    x[N // 2, N // 2] = 1.0
    out = simp.density_filter(x, 4.0)
    assert out[N // 2, N // 2 + 3] > 0
    assert out[N // 2, N // 2 + 5] == pytest.approx(0.0)

def test_projection_fixes_the_endpoints_and_is_monotone():
    x = np.linspace(0, 1, 101)
    p = simp.project(x, beta=8.0)
    assert p[0] == pytest.approx(0.0)
    assert p[-1] == pytest.approx(1.0)
    assert np.all(np.diff(p) > 0)

def test_projection_sharpens_toward_binary_as_beta_grows():
    x = np.array([0.3, 0.7])
    soft = simp.project(x, beta=2.0)
    hard = simp.project(x, beta=64.0)
    assert hard[0] < soft[0] and hard[1] > soft[1]

def test_zero_beta_is_the_identity():
    x = np.linspace(0, 1, 11)
    assert np.allclose(simp.project(x, beta=0.0), x)
    assert np.allclose(simp.project_grad(x, beta=0.0), 1.0)

def test_project_grad_matches_a_finite_difference():
    x = np.array([0.2, 0.45, 0.5, 0.8])
    h = 1e-6
    fd = (simp.project(x + h, beta=8.0) - simp.project(x - h, beta=8.0)) / (2 * h)
    assert np.allclose(simp.project_grad(x, beta=8.0), fd, rtol=1e-6)
