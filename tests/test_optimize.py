"""Tests for the SIMP optimizer loop, and the two published validations of it."""

import numpy as np
import pytest

import config
from baseline import mbb, optimize, simp

NU = config.NITINOL['poisson_austenite']
E_BULK = config.NITINOL['E_austenite_MPa']

# THE FILTER AND THE CHAIN RULE #
@pytest.mark.parametrize('mode', ['wrap', 'constant'])
def test_the_filter_leaves_a_uniform_field_alone(mode):
    """
    A normalized average of a constant field is that constant, at the boundary too.

    This is the whole job of the boundary correction: without it the beam's edges would be
    filtered toward zero and the optimizer would see a domain that tapers off at its edges.
    """
    assert simp.density_filter(np.ones((12, 16)), 2.5, mode) == pytest.approx(1.0)

def test_the_cell_filter_crosses_the_seam_and_the_beam_filter_does_not():
    x = np.zeros((16, 16))
    x[0, 0] = 1.0
    assert simp.density_filter(x, 2.5, 'wrap')[-1, -1] > 0.0
    assert simp.density_filter(x, 2.5, 'constant')[-1, -1] == 0.0

@pytest.mark.parametrize('mode', ['wrap', 'constant'])
def test_the_chain_rule_is_the_transpose_of_the_filter(mode):
    """
    dot(filter(x), y) == dot(x, chain_rule(y)), which is what makes the sensitivity correct.

    With the boundary correction the filter is diag(1/Hs) K, so the transpose divides before
    convolving.
    """
    rng = np.random.default_rng(11)
    x, y = rng.random((10, 14)), rng.random((10, 14))
    assert (simp.density_filter(x, 2.0, mode) * y).sum() == pytest.approx(
        (x * simp.filter_chain_rule(y, 2.0, mode)).sum())

# THE OC STEP #
def test_the_oc_step_meets_a_reachable_volume_target():
    rng = np.random.default_rng(5)
    x = np.full((8, 8), 0.5)
    benefit = rng.random((8, 8))
    new = optimize.oc_step(x, benefit, np.ones_like(x), lambda c: c.sum(), 0.45, move=0.1)
    assert new.sum() == pytest.approx(0.45 * x.size, rel=1e-6)
    assert np.all(np.abs(new - x) <= 0.1 + 1e-12)
    assert np.all((new >= 0.0) & (new <= 1.0))
    # More benefit means more material.
    assert np.corrcoef(benefit.ravel(), new.ravel())[0, 1] > 0.9

def test_the_move_limit_outranks_the_volume_target_in_one_step():
    """
    From x = 0.5 with move = 0.1 no step can reach volume 0.3, and the step has to saturate at
    the move limit - the loop walks the volume down over iterations instead.
    """
    x = np.full((8, 8), 0.5)
    new = optimize.oc_step(x, np.ones_like(x), np.ones_like(x), lambda c: c.sum(), 0.3, move=0.1)
    assert new == pytest.approx(0.4)

# VALIDATION (b): THE top88 MBB BEAM #
def test_the_compliance_gradient_matches_a_finite_difference():
    rng = np.random.default_rng(3)
    rho = 0.4 + 0.3 * rng.random((6, 10))
    _, dc = mbb.compliance(rho, nelx=10, nely=6)
    h = 1e-6
    for cell in [(0, 0), (3, 5), (5, 9)]:
        plus, minus = rho.copy(), rho.copy()
        plus[cell] += h
        minus[cell] -= h
        fd = (mbb.compliance(plus, nelx=10, nely=6)[0]
              - mbb.compliance(minus, nelx=10, nely=6)[0]) / (2.0 * h)
        assert dc[cell] == pytest.approx(fd, rel=1e-4)

def test_the_mbb_beam_reproduces_the_published_compliance():
    """
    Andreassen et al. (2011), 60 x 20, volfrac 0.5, penal 3, rmin 0.04 x 60 = 2.4: c = 233.71
    with sensitivity filtering (237.60 with their PDE density filter). We measure 233.64 with
    a convolution density filter.

    Tolerance is 1%: the published value uses a different filter variant, so a tighter number 
    would claim more agreement than the comparison has.
    """
    result = mbb.run(iters=200, beta_max=0.0, tol=1e-4)
    assert result.value == pytest.approx(mbb.PUBLISHED, rel=0.01)
    assert result.volume == pytest.approx(mbb.VOLFRAC, rel=1e-6)
    compliances = [row['value'] for row in result.history]
    assert all(b <= a + 1e-9 for a, b in zip(compliances, compliances[1:]))

def test_the_projection_makes_the_design_crisp():
    """Heaviside continuation is only worth its instability if it removes the gray."""
    filtered = mbb.run(nelx=30, nely=10, radius=1.2, iters=60, beta_max=0.0)
    projected = mbb.run(nelx=30, nely=10, radius=1.2, iters=60, beta_max=16.0)
    assert filtered.history[-1]['gray'] > 0.25
    assert projected.history[-1]['gray'] < 0.05

# VALIDATION (c): THE HASHIN-SHTRIKMAN BOUND #
def test_bulk_modulus_of_solid_material_is_the_closed_form():
    """K_2D = E / (2 (1 - nu)) for plane stress, which also pins `bulk`'s factor of 4."""
    solid = simp.homogenize_density(np.ones((8, 8)))
    assert simp.bulk(solid.C_eff) == pytest.approx(E_BULK / (2.0 * (1.0 - NU)), rel=1e-9)

def test_the_hs_bound_is_below_the_solid_and_rises_to_it():
    assert simp.hs_bulk_bound(1.0) == pytest.approx(E_BULK / (2.0 * (1.0 - NU)), rel=1e-12)
    assert simp.hs_bulk_bound(0.0) == 0.0
    assert simp.hs_bulk_bound(0.3) < simp.hs_bulk_bound(0.6)

def test_max_bulk_approaches_the_hashin_shtrikman_bound_without_exceeding_it():
    """The bound optimal 2D microstructures are known to attain."""
    result = optimize.optimize_cell(0.5, objective='bulk', n=24, iters=60)
    ratio = result.value / simp.hs_bulk_bound(0.5)
    assert 0.85 <= ratio <= 1.0
    assert result.volume == pytest.approx(0.5, rel=1e-6)
    assert result.history[-1]['gray'] < 0.10        # and it is a structure, not a gray wash

def test_a_uniform_start_is_a_stationary_point():
    """Why `initial_design` exists, kept as a test because this failure looks like success."""
    n = 16
    flat = optimize.optimize_cell(0.5, objective='bulk', n=n, iters=5, x0=np.full((n, n), 0.5))
    solid = simp.bulk(simp.homogenize_density(np.ones((8, 8))).C_eff)
    assert flat.value == pytest.approx(0.5 ** simp.PENAL * solid, rel=1e-6)
    assert optimize.initial_design((n, n), 0.5).std() > 0.0

# A CELL AT A PRESCRIBED STIFFNESS #
def test_k_radial_rises_with_the_volume_fraction():
    """The monotonicity `cell_for_k_radial`'s bisection rests on."""
    low = optimize.optimize_cell(0.3, n=16, iters=25)
    high = optimize.optimize_cell(0.5, n=16, iters=25)
    assert low.value < high.value

def test_a_prescribed_k_radial_is_reached_by_bisecting_the_volume():
    target = 600.0                  # between the 434 and 794 the first two brackets reach
    result = optimize.cell_for_k_radial(target, n=16, iters=25, max_outer=4)
    assert result.value == pytest.approx(target, rel=0.10)

def test_an_unreachable_target_returns_the_closest_run_rather_than_pretending():
    """
    These cells are far stiffer circumferentially than the conditioning ladder's K_radial
    targets, so this case is not hypothetical and the caller has to read `Result.value`.
    """
    result = optimize.cell_for_k_radial(1.0, n=16, iters=10, max_outer=3)
    assert result.value > 10.0
