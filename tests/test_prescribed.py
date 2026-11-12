"""Tests for the prescribed-stiffness formulation."""

import numpy as np
import pytest

import config
from baseline import optimize, prescribed, simp

N = 12          # The coarsest grid the periodic cell still means something on
RTOL_FD = 2e-3  # The forward solve is iterative, and the step is a central difference

@pytest.fixture(scope='module')
def gray():
    """A mid-density field and its solved stiffness."""
    rng = np.random.default_rng(7)
    rho = simp.density_filter(0.2 + 0.7 * rng.random((N, N)), 1.5)
    return rho, simp.homogenize_density(rho)

# NEW CALCULUS #
def test_axial_gradient_matches_finite_differences(gray):
    rho, solved = gray
    analytic = solved.dE(1)
    step = 1e-4
    rng = np.random.default_rng(0)
    for idx in map(tuple, rng.integers(0, N, size=(4, 2))):
        up, down = rho.copy(), rho.copy()
        up[idx] += step
        down[idx] -= step
        fd = ((simp.homogenize_density(up).E_axial
               - simp.homogenize_density(down).E_axial) / (2.0 * step))
        assert analytic[idx] == pytest.approx(fd, rel=RTOL_FD)

def test_circumferential_gradient_is_still_the_one_k_radial_uses(gray):
    """`dE` is a refactor of `dK_radial`, so the two have to stay in exact proportion."""
    _, solved = gray
    scale = config.STRUT_THICKNESS_MM / (config.D_DEPLOYED_MM / 2.0) ** 2
    assert solved.dK_radial() == pytest.approx(solved.dE(0) * scale, rel=1e-14)

def test_the_two_directions_have_different_gradients(gray):
    """
    The axial and circumferential derivatives can't be the same field.

    A transposed or mis-indexed axis would make them identical on a symmetric design and
    the whole axial prescription would be a no-op.
    """
    _, solved = gray
    assert not np.allclose(solved.dE(0), solved.dE(1))

# THE OBJECTIVE #
def test_e_circ_for_inverts_k_radial():
    for k in (1.0, 100.0, 300.0):
        e = prescribed.e_circ_for(k)
        assert e * config.STRUT_THICKNESS_MM / (config.D_DEPLOYED_MM / 2.0) ** 2 == \
            pytest.approx(k, rel=1e-14)

def test_g_is_one_when_both_prescriptions_are_exactly_met(gray):
    """
    At the design's own stiffness as the target, both normalized moduli are 1, but only
    if the floor is set to the design's own ratio.
    """
    _, solved = gray
    e_target = solved.E_circ
    floor = solved.E_axial / solved.E_circ
    g, _ = prescribed.binding(solved, e_target, floor=floor)
    assert g == pytest.approx(1.0 - np.log(2.0) / prescribed.AGGREGATE_P, rel=1e-12)

@pytest.mark.parametrize('p', [5.0, 20.0, 200.0])
def test_softmin_never_overstates_the_binding_modulus(gray, p):
    """
    `g <= min(...)`, within log(2)/p, and from below.

    Feasibility is reported as `g >= 1`, so a `g` that could exceed the true minimum would
    let a run be called matched while one of the two prescriptions was missed.
    """
    _, solved = gray
    e_target, floor = 0.5 * solved.E_circ, 0.8
    hard = min(solved.E_circ / e_target, solved.E_axial / (floor * e_target))
    g, _ = prescribed.binding(solved, e_target, floor=floor, p=p)
    assert g <= hard
    assert g >= hard - np.log(2.0) / p - 1e-12

def test_softmin_gradient_is_a_convex_blend_of_the_two(gray):
    """
    The gradient is a weighted average of both directions' gradients.
    """
    _, solved = gray
    e_target, floor = solved.E_circ, 1.0
    _, grad = prescribed.binding(solved, e_target, floor=floor, p=1.0)
    d_circ, d_axial = solved.dE(0) / e_target, solved.dE(1) / (floor * e_target)
    assert not np.allclose(grad, d_circ)
    assert not np.allclose(grad, d_axial)
    # Weights sum to one, so the blend lies between the two fields everywhere
    assert np.all(grad >= np.minimum(d_circ, d_axial) - 1e-9)
    assert np.all(grad <= np.maximum(d_circ, d_axial) + 1e-9)

def test_a_zero_floor_drops_the_axial_prescription(gray):
    """`floor = 0` has to be the unconstrained circumferential objective exactly."""
    _, solved = gray
    e_target = solved.E_circ
    g, grad = prescribed.binding(solved, e_target, floor=0.0)
    assert g == pytest.approx(1.0, rel=1e-14)
    assert np.allclose(grad, solved.dE(0) / e_target)

def test_binding_gradient_matches_finite_differences(gray):
    """The blended gradient, not just its two ingredients."""
    rho, solved = gray
    e_target, floor, p = 0.9 * solved.E_circ, 0.9, 20.0
    analytic = prescribed.binding(solved, e_target, floor, p)[1]
    step = 1e-4
    rng = np.random.default_rng(1)
    for idx in map(tuple, rng.integers(0, N, size=(3, 2))):
        up, down = rho.copy(), rho.copy()
        up[idx] += step
        down[idx] -= step
        fd = ((prescribed.binding(simp.homogenize_density(up), e_target, floor, p)[0]
               - prescribed.binding(simp.homogenize_density(down), e_target, floor, p)[0])
              / (2.0 * step))
        assert analytic[idx] == pytest.approx(fd, rel=RTOL_FD)

# FEASIBILITY TEST #
def test_matched_is_one_sided_on_k_radial():
    """Overshoot is feasible and merely heavy; only a shortfall is a miss."""
    over = {'k_rel_error': 1.5, 'floor_met': True}
    under = {'k_rel_error': -0.2, 'floor_met': True}
    edge = {'k_rel_error': -prescribed.MATCH_RTOL, 'floor_met': True}
    assert prescribed.matched(over)
    assert not prescribed.matched(under)
    assert prescribed.matched(edge)

def test_matched_requires_the_axial_floor_too():
    assert not prescribed.matched({'k_rel_error': 0.0, 'floor_met': False})

def test_achieved_reports_the_floor_against_the_target_not_the_design(gray):
    """`floor_rel` is E_axial / (floor * E_circ_TARGET), not E_axial / E_circ_achieved."""
    rho, solved = gray
    k_target = 0.5 * solved.K_radial
    record = prescribed.achieved(rho, k_target, floor=1.0, binary=False)
    assert record['floor_rel'] == pytest.approx(
        solved.E_axial / prescribed.e_circ_for(k_target), rel=1e-12)
    assert record['ratio'] == pytest.approx(solved.E_axial / solved.E_circ, rel=1e-12)
    assert record['k_rel_error'] == pytest.approx(1.0, rel=1e-9)

# OUTER SEARCH #
def test_min_volume_cell_returns_the_lightest_it_found_feasible():
    """A target this coarse grid can reach comfortably, two bisection steps, few iterations."""
    got = prescribed.min_volume_cell(50.0, steps=2, n=N, iters=6, binary=False, repair=())
    assert len(got.bracket) == 2
    hits = [row['volfrac'] for row in got.bracket if row['matched']]
    if hits:
        assert got.volfrac == pytest.approx(min(hits))
        assert prescribed.matched(got.achieved)
    else:   # nothing feasible in the bracket: it must hand back the closest, not a hit
        assert got.volfrac in [row['volfrac'] for row in got.bracket]

def test_the_bisection_goes_lighter_on_a_hit_and_heavier_on_a_miss():
    """
    A volume fraction that reaches the target has to send the search lighter.

    Backwards, this would still return a feasible cell - just never the lightest one - so
    the min-volume half of the formulation would be silently absent rather than wrong.
    """
    got = prescribed.min_volume_cell(50.0, steps=2, n=N, iters=4, binary=False, repair=())
    lo, hi = prescribed.VOLFRAC_BOUNDS
    first, second = got.bracket[0], got.bracket[1]
    assert first['volfrac'] == pytest.approx(0.5 * (lo + hi))
    assert second['volfrac'] == pytest.approx(0.5 * (lo + first['volfrac'])
                                              if first['reached']
                                              else 0.5 * (first['volfrac'] + hi))
    assert all({'g', 'matched', 'reached', 'K_radial', 'valid'} <= set(row)
               for row in got.bracket)

def test_the_bisection_follows_achieved_k_radial_and_not_the_aggregate():
    """The branch is decided by the matched quantity."""
    target = 50.0
    got = prescribed.min_volume_cell(target, steps=4, n=N, iters=4, binary=False, repair=())
    for first, second in zip(got.bracket, got.bracket[1:]):
        went_lighter = second['volfrac'] < first['volfrac']
        assert first['reached'] == went_lighter
        assert went_lighter == (first['K_radial'] >= target * (1.0 - prescribed.MATCH_RTOL))

# LENGTH-SCALE REPAIR #
def test_the_repair_only_widens_the_filter_and_never_moves_the_volume():
    """
    The repair exists so the baseline passes the manufacturability screen, 
    and it can't just do that by making the cell heavier.
    """
    got = prescribed.min_volume_cell(50.0, steps=2, n=N, iters=4, binary=False, repair=(1.5, 2.0))
    repairs = [row for row in got.bracket if row.get('repair')]
    searched = [row for row in got.bracket if not row.get('repair')]
    assert all(row['volfrac'] == pytest.approx(got.volfrac) for row in repairs)
    assert len(searched) == 2
    if repairs:
        radii = [row['radius'] for row in repairs]
        assert radii == sorted(radii)   # Smallest radius that works tried first

def test_no_repair_rows_when_repair_is_off():
    got = prescribed.min_volume_cell(50.0, steps=2, n=N, iters=4, binary=False, repair=())
    assert not any(row.get('repair') for row in got.bracket)
    assert got.radius == pytest.approx(optimize.default_radius(N))

# THE ROBUST (ERODED) FORMULATION #
def test_the_delivered_design_is_thicker_than_the_one_optimized():
    """
    The erosion margin is the length scale.

    The loop is scored on the eroded field and the cell handed over is the intermediate
    one, so the delivered design has to carry more material than what was optimized.
    """
    run = prescribed.optimize_prescribed(50.0, 0.3, n=N, iters=6)
    rho = prescribed.delivered(run, n=N, iters=6)
    assert rho.mean() > run.rho.mean()

def test_a_half_eta_run_delivers_exactly_what_it_optimized():
    """eta = 0.5 has to be the plain formulation."""
    run = prescribed.optimize_prescribed(50.0, 0.3, n=N, iters=6, eta=0.5)
    assert prescribed.delivered(run, n=N, iters=6) == pytest.approx(run.rho, rel=1e-12)

def test_the_delivered_field_is_rebuilt_at_the_beta_the_run_ended_at():
    """A mismatched beta would compare two different fields."""
    run = prescribed.optimize_prescribed(50.0, 0.3, n=N, iters=6, eta=0.5)
    assert optimize.beta_at(6, 6, 16.0) < 16.0          # the case worth guarding
    wrong = prescribed.delivered(run, n=N, iters=60)
    assert not np.allclose(wrong, run.rho)

# THE SCREENED ARM #
def test_validity_stays_out_of_the_search_direction():
    """An invalid cell says nothing about which way to go, so it can't steer the bisection."""
    got = prescribed.min_volume_cell(50.0, steps=3, n=N, iters=4, binary=False, repair=(), require_valid=True)
    for row in got.bracket:
        assert row['reached'] == (row['K_radial'] >= 50.0 * (1.0 - prescribed.MATCH_RTOL))

def test_an_explicit_ladder_visits_exactly_what_it_was_given():
    """Sweep mode, which is how the screened arm samples the valid band."""
    ladder = (0.2, 0.3, 0.4)
    got = prescribed.min_volume_cell(50.0, n=N, iters=4, binary=False, repair=(), require_valid=True, volfracs=ladder)
    assert [row['volfrac'] for row in got.bracket] == pytest.approx(list(ladder))
    valid_hits = [row['volfrac'] for row in got.bracket if row['matched'] and row['valid']]
    if valid_hits:
        assert got.volfrac == pytest.approx(min(valid_hits))

def test_without_require_valid_validity_is_recorded_but_not_enforced():
    """The unscreened arm still reports `valid`."""
    got = prescribed.min_volume_cell(50.0, steps=2, n=N, iters=4, binary=False, repair=())
    assert all('valid' in row for row in got.bracket)
    for row in got.bracket:
        assert row['reached'] == (row['K_radial'] >= 50.0 * (1.0 - prescribed.MATCH_RTOL))
