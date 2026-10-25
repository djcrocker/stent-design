"""
The bending load is imposed as an explicit displacement field on the end faces, so a sign
or factor error wouldn't crash anything, it'd just solve the wrong problem and report
a plausible number. These check the map against what pure bending is defined to do.
"""

import numpy as np
import pytest

import config
from sim3d import apdl, flex, loadsteps

R = 27.0

def test_midspan_is_undisplaced():
    """zeta = 0 is the fixed point of the map; otherwise the ring translates bodily."""
    uy, uz = apdl.bend_displacement(np.array([-0.3, 0.0, 0.3]), 0.0, R)
    assert np.allclose(uy, 0.0)
    assert np.allclose(uz, 0.0)

def test_neutral_axis_lands_on_the_arc():
    """At y = 0 the map has to put the node exactly on a circle of radius R centred at y = R."""
    zeta = np.linspace(-5.0, 5.0, 21)
    uy, uz = apdl.bend_displacement(0.0, zeta, R)
    y_new, z_new = uy, zeta + uz
    assert np.allclose(np.hypot(y_new - R, z_new), R)

def test_neutral_axis_arc_length_is_preserved():
    """Pure bending neither stretches nor shortens the neutral axis."""
    zeta = np.linspace(-3.14, 3.14, 401)
    uy, uz = apdl.bend_displacement(0.0, zeta, R)
    y_new, z_new = uy, zeta + uz
    arc = np.hypot(np.diff(y_new), np.diff(z_new)).sum()
    assert arc == pytest.approx(zeta[-1] - zeta[0], rel=1e-7)

def test_curvature_matches_one_over_radius():
    """End-to-end rotation over length L has to be L/R, which is what the curvature means."""
    L = 4 * config.cell_extent_mm()[1]
    uy, uz = apdl.bend_displacement(0.0, np.array([-L / 2, L / 2]), R)
    z_new = np.array([-L / 2, L / 2]) + uz
    # Angle subtended at the arc centre (y = R, z = 0).
    ang = np.arctan2(z_new, R - uy)
    assert float(ang[1] - ang[0]) == pytest.approx(L / R, rel=1e-9)

def _arc_length_at(y, zeta, radius):
    """Deformed arc length of the fiber at offset `y`, at an arbitrary bend radius."""
    uy, uz = apdl.bend_displacement(y, zeta, radius)
    return float(np.hypot(np.diff(y + uy), np.diff(zeta + uz)).sum())

def _arc_length(y, zeta):
    """Deformed arc length of the fiber at offset `y` over the span `zeta`."""
    uy, uz = apdl.bend_displacement(y, zeta, R)
    return float(np.hypot(np.diff(y + uy), np.diff(zeta + uz)).sum())

def test_surface_strain_matches_kappa_times_y():
    """
    Fiber strain in pure bending is -kappa*y.

    Measured as the stretch of a fiber offset from the neutral plane, which is what the
    imposed end displacements actually deliver to the mesh. Arc length, not chord: the
    chord of a curved fiber is shorter than the fiber, by enough (0.7% here) to look
    like a real error.
    """
    y = config.STRUT_THICKNESS_MM
    zeta = np.linspace(-0.5, 0.5, 4001)
    strain = _arc_length(y, zeta) / (zeta[-1] - zeta[0]) - 1.0
    assert strain == pytest.approx(-y / R, rel=1e-6)

def test_strain_is_antisymmetric_across_the_neutral_plane():
    """Tension on one face, equal compression on the other. y = 0 is strain-free."""
    y = config.STRUT_THICKNESS_MM
    zeta = np.linspace(-0.5, 0.5, 4001)
    span = zeta[-1] - zeta[0]
    inner = _arc_length(y, zeta) / span - 1.0
    outer = _arc_length(-y, zeta) / span - 1.0
    assert inner == pytest.approx(-outer, rel=1e-6)
    assert _arc_length(0.0, zeta) / span - 1.0 == pytest.approx(0.0, abs=1e-9)

def test_compression_shortens_the_chord_without_unbending():
    """
    The combined state has to compress and bend. The neutral axis still lands on the same
    arc - compression moves nodes along it rather than off it.
    """
    c = config.FLEX_AXIAL_COMPRESSION
    zeta = np.linspace(-3.0, 3.0, 4001)
    uy, uz = apdl.bend_displacement(0.0, zeta, R, compression=c)
    y_new, z_new = uy, zeta + uz
    assert np.allclose(np.hypot(y_new - R, z_new), R)
    arc = np.hypot(np.diff(y_new), np.diff(z_new)).sum()
    assert arc == pytest.approx((1.0 - c) * (zeta[-1] - zeta[0]), rel=1e-7)

def test_zero_compression_is_pure_bending():
    zeta = np.linspace(-2.0, 2.0, 9)
    a = apdl.bend_displacement(0.1, zeta, R, compression=0.0)
    b = apdl.bend_displacement(0.1, zeta, R)
    assert np.allclose(a, b)

def test_straighter_radius_gives_smaller_displacement():
    """Monotonicity: a gentler bend has to move the ends less."""
    zeta = 3.0
    gentle = apdl.bend_displacement(0.0, zeta, 40.0)[0]
    severe = apdl.bend_displacement(0.0, zeta, 10.0)[0]
    assert 0.0 < gentle < severe

# CENTRAL WINDOW #

def _column(n=101, length=10.0):
    z = np.linspace(0.0, length, n)
    return np.column_stack([np.zeros(n), np.zeros(n), z])

def test_window_keeps_the_middle_and_drops_both_ends():
    points = _column()
    keep = loadsteps.central_window(points, fraction=0.5)
    z = points[:, 2]
    assert not keep[0] and not keep[-1]
    assert keep[len(z) // 2]
    assert z[keep].min() == pytest.approx(2.5)
    assert z[keep].max() == pytest.approx(7.5)

def test_window_fraction_one_keeps_everything():
    assert loadsteps.central_window(_column(), fraction=1.0).all()

def test_window_default_matches_the_deck_constant():
    points = _column()
    assert np.array_equal(loadsteps.central_window(points),
                          loadsteps.central_window(points, apdl.BEND_SAMPLE_FRACTION))

def test_a_over_lim_restricted_to_a_window_is_a_fraction_of_that_window():
    """
    A node over the limit outside the window shouldn't count, and the denominator has to
    shrink with the window, otherwise the bending arm's A_over_lim is diluted by material
    it never measured, which would make every design look better than it is.
    """
    from tests.test_scalars import unit_cube_stack

    points, hexes = unit_cube_stack(n=4)
    amp = np.zeros(points.shape[0])
    keep = loadsteps.central_window(points, fraction=0.5)

    amp[~keep] = 1.0            # Everything outside the window is over the limit
    assert loadsteps.a_over_lim_3d(amp, points, hexes, limit=0.5, keep=keep) == 0.0

    amp[:] = 0.0
    amp[keep] = 1.0             # Everything inside it is
    assert loadsteps.a_over_lim_3d(amp, points, hexes, limit=0.5, keep=keep) == pytest.approx(1.0)

# DECK GENERATION #

def test_deck_carries_the_config_load_and_no_cyclic_symmetry():
    deck = apdl.flex_cycle('s9_4_mesh')
    assert f'RBEND = {config.FLEX_BEND_AMPLITUDE_RADIUS_MM}' in deck
    assert 'CAXIAL = 0.0' in deck or 'CAXIAL = 0' in deck
    # Bending is not axisymmetric; rotating nodal DOFs into cylindrical would break it.
    assert 'CSYS,1' not in deck
    assert 'NROTAT' not in deck
    # The mesh include ends in FINISH, so the deck must re-enter PREP7 after /INPUT.
    assert deck.index('/PREP7') > deck.index("/INPUT,'s9_4_mesh'")

def test_combined_arm_differs_only_in_the_load():
    bend = apdl.flex_cycle('m', compression=0.0)
    comb = apdl.flex_cycle('m', compression=config.FLEX_AXIAL_COMPRESSION)
    assert f'CAXIAL = {config.FLEX_AXIAL_COMPRESSION}' in comb
    differing = [a for a, b in zip(bend.splitlines(), comb.splitlines()) if a != b]
    assert len(differing) == 4
    assert all(d.startswith(('!', '/TITLE', 'CAXIAL')) for d in differing)

def test_deck_writes_both_raw_per_node_dumps():
    deck = apdl.flex_cycle('m', out_stem='s9_4_flex')
    assert '/OUTPUT,s9_4_flex_amp,txt' in deck
    assert '/OUTPUT,s9_4_flex_zloc,txt' in deck

def test_every_generated_input_path_ends_in_a_separator():
    import re

    sep = chr(92)
    decks = {'flex bend': apdl.flex_cycle('m', deck_dir='C:' + sep + 'stent'),
             'flex comb': apdl.flex_cycle('m', deck_dir='C:' + sep + 'stent',
                                          compression=0.09),
             'spike_a_model': apdl.spike_a_model(deck_dir='C:' + sep + 'stent'),
             'spike_a_loadsteps': apdl.spike_a_loadsteps(deck_dir='C:' + sep + 'stent')}
    for name, deck in decks.items():
        for path in re.findall(r"/INPUT,'[^']*','inp','([^']*)'", deck):
            assert path.endswith(sep), f'{name}: /INPUT path {path!r} has no separator'

def test_flex_only_is_the_default_and_has_no_release_step():
    """
    The default deck stops at the flexed state and treats released as stress-free. The deck
    imposes the amplitude curvature directly, so the flexed state is the amplitude.
    """
    deck = apdl.flex_cycle('m')
    assert 'TIME,2' not in deck
    assert deck.count('SOLVE') == 1
    assert 'EPS2W = 0' in deck and 'EPS2 = 0' in deck
    # A2 hsa to stay at its *DIM zeros, so AD is the range measured from zero.
    assert '*DIM,A2,ARRAY' in deck
    assert '*VGET,A2(1)' not in deck
    assert '*VOPER,AD(1),A1(1),SUB,A2(1)' in deck
    # AD is already an amplitude here, so the deck shouldn't halve it either.
    assert 'EPSAMP = 1.0*EPSRANGE' in deck

def test_release_arm_reports_the_range_halved():
    assert 'EPSAMP = 0.5*EPSRANGE' in apdl.flex_cycle('m', release=True)

def test_release_arm_still_available_for_the_residual_measurement():
    deck = apdl.flex_cycle('m', release=True)
    assert deck.count('SOLVE') == 2
    assert 'TIME,2' in deck
    assert '*VGET,A2(1),NODE,1,EPTO,1' in deck
    assert '*GET,EPS2W,SORT,,MAX' in deck

def test_release_flag_changes_nothing_about_the_load():
    """The grip and the imposed map have to be identical, or the two are not comparable."""
    a = apdl.flex_cycle('m').splitlines()
    b = apdl.flex_cycle('m', release=True).splitlines()
    for deck in (a, b):
        assert f'RBEND = {config.FLEX_BEND_AMPLITUDE_RADIUS_MM}' in deck
    load = lambda d: d[d.index('CMSEL,S,ENDS'):d.index('NSUBST,100,2000,50')]
    assert load(a) == load(b)

def test_deck_imposes_the_amplitude_curvature_not_the_full_excursion():
    """
    At the full 1/27 the extreme compressive fiber reaches ~11.7% and dense cells locally
    buckle. Imposed fiber strain is -kappa*y, so half the curvature is exactly half the
    strain and that half is the amplitude.
    """
    R = config.FLEX_BEND_AMPLITUDE_RADIUS_MM
    assert R == pytest.approx(2.0 * config.FLEX_BEND_RADIUS_MM)
    assert R == pytest.approx(1.0 / config.FLEX_BEND_AMPLITUDE)
    assert f'RBEND = {R}' in apdl.flex_cycle('m')

    zeta = np.linspace(-0.5, 0.5, 4001)
    span = zeta[-1] - zeta[0]
    for yi in np.linspace(-3.22, 3.22, 9):
        full = _arc_length_at(yi, zeta, config.FLEX_BEND_RADIUS_MM) / span - 1.0
        half = _arc_length_at(yi, zeta, R) / span - 1.0
        assert half == pytest.approx(0.5 * full, rel=1e-4)

def test_amplitude_mode_must_not_be_halved_again(tmp_path):
    """
    The dump is a RANGE only when there is a release state. Halving an amplitude-mode dump
    would silently report half the real strain, which no other check would catch.
    """
    from tests.test_scalars import unit_cube_stack

    points, hexes = unit_cube_stack(n=4)
    eps = np.full(points.shape[0], 0.01)
    dump = tmp_path / 'amp.txt'
    dump.write_text(chr(10).join(f'{v:16.8E}' for v in eps), encoding='utf-8')
    status = tmp_path / 'status.txt'
    status.write_text('', encoding='utf-8')

    direct = loadsteps.flex_scalars(status, dump, points, hexes, halve=False)
    halved = loadsteps.flex_scalars(status, dump, points, hexes, halve=True)
    assert direct['eps_a_max_bend'] == pytest.approx(0.01)
    assert halved['eps_a_max_bend'] == pytest.approx(0.005)
    assert loadsteps.flex_scalars(status, dump, points, hexes)['eps_a_max_bend']  == pytest.approx(0.01)

# S9.4 RANK AUDIT #

def _audit_rows(axial, bend, grip=None, converged=None):
    """n synthetic cells; f_metal is spread evenly so the grip regression has an x-axis."""
    n = len(axial)
    return [{'pool_index': i, 'stem': f'c{i}', 'f_metal': 0.30 + 0.02 * i,
             'axial_A_over_lim': axial[i], 'axial_eps_a_max': axial[i],
             'axial_eps_a_p99': axial[i],
             'bend_A_over_lim': bend[i], 'bend_eps_a_max': bend[i],
             'bend_eps_a_p99': bend[i],
             'bend_grip_inflation': None if grip is None else grip[i],
             'bend_converged': True if converged is None else converged[i]}
            for i in range(n)]

def test_false_negative_rate_is_the_complement_of_retention():
    rows = _audit_rows([1, 2, 3, 4, 5, 6], [2, 1, 9, 3, 8, 7])
    s = flex.rank_audit(rows, ['bend'])['arms']['bend']['metrics']['A_over_lim']
    assert s['retention'] + s['false_negative_rate'] == pytest.approx(1.0)

def test_perfect_agreement_is_no_false_negatives():
    # A_over_lim is lower-is-better, so agreement means the same order.
    rows = _audit_rows([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    s = flex.rank_audit(rows, ['bend'])['arms']['bend']['metrics']['A_over_lim']
    assert s['rho'] == pytest.approx(1.0)
    assert s['false_negative_rate'] == pytest.approx(0.0)
    assert s['passes']

def test_reversed_ranking_misses_every_good_design():
    rows = _audit_rows([1, 2, 3, 4, 5, 6], [60, 50, 40, 30, 20, 10])
    s = flex.rank_audit(rows, ['bend'])['arms']['bend']['metrics']['A_over_lim']
    assert s['rho'] == pytest.approx(-1.0)
    assert s['false_negative_rate'] == pytest.approx(1.0)
    assert not s['passes']

def test_unconverged_cells_are_counted_not_dropped():
    """A topology the 3D tier can't solve is data about the envelope."""
    rows = _audit_rows([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                       converged=[True, True, True, True, False, False])
    per = flex.rank_audit(rows, ['bend'])['arms']['bend']
    assert per['n'] == 4 and per['n_missing'] == 2
    assert per['metrics']['A_over_lim']['n'] == 4

def test_grip_regression_catches_a_density_correlated_artifact():
    """
    An inflation that tracks f_metal biases the ranking along the same axis
    as the metric, so a fail could be the boundary condition rather than the physics.
    """
    rows = _audit_rows([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                       grip=[1.0 + 0.02 * i for i in range(6)])
    s = flex.grip_vs_density(rows, ['bend'])['bend']
    assert s['rho'] == pytest.approx(1.0)
    # f_metal steps 0.02 per cell and so does the grip, hence unit slope.
    assert s['slope_per_f_metal'] == pytest.approx(1.0)
    assert s['spread'] == pytest.approx(0.10)

def test_grip_regression_declines_to_regress_on_too_few_cells():
    rows = _audit_rows([1, 2], [1, 2], grip=[1.0, 1.2])
    assert flex.grip_vs_density(rows, ['bend'])['bend'] == {'n': 2}

def test_both_arms_are_loaded_at_the_amplitude_not_the_excursion():
    deck = apdl.flex_cycle('m', compression=config.FLEX_AXIAL_AMPLITUDE)
    assert f'CAXIAL = {config.FLEX_AXIAL_AMPLITUDE}' in deck
    assert config.FLEX_AXIAL_AMPLITUDE == pytest.approx(config.FLEX_AXIAL_COMPRESSION / 2)
    # Both arms halve the same walking excursion, so the ratio of nominal strains is 1:1.
    assert (config.FLEX_AXIAL_AMPLITUDE / config.FLEX_AXIAL_COMPRESSION
            == pytest.approx(config.FLEX_BEND_AMPLITUDE / config.FLEX_BEND_CURVATURE))

def test_decks_and_runner_are_path_independent():
    deck = apdl.flex_cycle('s9_4_cell02_mesh', deck_dir='')
    assert "/INPUT,'s9_4_cell02_mesh','inp'\n" in deck
    import re
    assert not re.search('[A-Za-z]:[' + chr(92) + '/]', deck)

def test_an_explicit_deck_dir_still_works_for_local_runs():
    deck = apdl.flex_cycle('m', deck_dir='C:' + chr(92) + 'tmp')
    assert "/INPUT,'m','inp','C:" + chr(92) + "tmp" + chr(92) + "'" in deck

def test_runner_takes_its_folder_from_its_own_location():
    man = {'cells': [{'stem': 's9_4_cell02'}], 'arms': ['bend']}
    import tempfile, pathlib as _pl
    with tempfile.TemporaryDirectory() as d:
        out = _pl.Path(d) / 'run_flex.ps1'
        flex.write_vm_script(man, path=out)
        s = out.read_text(encoding='utf-8')
    assert '$dir = $PSScriptRoot' in s
    assert 'C:' + chr(92) + 'stent' not in s
    assert '-np 4' in s

def test_runner_backs_each_result_off_the_vm_as_it_lands():
    man = {'cells': [{'stem': 's9_4_cell02'}], 'arms': ['bend']}
    import tempfile, pathlib as _pl
    with tempfile.TemporaryDirectory() as d:
        out = _pl.Path(d) / 'run_flex.ps1'
        flex.write_vm_script(man, path=out)
        s = out.read_text(encoding='utf-8')
        flex.write_vm_script(man, path=out, backup='')
        none = out.read_text(encoding='utf-8')
    assert 'Test-Path $backup' in s and 'Copy-Item' in s
    # The copy is inside the exit-0 branch, so a failed job never backs up a bad result.
    assert s.index('"exit 0" | Out-File') < s.index('Copy-Item') < s.index('} else {')
    assert 'catch' in s                      # non-fatal
    assert '$backup = ""' in none            # opt out cleanly

def test_volume_weighted_percentile_ignores_node_density():
    """
    The same metal meshed finely can't outvote itself.
    Here 90% of the volume is at 0.001 but most of the nodes are at 0.05, so an unweighted
    percentile would report the high value and a weighted one shouldn't.
    """
    amp = np.array([0.001, 0.05, 0.05, 0.05, 0.05])
    vol = np.array([0.90, 0.025, 0.025, 0.025, 0.025])
    assert loadsteps.vol_weighted_percentile(amp, vol, 0.5) == pytest.approx(0.001)
    assert np.percentile(amp, 50) == pytest.approx(0.05)      # what we are avoiding
    assert loadsteps.vol_weighted_percentile(amp, vol, 0.99) == pytest.approx(0.05)

def test_percentile_is_monotone_in_q():
    rng = np.random.default_rng(0)
    amp, vol = rng.random(500), rng.random(500) + 0.1
    qs = [loadsteps.vol_weighted_percentile(amp, vol, q) for q in (0.5, 0.95, 0.99)]
    assert qs == sorted(qs)

def test_audit_ranks_the_percentile_as_well_as_the_peak():
    rows = [{'f_metal': 0.3 + 0.01 * i, 'bend_converged': True,
             'axial_A_over_lim': i, 'axial_eps_a_max': i, 'axial_eps_a_p99': i,
             'bend_A_over_lim': i, 'bend_eps_a_max': -i, 'bend_eps_a_p99': i}
            for i in range(6)]
    m = flex.rank_audit(rows, ['bend'])['arms']['bend']['metrics']
    assert set(m) == {'A_over_lim', 'eps_a_p99', 'eps_a_max'}
    assert m['eps_a_p99']['rho'] == pytest.approx(1.0) and m['eps_a_p99']['passes']
    assert m['eps_a_max']['rho'] == pytest.approx(-1.0) and not m['eps_a_max']['passes']
