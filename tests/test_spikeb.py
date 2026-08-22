"""Spike B cell set and its batch machinery."""

import numpy as np
import pytest

import config
from sim3d import spikeb
from geom import parametric
from sim3d import mesh3d

@pytest.fixture(scope='module')
def specs():
    return spikeb.select_cells()

def test_cell_set_is_the_expected_size_and_mix(specs):
    families = [s['family'] for s in specs]
    assert families.count('crown') == spikeb.N_PARAMETRIC
    assert families.count('handmade') == 5
    assert families.count('reference') == 1
    assert len(specs) == spikeb.N_PARAMETRIC + 6

def test_stratify_sample_fits_within_the_valid_sweep():
    valid, _ = parametric.sweep()
    assert spikeb.N_PARAMETRIC <= len(valid)

def test_names_are_unique_and_stable(specs):
    names = [s['name'] for s in specs]
    assert len(set(names)) == len(names)
    assert [s['name'] for s in spikeb.select_cells()] == names

def test_stratify_covers_every_axis_of_the_grid(specs):
    """Taking the first n of the sweep would cluster in one corner instead."""
    crown = [s for s in specs if s['family'] == 'crown']
    assert len({s['params']['strut_width_mm'] for s in crown}) == 6
    assert len({s['params']['crown_amplitude'] for s in crown}) == 6
    assert len({s['params']['n_periods'] for s in crown}) == 3

def test_stratify_beats_a_naive_prefix_on_spread():
    valid, _ = parametric.sweep()
    X = spikeb._param_matrix(valid)
    picked = X[spikeb.stratify(valid, 24)]
    prefix = X[:24]
    def mean_pairwise(a):
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        return d[np.triu_indices(len(a), 1)].mean()
    assert mean_pairwise(picked) > mean_pairwise(prefix)

def test_stratify_returns_everything_when_n_exceeds_the_pool():
    valid, _ = parametric.sweep()
    assert spikeb.stratify(valid, len(valid) + 5) == list(range(len(valid)))

def test_every_mesh_clears_the_ansys_face_angle_limit(specs):
    for spec in specs[::7]:
        pts, hexes, _ = mesh3d.tube_hex_mesh(spec['cell'], **spikeb.MESH)
        q = mesh3d.quality(pts, hexes)
        assert q['max_face_angle_deg'] < spikeb.MAX_FACE_ANGLE_DEG
        assert q['n_nonpositive'] == 0

def test_the_old_clamp_would_have_failed():
    """Guards the finding itself, so a later default change can't undo it."""
    worst = 0.0
    for spec in spikeb.select_cells()[:8]:
        pts, hexes, _ = mesh3d.tube_hex_mesh(spec['cell'], n_circ=1, n_axial=2, layers=4, limit=0.30)
        worst = max(worst, mesh3d.quality(pts, hexes)['max_face_angle_deg'])
    assert worst > spikeb.MAX_FACE_ANGLE_DEG

def test_label_2d_returns_the_screen_metrics(specs):
    labels = spikeb.label_2d(specs[0]['cell'])
    assert set(labels) == {'K_radial_2D', 'eps_a_max_2D', 'eps_a_p99_2D',
                           'A_over_lim_2D', 'f_metal'}
    assert labels['K_radial_2D'] > 0
    assert config.F_METAL_MIN <= labels['f_metal'] <= config.F_METAL_MAX

def test_batch_script_lists_every_cell_and_keeps_jobs_independent(specs, tmp_path):
    path = spikeb.write_batch_script(specs, path=tmp_path / 'run.ps1', out_dir=tmp_path / 'out')
    text = path.read_text()
    for spec in specs:
        assert spec['name'] in text
    # A failing cell has to be recorded
    assert '$LASTEXITCODE' in text
    # Timestamped
    assert 'spikeb_timing_' in text
    assert 'spikeb_timing.csv' not in text
    assert 's6_*.lock' in text

def test_collect_marks_missing_runs_instead_of_dropping_them(specs, tmp_path):
    """A topology the 3D tier can't solve is data about the envelope."""
    subset = specs[:3]
    for s in subset:
        s['labels_2d'] = spikeb.label_2d(s['cell'])
    rows = spikeb.collect(subset, results_dir=tmp_path, out_dir=tmp_path)
    assert len(rows) == len(subset)
    assert all(r['converged'] is False for r in rows)
    assert all(r['K_radial_3D'] is None for r in rows)
    # The 2D half must still be present - it needs no Ansys.
    assert all(r['K_radial_2D'] > 0 for r in rows)
    # And nothing may be written into the tracked results directory.
    assert (tmp_path / 's6_1_labels.csv').exists()
    assert not (spikeb.RESULTS_DIR / 'tmp_labels.csv').exists()

def test_convergence_is_read_from_reached_time_not_exit_code(tmp_path):
    m = tmp_path / 'c.mntr'
    m.write_text(
        '  LOAD   SUB-  NO.  NO.    TOTL  INCREMENT    TOTAL\n'
        '     1      1    1     2      2   0.20000E-01  0.20000E-01   13.9  1.6  0.0  0.0  0.0\n'
        '     4     42    3     2    584   0.65288E-02   4.0000       939.8  1.3  0.0  0.0  0.0\n')
    c = spikeb.read_convergence(m)
    assert c['converged'] is True
    assert c['reached_time'] == pytest.approx(4.0)
    assert c['max_attempts'] == 3

def test_a_run_that_stopped_partway_is_not_converged(tmp_path):
    m = tmp_path / 'c.mntr'
    m.write_text(
        '  LOAD   SUB-  NO.  NO.    TOTL  INCREMENT    TOTAL\n'
        '     1     32    5     9    595   0.50000E-01  0.65347       3884.0  1.6  0.0  0.0  0.0\n')
    c = spikeb.read_convergence(m)
    assert c['converged'] is False
    assert c['reached_time'] == pytest.approx(0.65347)
    assert c['retries'] == 1

def test_missing_monitor_file_is_not_converged(tmp_path):
    c = spikeb.read_convergence(tmp_path / 'nope.mntr')
    assert c['converged'] is False
    assert c['reached_time'] is None

def test_retry_decks_get_finer_substepping_and_others_do_not(tmp_path):
    """
    The retry exists because failures aren't random: they skew to high n_periods, high
    strain and high f_metal.
    """
    from sim3d import apdl
    default = apdl.spike_a_loadsteps(out_stem='x')
    retried = apdl.spike_a_loadsteps(out_stem='x', nsubst=spikeb.RETRY_NSUBST)
    assert 'NSUBST,20,200,10' in default
    assert f'NSUBST,{spikeb.RETRY_NSUBST}' in retried
    assert 'NSUBST,20,200,10' not in retried

def test_failed_cells_reads_from_monitor_files(tmp_path):
    specs = spikeb.select_cells()[:2]
    # No monitor files at all -> every cell counts as failed, not as silently fine.
    assert spikeb.failed_cells(results_dir=tmp_path, specs=specs) == [s['name'] for s in specs]
