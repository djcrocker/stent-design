"""Cell generation and the coverage subsample."""

import numpy as np
import pytest

import config
from diffusion import dataset, sampler
from geom import validity
from geom.cell import UnitCell

@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(0)

def test_random_field_hits_its_target_metal_fraction():
    """Quantile thresholding is what keeps f_metal inside the validity band by construction."""
    r = np.random.default_rng(1)
    for target in (0.15, 0.30, 0.45):
        arr = sampler.random_field(r, target_f_metal=target)
        assert arr.mean() == pytest.approx(target, abs=0.02)

def test_random_field_rarely_wraps():
    """Documents why random_field was demoted."""
    r = np.random.default_rng(2)
    ok = sum(validity.check(UnitCell(sampler.random_field(r))).ok for _ in range(40))
    assert ok <= 4

def test_random_lattice_is_far_more_likely_to_be_valid():
    """The lattice builds connectivity and wrapping into the construction instead."""
    r = np.random.default_rng(3)
    ok = sum(validity.check(UnitCell(sampler.random_lattice(r))).ok for _ in range(40))
    assert ok >= 8

def test_random_lattice_respects_the_requested_grid():
    r = np.random.default_rng(4)
    assert sampler.random_lattice(r, n=32).shape == (32, 32)

def test_perturbation_actually_changes_the_cell(rng):
    from geom import reference
    cell = reference.build()
    changed = sampler.perturb(cell, rng)
    assert changed.shape == cell.to_array().shape
    assert (changed != cell.to_array()).any()

def test_pack_roundtrips_exactly():
    from geom import handmade
    cells = [f() for f in handmade.VALID_CELLS.values()]
    packed, shape = sampler.pack(cells)
    back = sampler.unpack(packed, shape)
    assert (back == np.stack([c.to_array() for c in cells])).all()
    assert packed.nbytes == shape[0] * shape[1] * shape[2] // 8

def test_generate_pool_returns_only_valid_cells():
    cells, meta, stats = sampler.generate_pool(40, seed=5)
    assert len(cells) == 40
    assert all(validity.check(c).ok for c in cells)
    assert len(meta) == len(cells)
    assert set(m['source'] for m in meta) <= set(sampler.SOURCES)

def test_generate_pool_reports_yield_per_source():
    _, _, stats = sampler.generate_pool(40, seed=6)
    for source, attempts in stats['attempts'].items():
        if attempts:
            assert 0.0 <= stats['yield'][source] <= 1.0

def test_generate_pool_is_deterministic_for_a_seed():
    a, _, _ = sampler.generate_pool(15, seed=7)
    b, _, _ = sampler.generate_pool(15, seed=7)
    assert all((x.to_array() == y.to_array()).all() for x, y in zip(a, b))

def _labels(n, spread=True):
    rng = np.random.default_rng(0)
    if spread:
        return [{'K_radial': float(10 ** rng.uniform(0, 2)),
                 'eps_a_max': float(rng.uniform(0.02, 0.12)),
                 'A_over_lim': float(rng.uniform(0.05, 0.9)),
                 'f_metal': float(rng.uniform(0.2, 0.5))} for _ in range(n)]
    return [{'K_radial': 10.0, 'eps_a_max': 0.05, 'A_over_lim': 0.4, 'f_metal': 0.3}
            for _ in range(n)]

def test_coverage_subsample_returns_everything_when_target_exceeds_the_pool():
    labels = _labels(20)
    assert len(dataset.coverage_subsample(labels, 50)) == 20

def test_coverage_subsample_keeps_more_bins_than_a_random_draw():
    """Justification for subsampling instead of taking a random slice."""
    labels = _labels(1500)
    target = 400
    cov = dataset.coverage_subsample(labels, target, n_bins=6)
    rand = np.random.default_rng(1).choice(len(labels), target, replace=False)

    def bins(sel):
        idx = dataset.bin_indices([labels[i] for i in sel], n_bins=6)
        return len({tuple(r) for r in idx})

    assert len(cov) == target
    assert bins(cov) > bins(rand)

def test_coverage_subsample_survives_a_degenerate_objective():
    """All-identical labels give one bin; it must still return the requested count."""
    labels = _labels(50, spread=False)
    assert len(dataset.coverage_subsample(labels, 20)) == 20

def test_bin_indices_uses_a_log_scale_for_k_radial():
    """K_radial spans two decades, so linear bins would put nearly everything in bin 0."""
    labels = [{'K_radial': v, 'eps_a_max': 0.05, 'A_over_lim': 0.4, 'f_metal': 0.3}
              for v in (1.0, 10.0, 100.0)]
    col = dataset.bin_indices(labels, n_bins=4)[:, 0]
    assert len(set(col.tolist())) == 3

def test_label_one_produces_the_full_objective_vector():
    from geom import reference
    out = dataset.label_one(reference.build().to_array())
    assert set(dataset.Y_KEYS) <= set(out)
    assert out['K_radial'] > 0
    assert 0.0 <= out['f_metal'] <= 1.0
