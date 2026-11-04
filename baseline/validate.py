"""
Reproduces the S10.1 validation record at the real grid.

    python -m baseline.validate -> baseline/results/s10_1_validation.json

`tests/test_optimize.py` already asserts all of this.
"""

import json
import time

import numpy as np

import config
from baseline import mbb, optimize, simp
from geom import validity

OUT = config.PROJECT_ROOT / 'baseline' / 'results' / 's10_1_validation.json'

def main():
    started = time.perf_counter()
    e0, nu = config.NITINOL['E_austenite_MPa'], config.NITINOL['poisson_austenite']
    record = {'grid_n': config.GRID_N, 'penal': simp.PENAL}

    # (a) At rho = 1 the density solve must be the solid material.
    solid = simp.homogenize_density(np.ones((config.GRID_N, config.GRID_N)))
    exact = e0 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    record['a_solid'] = {
        'max_rel_error': float(np.abs(solid.C_eff - exact).max() / exact[0, 0]),
        'bulk': simp.bulk(solid.C_eff),
        'bulk_closed_form': e0 / (2.0 * (1.0 - nu)),
    }
    print('(a) solid cell   max rel error {max_rel_error:.2e}   bulk {bulk:.3f} vs closed form '
          '{bulk_closed_form:.3f}'.format(**record['a_solid']), flush=True)

    # (b) The top88 MBB beam, at the paper's own mesh and filter radius.
    beam = mbb.run(iters=200, beta_max=0.0, tol=1e-4)
    record['b_mbb'] = {
        'compliance': beam.value, 'published': mbb.PUBLISHED,
        'rel_error': beam.value / mbb.PUBLISHED - 1.0, 'volume': beam.volume,
        'nelx': mbb.NELX, 'nely': mbb.NELY, 'radius': mbb.RADIUS,
        'iterations': len(beam.history), 'seconds': beam.seconds,
    }
    print('(b) MBB beam     c {compliance:.3f}   published {published}   '
          '{rel_error:+.2%}   vol {volume:.4f}   {seconds:.0f}s'
          .format(**record['b_mbb']), flush=True)

    # (c) Max bulk modulus against the Hashin-Shtrikman upper bound.
    volfrac = 0.5
    cell = optimize.optimize_cell(volfrac, objective='bulk', iters=60)
    bound = simp.hs_bulk_bound(volfrac)
    record['c_hashin_shtrikman'] = {
        'volfrac': volfrac, 'bulk': cell.value, 'bound': bound, 'ratio': cell.value / bound,
        'gray': cell.history[-1]['gray'], 'seconds': cell.seconds,
    }
    print('(c) max bulk     K {bulk:.1f}   HS bound {bound:.1f}   ratio {ratio:.3f}   '
          'gray {gray:.3f}   {seconds:.0f}s'.format(**record['c_hashin_shtrikman']), flush=True)

    # The finding that gated Part 3: the unconstrained optimum for hoop
    # stiffness has no axial load path, so it's not a stent.
    record['k_radial'] = []
    for volfrac in (0.25, 0.40):
        run = optimize.optimize_cell(volfrac, iters=60)
        check = validity.check(optimize.to_cell(run.rho))
        record['k_radial'].append({
            'volfrac': volfrac, 'K_radial': run.value,
            'f_metal': float((run.rho > 0.5).mean()), 'valid': bool(check.ok),
            'reasons': list(check.reasons), 'gray': run.history[-1]['gray'],
            'seconds': run.seconds,
        })
        print('    k_radial vf {volfrac}   K_radial {K_radial:.3f}   f_metal {f_metal:.3f}   '
              'valid {valid} {reasons}   {seconds:.0f}s'
              .format(**record['k_radial'][-1]), flush=True)

    record['seconds'] = time.perf_counter() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))
    print('Wrote', OUT)

if __name__ == '__main__':
    main()
