"""
The SIMP baseline at a prescribed homogenized stiffness.

What is prescribed & why the two prescriptions have different forms:

  * `E_circ` is matched as an equality, because it's the quantity the comparison is matched
    on. `K_radial = E_circ t / R^2`, so prescribing one prescribes the other.
  * `E_axial` is held to a floor.A floor leaves the baseline more freedom than an equality 
    would.
"""

import json
import time
from collections import namedtuple

import numpy as np

import config
from baseline import optimize, simp
from geom import validity
from sim2d.homogenize import homogenize

AXIAL_FLOOR = 0.368                 # E_axial >= AXIAL_FLOOR * E_circ_target; the S3 extended family's
                                    # median E_axial/E_circ, measured 0.3679502704697908
AGGREGATE_P = 20.0                  # softmin sharpness; log(2)/p is the bias, 0.035 here
MATCH_RTOL = 0.05                   # how far below the target K_radial may fall and still count
VOLFRAC_BOUNDS = (0.10, 0.80)
K_TARGETS = (100.0, 200.0, 300.0)   # the ladder the diffusion model was conditioned on

REPAIR_RADII = ()

Prescribed = namedtuple('Prescribed', 'k_target volfrac radius run achieved bracket seconds')

def e_circ_for(k_radial):
    """The circumferential modulus a prescribed `K_radial` asks for, MPa."""
    R = config.D_DEPLOYED_MM / 2.0
    return k_radial * R ** 2 / config.STRUT_THICKNESS_MM

def binding(solved, e_target, floor=AXIAL_FLOOR, p=AGGREGATE_P):
    """
    A smooth minimum of the two normalized moduli and its gradient, to be maximized.

        g = softmin(E_circ / E*, E_axial / (floor E*))
          = lo - log(exp(-p (circ - lo)) + exp(-p (axial - lo))) / p
    """
    circ = solved.E_circ / e_target
    if floor <= 0.0:
        return circ, solved.dE(0) / e_target
    scale = floor * e_target
    axial = solved.E_axial / scale
    lo = min(circ, axial)
    w_circ, w_axial = np.exp(-p * (circ - lo)), np.exp(-p * (axial - lo))
    total = w_circ + w_axial
    grad = (w_circ * solved.dE(0) / e_target + w_axial * solved.dE(1) / scale) / total
    return float(lo - np.log(total) / p), grad

def optimize_prescribed(k_target, volfrac, n=None, floor=AXIAL_FLOOR, p=AGGREGATE_P, **kw):
    """Maximize the binding modulus at a fixed volume fraction."""
    n = config.GRID_N if n is None else n
    e_target = e_circ_for(k_target)
    if kw.get('x0') is None:    # Also catches an explicit x0=None warm start
        kw['x0'] = optimize.initial_design((n, n), volfrac)

    def grad_fn(rho):
        return binding(simp.homogenize_density(rho), e_target, floor, p)

    return optimize.optimize(grad_fn, (n, n), volfrac, **kw)

def achieved(rho, k_target, floor=AXIAL_FLOOR, binary=True):
    """What a delivered density field actually got."""
    solved = simp.homogenize_density(rho)
    e_floor = floor * e_circ_for(k_target)
    cell = optimize.to_cell(rho)
    check = validity.check(cell)
    out = {'K_radial': solved.K_radial, 'E_circ': solved.E_circ, 'E_axial': solved.E_axial,
           'ratio': solved.E_axial / solved.E_circ,
           'k_rel_error': solved.K_radial / k_target - 1.0,
           'floor_met': bool(solved.E_axial >= e_floor),
           'floor_rel': solved.E_axial / e_floor,
           'f_metal': float(cell.f_metal), 'gray': float(np.mean(4.0 * rho * (1.0 - rho))),
           'valid': bool(check.ok), 'reasons': list(check.reasons)}
    if binary:
        h = homogenize(cell)
        out['binary'] = {'K_radial': h.K_radial, 'E_circ': h.E_circ, 'E_axial': h.E_axial,
                         'ratio': h.E_axial / h.E_circ,
                         'k_rel_error': h.K_radial / k_target - 1.0}
    return out

def matched(record, rtol=MATCH_RTOL):
    """Did a run meet both prescriptions?"""
    return bool(record['k_rel_error'] >= -rtol and record['floor_met'])

def min_volume_cell(k_target, bounds=VOLFRAC_BOUNDS, steps=8, n=None, rtol=MATCH_RTOL,
                    floor=AXIAL_FLOOR, binary=True, repair=REPAIR_RADII, progress=False,
                    **kw):
    """
    The lightest cell meeting the prescription: bisect the volume fraction.

    The bisection follows the achieved K_radial. Bisecting on `matched` looks right but is 
    wrong because the inner objective has no upper branch.
    
    Achieved K_radial is monotone in volume because the inner problem is a max-stiffness
    one, so the lightest volume that still reaches the target is the bisection's own
    boundary.
    """
    started = time.perf_counter()
    lo, hi = bounds
    tried, bracket, warm = [], [], None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        run = optimize_prescribed(k_target, mid, n=n, floor=floor,
                                  x0=(None if warm is None else warm.x), **kw)
        warm = run
        record = achieved(run.rho, k_target, floor, binary=binary)
        ok = matched(record, rtol)
        # The bisection follows K_radial, NOT `ok`: see the docstring.
        reached = bool(record['K_radial'] >= k_target * (1.0 - rtol))
        tried.append((mid, run, record, ok))
        bracket.append({'volfrac': mid, 'g': run.value, 'matched': ok,
                        'reached': reached,
                        **{k: record[k] for k in ('K_radial', 'k_rel_error', 'floor_met',
                                                  'ratio', 'f_metal', 'valid')}})
        if progress:
            print(f'    vf {mid:.4f}  g {run.value:.3f}  K {record["K_radial"]:.1f} '
                  f'({record["k_rel_error"]:+.1%})  ratio {record["ratio"]:.2f}  '
                  f'valid {record["valid"]}  matched {ok}  {run.seconds:.0f}s', flush=True)
        lo, hi = (lo, mid) if reached else (mid, hi)

    hits = [t for t in tried if t[3]]
    mid, run, record, _ = (min(hits, key=lambda t: t[0]) if hits else
                           min(tried, key=lambda t: abs(t[2]['k_rel_error'])))
    radius = kw.get('radius') or optimize.default_radius(config.GRID_N if n is None else n)

    if repair and not record['valid']:
        mid, radius, run, record = _repair_length_scale(
            k_target, mid, radius, repair, n=n, floor=floor, rtol=rtol, binary=binary,
            bracket=bracket, progress=progress, best=(run, record), **kw)

    return Prescribed(k_target, mid, radius, run, record, bracket,
                      time.perf_counter() - started)

def _repair_length_scale(k_target, volfrac, radius, multipliers, n=None, floor=AXIAL_FLOOR,
                         rtol=MATCH_RTOL, binary=True, bracket=None, progress=False,
                         best=None, **kw):
    """
    Re-run the chosen volume fraction on a coarser filter until the cell passes validity.

    The density filter sets a length scale but doesn't guarantee our minimum-feature
    criterion.

    The repair re-runs at a fixed volume fraction, so the cell it returns is no lighter
    than the one it replaces; if none of the radii pass, the original is kept and its
    `reasons` carry the failure.
    """
    kw.pop('radius', None)
    run, record = best
    for multiple in multipliers:
        wide = radius * multiple
        trial = optimize_prescribed(k_target, volfrac, n=n, floor=floor, radius=wide, **kw)
        got = achieved(trial.rho, k_target, floor, binary=binary)
        if bracket is not None:
            bracket.append({'volfrac': volfrac, 'radius': wide, 'g': trial.value,
                            'repair': True, 'matched': matched(got, rtol),
                            **{k: got[k] for k in ('K_radial', 'k_rel_error', 'floor_met',
                                                   'ratio', 'f_metal', 'valid')}})
        if progress:
            print(f'    repair radius {wide:.2f} px  g {trial.value:.3f}  '
                  f'K {got["K_radial"]:.1f} ({got["k_rel_error"]:+.1%})  '
                  f'valid {got["valid"]} {got["reasons"]}  {trial.seconds:.0f}s', flush=True)
        if got['valid'] and matched(got, rtol):
            return volfrac, wide, trial, got
    return volfrac, radius, run, record

# THE CAMPAIGN #
RESULTS_DIR = config.PROJECT_ROOT / 'baseline' / 'results'

def campaign(k_targets=K_TARGETS, out_stem='s10_1_prescribed', n=None, steps=8,
             floor=AXIAL_FLOOR, progress=True, **kw):
    """
    One baseline cell per prescribed `K_radial`.

    Writes `<out_stem>.json` with every target's bracket and achieved values, and
    `<out_stem>_cells.npz` with the binary cells and the density fields they came from.
    """
    started = time.perf_counter()
    record = {'k_targets': list(k_targets), 'grid_n': config.GRID_N if n is None else n,
              'axial_floor': floor, 'aggregate_p': kw.get('p', AGGREGATE_P),
              'match_rtol': kw.get('rtol', MATCH_RTOL), 'penal': simp.PENAL,
              'volfrac_bounds': list(kw.get('bounds', VOLFRAC_BOUNDS)), 'targets': []}
    cells, fields = {}, {}

    for k_target in k_targets:
        if progress:
            print(f'K_radial {k_target:g}  (E_circ* {e_circ_for(k_target):.1f} MPa)',
                  flush=True)
        got = min_volume_cell(k_target, steps=steps, n=n, floor=floor, progress=progress,
                              **kw)
        row = {'k_target': k_target, 'volfrac': got.volfrac, 'radius_px': got.radius,
               'g': got.run.value, 'iterations': len(got.run.history),
               'bracket': got.bracket, 'seconds': got.seconds, **got.achieved}
        record['targets'].append(row)
        cells[f'k{k_target:g}'] = optimize.to_cell(got.run.rho).to_array()
        fields[f'k{k_target:g}'] = got.run.rho.astype(np.float32)
        if progress:
            print(f'  -> volfrac {got.volfrac:.4f}  radius {got.radius:.2f} px  '
                  f'K_radial {row["K_radial"]:.1f} ({row["k_rel_error"]:+.1%})  '
                  f'ratio {row["ratio"]:.3f}  f_metal {row["f_metal"]:.3f}  '
                  f'valid {row["valid"]} {row["reasons"]}', flush=True)
            if 'binary' in row:
                print(f'     thresholded cell: K_radial {row["binary"]["K_radial"]:.1f} '
                      f'({row["binary"]["k_rel_error"]:+.1%})  '
                      f'ratio {row["binary"]["ratio"]:.3f}', flush=True)

    record['seconds'] = time.perf_counter() - started
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(record, indent=2),
                                                  encoding='utf-8')
    np.savez_compressed(RESULTS_DIR / f'{out_stem}_cells.npz',
                        **{f'cell_{k}': v for k, v in cells.items()},
                        **{f'rho_{k}': v for k, v in fields.items()})
    if progress:
        print(f'\nWrote {RESULTS_DIR / f"{out_stem}.json"} and the cells '
              f'({record["seconds"]:.0f}s)', flush=True)
    return record

def main():
    import argparse

    ap = argparse.ArgumentParser(description='S10.1 Part 3: SIMP at a prescribed stiffness')
    ap.add_argument('--targets', type=float, nargs='+', default=list(K_TARGETS),
                    help='K_radial targets, N/mm^3')
    ap.add_argument('--n', type=int, default=None, help='grid (default config.GRID_N)')
    ap.add_argument('--steps', type=int, default=8,
                    help='volume-bisection steps per target')
    ap.add_argument('--floor', type=float, default=AXIAL_FLOOR,
                    help='E_axial floor as a multiple of the circumferential target')
    ap.add_argument('--iters', type=int, default=60, help='inner OC iterations')
    ap.add_argument('--out-stem', default='s10_1_prescribed',
                    help='results basename, so a sensitivity arm does not overwrite the run')
    args = ap.parse_args()
    campaign(k_targets=tuple(args.targets), out_stem=args.out_stem, n=args.n,
             steps=args.steps, floor=args.floor, iters=args.iters)

if __name__ == '__main__':
    main()
