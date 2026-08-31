"""
A stent that varies along its own length.

We've established that two differently-conditioned cells stack when they share an interface
signature. This builds the device: walk `y` along the axis and generate the cell
sequence that follows it, every cell pinned to the same `sigma` so consecutive cells join.

This is the thing inverse homogenization can't express. Past work has optimized one cell
and repeated it, so their device is uniform by construction. The FPA isn't one loading
environment (peak axial compression runs SFA 9 % / adductor hiatus 11 % / popliteal 13 %
during walking), and a long stent spans more than one of those.

We measured ~34% of pinned pairs as usable, so a chain built by taking the first draw at each 
position would almost never survive. Each position draws a pool, keeps the cells that are 
individually valid, and then picks the one that actually joins the cell already chosen below it.
That is a beam of width 1, which is enough because the join rate among valid pinned cells is 
high (83%); the cost is the pool, not the search.

Two phases:
    python -m diffusion.graded sample
    python -m diffusion.graded analyze
"""

import argparse
import json

import numpy as np

import config
from diffusion import interface

RESULTS_DIR = config.PROJECT_ROOT / 'diffusion' / 'results'
N_CELLS = 6        # 6 x 1.5708 mm = 9.42 mm of stent
POOL = 24          # Candidates drawn per axial position
GUIDANCE = interface.GUIDANCE
RESAMPLE = interface.RESAMPLE
BAND_ROWS = interface.BAND_ROWS

def gradient(frame, n_cells=N_CELLS, key='K_radial', lo=120.0, hi=380.0):
    """
    The `y` trajectory along the axis, one target per cell.

    A monotone ramp in one component, with the rest completed from the conditional median
    at each step.
    """
    from diffusion import generate

    out = []
    for i, value in enumerate(np.linspace(lo, hi, n_cells)):
        asked = {key: float(value)}
        full, support = generate.complete_target(frame, asked)
        out.append({'index': i, 'asked': asked, 'target': full, 'support': int(support)})
    return out

def pick_next(candidates, below, rows=BAND_ROWS):
    """
    The first candidate that both is valid and joins the cell already placed below it.

    Returns (index, report) or (None, None). `below` is None at the first position, where
    only validity is required.
    """
    from geom import validity
    from geom.cell import UnitCell

    for i, cand in enumerate(candidates):
        if not validity.check(UnitCell(cand)).ok:
            continue
        if below is None:
            return i, None
        rep = interface.interface_report([below, cand])
        if rep['joined']:
            return i, rep
    return None, None

def sample_phase(n_cells=N_CELLS, pool=POOL, guidance=GUIDANCE, steps=50, seed=0,
                 band_rows=BAND_ROWS, resample=RESAMPLE, key='K_radial',
                 lo=120.0, hi=380.0, out_stem='s11_2_graded'):
    """Torch only. Build the chain, and a uniform control for comparison."""
    import torch

    from diffusion import dataset
    from diffusion.sample import load_model, to_fields

    ddpm, norm, _ = load_model()
    _, frame = dataset.load()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shape = (1, config.GRID_N, config.GRID_N)

    targets = gradient(frame, n_cells, key, lo, hi)
    print(f'{n_cells} cells, {key} {lo:.0f} -> {hi:.0f}, pool {pool} per position',
          flush=True)

    torch.manual_seed(seed)

    def draw(target, known=None, mask=None, n=pool):
        z = torch.tensor(norm.transform_dict(target), dtype=torch.float32)
        y = z[None].repeat(n, 1).to(ddpm.device)
        return to_fields(ddpm.ddim_sample(n, y, guidance=guidance, steps=steps,
                                          shape=shape, known=known, known_mask=mask,
                                          resample=resample))

    # sigma comes from a free draw at the middle of the ramp, so it is representative of
    # the whole chain rather than of either end.
    mid = targets[n_cells // 2]
    sigma, sig_info = interface.choose_sigma(list(draw(mid['target'])), band_rows)
    print(f"  sigma: density {sig_info['band_density']:.3f} "
          f"(median {sig_info['median_density']:.3f}), "
          f"coverage {sig_info['column_coverage']:.2f}", flush=True)

    known = torch.zeros(pool, *shape)
    mask = torch.zeros(pool, *shape)
    known[:, 0, :band_rows] = torch.from_numpy(sigma.astype(np.float32)) * 2.0 - 1.0
    mask[:, 0, :band_rows] = 1.0

    chain, chosen_from, misses = [], [], 0
    for t in targets:
        cands = draw(t['target'], known, mask)
        idx, _ = pick_next(cands, chain[-1] if chain else None, band_rows)
        if idx is None:
            misses += 1
            print(f"  cell {t['index']}: NO CANDIDATE joined out of {pool}", flush=True)
            # Fall back to any valid cell so the chain still reports honestly; the
            # interface check in analyze() will show the break rather than hide it.
            from geom import validity
            from geom.cell import UnitCell
            ok = [c for c in cands if validity.check(UnitCell(c)).ok]
            if not ok:
                raise RuntimeError(f"position {t['index']}: no valid cell at all")
            chain.append(ok[0])
            chosen_from.append(-1)
        else:
            chain.append(cands[idx])
            chosen_from.append(int(idx))
            print(f"  cell {t['index']}: candidate {idx + 1}/{pool}", flush=True)

    # Uniform control: the same chain length, one target, one repeated cell.
    uniform_target = targets[n_cells // 2]
    ucands = draw(uniform_target['target'], known, mask)
    uidx, _ = pick_next(ucands, None, band_rows)
    uniform = np.array([ucands[uidx if uidx is not None else 0]] * n_cells)

    chain = np.array(chain)
    np.savez_compressed(
        RESULTS_DIR / f'{out_stem}.npz',
        chain=np.packbits(chain, axis=None), chain_shape=np.array(chain.shape),
        uniform=np.packbits(uniform, axis=None), uniform_shape=np.array(uniform.shape),
        sigma=np.packbits(sigma, axis=None), sigma_shape=np.array(sigma.shape),
        band_rows=np.array(band_rows),
    )
    (RESULTS_DIR / f'{out_stem}_targets.json').write_text(
        json.dumps({'targets': targets, 'key': key, 'lo': lo, 'hi': hi,
                    'n_cells': n_cells, 'pool': pool, 'guidance': guidance,
                    'steps': steps, 'resample': resample, 'band_rows': band_rows,
                    'sigma': sig_info, 'chosen_from': chosen_from,
                    'positions_with_no_join': misses}, indent=2),
        encoding='utf-8')
    print(f'  wrote {out_stem}.npz  ({misses} position(s) without a join)', flush=True)

def load_chain(out_stem='s11_2_graded'):
    blob = np.load(RESULTS_DIR / f'{out_stem}.npz')

    def unpack(key, shape_key):
        shape = tuple(blob[shape_key])
        return np.unpackbits(blob[key], count=int(np.prod(shape))).reshape(shape).astype(bool)

    meta = json.loads((RESULTS_DIR / f'{out_stem}_targets.json').read_text(encoding='utf-8'))
    return (unpack('chain', 'chain_shape'), unpack('uniform', 'uniform_shape'),
            unpack('sigma', 'sigma_shape'), meta)

def analyze(out_stem='s11_2_graded'):
    """
    No torch. Two questions:
    Does it hold together?  Every interface joined, the whole stack one component, wrapping
                            both ways, and every cell individually valid.
    Does it grade?          The labeled property of each cell should track the ramp it was
                            asked for. A chain that assembles but does not vary is a
                            uniform stent with extra steps.
    """
    from diffusion.dataset import label_one
    from geom import validity
    from geom.cell import UnitCell

    chain, uniform, sigma, meta = load_chain(out_stem)
    key = meta['key']
    asked = [t['asked'][key] for t in meta['targets']]

    whole = interface.interface_report(list(chain))
    per_cell = [validity.check(UnitCell(c)) for c in chain]
    labels = [label_one(c) if v.ok else None for c, v in zip(chain, per_cell)]
    got = [None if l is None else l[key] for l in labels]

    pairs = [interface.interface_report([chain[i], chain[(i + 1) % len(chain)]])
             for i in range(len(chain))]

    print(f"{len(chain)} cells, {key} {meta['lo']:.0f} -> {meta['hi']:.0f}, "
          f"band {meta['band_rows']} rows")
    print('\ncell | asked | achieved | f_metal | valid | crossings to next')
    for i in range(len(chain)):
        g = 'n/a' if got[i] is None else f'{got[i]:8.1f}'
        fm = f"{chain[i].mean():.3f}"
        print(f'{i:4d} | {asked[i]:5.0f} | {g:>8} | {fm:>7} | '
              f'{str(per_cell[i].ok):5} | {pairs[i]["min_crossings"]:3d}')

    ok = [(a, g) for a, g in zip(asked, got) if g is not None]
    if len(ok) >= 3:
        a = np.array([x[0] for x in ok]); g = np.array([x[1] for x in ok])
        # Spearman by hand; scipy is not imported in the torch-free half.
        ra = np.argsort(np.argsort(a)); rg = np.argsort(np.argsort(g))
        rho = float(np.corrcoef(ra, rg)[0, 1])
        span = float(g.max() - g.min())
    else:
        rho, span = float('nan'), float('nan')

    report = {
        'key': key, 'n_cells': int(len(chain)),
        'asked': asked, 'achieved': got,
        'f_metal': [float(c.mean()) for c in chain],
        'all_cells_valid': bool(all(v.ok for v in per_cell)),
        'n_valid_cells': int(sum(v.ok for v in per_cell)),
        'stack_connected': whole['connected'],
        'stack_wrap_circ': whole['wrap_circ'],
        'stack_wrap_axial': whole['wrap_axial'],
        'interfaces_joined': int(sum(p['joined'] for p in pairs)),
        'min_crossings': [p['min_crossings'] for p in pairs],
        'gradient_rho': rho,
        'achieved_span': span,
        'positions_with_no_join': meta.get('positions_with_no_join', 0),
    }
    (RESULTS_DIR / f'{out_stem}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(f"\nHOLDS TOGETHER: {report['interfaces_joined']}/{len(chain)} interfaces joined, "
          f"stack connected {whole['connected']}, wraps "
          f"{whole['wrap_circ']}/{whole['wrap_axial']}, "
          f"{report['n_valid_cells']}/{len(chain)} cells valid")
    print(f"GRADES: rank correlation asked vs achieved rho = {rho:.3f}, "
          f"achieved span {span:.1f}")
    return report

if __name__ =="__main__":
    ap = argparse.ArgumentParser(description='Graded stent assembly')
    ap.add_argument('phase', choices=('sample', 'analyze'))
    ap.add_argument('--cells', type=int, default=N_CELLS)
    ap.add_argument('--pool', type=int, default=POOL)
    ap.add_argument('--key', default='K_radial')
    ap.add_argument('--lo', type=float, default=120.0)
    ap.add_argument('--hi', type=float, default=380.0)
    args = ap.parse_args()
    if args.phase == 'sample':
        sample_phase(n_cells=args.cells, pool=args.pool, key=args.key, lo=args.lo, hi=args.hi)
    else:
        analyze()
