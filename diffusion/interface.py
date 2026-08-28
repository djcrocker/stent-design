"""
Can two different cells be stacked into one stent?

A cell is periodic on a torus, so it tiles with itself. Stacking cell A above a different 
cell B isn't automatic, as the interface only carries load if metal crosses the shared edge
at the same circumferential positions in both.

Fix one interface signature `sigma` and generate every cell with `sigma` pinned at its top 
edge. Then any two such cells stack, so: each cell is axially periodic, so its own bottom row
already connects to its own top row, which is `sigma`; and the cell above presents `sigma` at
its bottom. So every interface in the chain is the one interface each cell was built to meet.

Two phases:
    python -m diffusion.interface sample
    python -m diffusion.interface analyze
"""

import argparse
import json

import numpy as np

import config

RESULTS_DIR = config.PROJECT_ROOT / 'diffusion' / 'results'
BAND_ROWS = 4                       # ~0.10 mm, one minimum feature
N_PER_TARGET = 24
GUIDANCE = 5.0
RESAMPLE = 4
# Least circumferential coverage an interface signature may have.
MIN_SIGMA_COVERAGE = 0.15

def band_of(field, rows=BAND_ROWS):
    """The interface signature: the top `rows` of a cell."""
    return np.asarray(field, dtype=bool)[:rows].copy()

def choose_sigma(fields, rows=BAND_ROWS, valid_only=True):
    """Pick an interface signature the model can actually live with."""
    from geom import validity
    from geom.cell import UnitCell

    pool = [np.asarray(f, dtype=bool) for f in fields]
    if valid_only:
        ok = [f for f in pool if validity.check(UnitCell(f)).ok]
        pool = ok or pool

    bands = [band_of(f, rows) for f in pool]
    # A band with no metal, or metal at almost no angles, carries no axial load path and
    # every cell built on it fails `no_wrap_axial`.
    usable_bands = [b for b in bands if b.any(axis=0).mean() >= MIN_SIGMA_COVERAGE]
    bands = usable_bands or bands
    dens = np.array([b.mean() for b in bands])
    # Columns carrying metal anywhere in the band.
    spread = np.array([b.any(axis=0).mean() for b in bands])

    target = float(np.median(dens))
    score = np.abs(dens - target) / max(target, 1e-9) + 0.5 * np.maximum(spread - 0.45, 0.0)
    best = int(np.argmin(score))
    return bands[best], {'band_density': float(dens[best]),
                         'median_density': float(np.median(dens)),
                         'column_coverage': float(spread[best]),
                         'n_candidates': len(pool)}

def stack(fields):
    """Stack cells along the axis, first at the top. Returns one tall array."""
    return np.concatenate([np.asarray(f, dtype=bool) for f in fields], axis=0)

def seam_rows(n_cells, n=None):
    """Row index of the first row of each cell in a stack (each is an interface)."""
    n = config.GRID_N if n is None else n
    return [i * n for i in range(n_cells)]

def interface_report(fields):
    """Is the stack one connected structure that still wraps both ways?"""
    from geom import periodic, validity

    arr = stack(fields)
    n = np.asarray(fields[0]).shape[0]
    _, n_components = periodic.label(arr)
    wrap_circ, wrap_axial = validity.wraps(arr)

    crossings = []
    for r in seam_rows(len(fields), n):
        above = arr[(r - 1) % arr.shape[0]]
        below = arr[r]
        crossings.append(int((above & below).sum()))

    return {
        'n_cells': len(fields),
        'n_components': int(n_components),
        'connected': bool(n_components == 1),
        'wrap_circ': bool(wrap_circ),
        'wrap_axial': bool(wrap_axial),
        'crossings': crossings,
        'min_crossings': int(min(crossings)) if crossings else 0,
        'joined': bool(n_components == 1 and min(crossings) > 0) if crossings else False,
    }

def sample_phase(n_per_target=N_PER_TARGET, guidance=GUIDANCE, steps=50, seed=0,
                 band_rows=BAND_ROWS, resample=RESAMPLE, out_stem='s11_1_interface'):
    """
    Torch only.

    Three pieces:
      Free     - No pinning at all.
      Pinned   - Every cell generated with the same `sigma` inpainted at its top edge.
      Shuffled - Pinned cells stacked with a cell from a different target.
    """
    import torch

    from diffusion import dataset, generate
    from diffusion.sample import load_model, to_fields

    ddpm, norm, _ = load_model()
    _, frame = dataset.load()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # A spread of targets.
    ladder = [t for t in generate.TARGET_LADDER
              if t['name'] in ('K100_A25', 'K200_A10', 'K300_A05', 'control_mid')]
    targets = []
    for spec in ladder:
        asked = {k: v for k, v in spec.items() if k in dataset.Y_KEYS}
        full, support = generate.complete_target(frame, asked)
        targets.append({'name': spec['name'], 'asked': asked, 'target': full,
                        'support': int(support)})
    print(f'{len(targets)} targets x {n_per_target} cells, band {band_rows} rows',
          flush=True)

    torch.manual_seed(seed)
    shape = (1, config.GRID_N, config.GRID_N)

    def draw(target, known=None, mask=None):
        z = torch.tensor(norm.transform_dict(target), dtype=torch.float32)
        y = z[None].repeat(n_per_target, 1).to(ddpm.device)
        x = ddpm.ddim_sample(n_per_target, y, guidance=guidance, steps=steps,
                             shape=shape, known=known, known_mask=mask,
                             resample=resample)
        return to_fields(x)

    # Free arm first: sigma is chosen from what the model actually produces, so the
    # constraint sits inside the learned distribution.
    free = {t['name']: draw(t['target']) for t in targets}
    sigma, sig_info = choose_sigma([f for t in targets for f in free[t['name']]], band_rows)
    print(f"  sigma: {int(sigma.sum())}/{sigma.size} px metal "
          f"(density {sig_info['band_density']:.3f} vs median "
          f"{sig_info['median_density']:.3f}, column coverage "
          f"{sig_info['column_coverage']:.2f})", flush=True)

    known = torch.zeros(n_per_target, *shape, dtype=torch.float32)
    mask = torch.zeros(n_per_target, *shape, dtype=torch.float32)
    known[:, 0, :band_rows] = torch.from_numpy(sigma.astype(np.float32)) * 2.0 - 1.0
    mask[:, 0, :band_rows] = 1.0

    pinned = {t['name']: draw(t['target'], known, mask) for t in targets}

    np.savez_compressed(
        RESULTS_DIR / f'{out_stem}.npz',
        sigma=np.packbits(sigma, axis=None),
        sigma_shape=np.array(sigma.shape),
        shape=np.array((n_per_target, config.GRID_N, config.GRID_N)),
        band_rows=np.array(band_rows),
        names=np.array([t['name'] for t in targets]),
        **{f'free_{t["name"]}': np.packbits(free[t['name']], axis=None) for t in targets},
        **{f'pinned_{t["name"]}': np.packbits(pinned[t['name']], axis=None) for t in targets},
    )
    (RESULTS_DIR / f'{out_stem}_targets.json').write_text(
        json.dumps({'targets': targets, 'guidance': guidance, 'steps': steps,
                    'band_rows': band_rows, 'n_per_target': n_per_target,
                    'resample': resample, 'sigma': sig_info}, indent=2),
        encoding='utf-8')
    print(f'  Wrote {out_stem}.npz', flush=True)

def load_samples(out_stem='s11_1_interface'):
    blob = np.load(RESULTS_DIR / f'{out_stem}.npz')
    shape = tuple(blob['shape'])
    names = [str(x) for x in blob['names']]
    count = int(np.prod(shape))

    def unpack(key):
        return np.unpackbits(blob[key], count=count).reshape(shape).astype(bool)

    free = {n: unpack(f'free_{n}') for n in names}
    pinned = {n: unpack(f'pinned_{n}') for n in names}
    sig_shape = tuple(blob['sigma_shape'])
    sigma = np.unpackbits(blob['sigma'], count=int(np.prod(sig_shape))).reshape(sig_shape).astype(bool)
    return free, pinned, sigma, int(blob['band_rows']), names

def analyze(out_stem='s11_1_interface'):
    """No torch. Do pinned cells actually stack, and are they still valid?"""
    from geom import validity
    from geom.cell import UnitCell

    free, pinned, sigma, band_rows, names = load_samples(out_stem)
    rng = np.random.default_rng(0)

    def arm_report(pool, label, cross_target):
        """Stack pairs and count how many are genuinely joined."""
        joined, connected, valid_both, crossings, rows = 0, 0, 0, [], []
        usable, n_pairs = 0, 0
        for i, a_name in enumerate(names):
            b_name = names[(i + 1) % len(names)] if cross_target else a_name
            A, B = pool[a_name], pool[b_name]
            for k in range(len(A)):
                a = A[k]
                b = B[(k + 7) % len(B)]        # A different cell, not the same index
                rep = interface_report([a, b])
                n_pairs += 1
                joined += rep['joined']
                connected += rep['connected']
                crossings.append(rep['min_crossings'])
                both_ok = (validity.check(UnitCell(a)).ok
                           and validity.check(UnitCell(b)).ok)
                valid_both += both_ok
                usable += bool(both_ok and rep['joined'])
        crossings = np.array(crossings)
        out = {
            'arm': label, 'cross_target': cross_target, 'n_pairs': n_pairs,
            'usable_rate': usable / n_pairs,
            'joined_rate': joined / n_pairs,
            'connected_rate': connected / n_pairs,
            'both_cells_valid_rate': valid_both / n_pairs,
            'min_crossings_mean': float(crossings.mean()),
            'min_crossings_zero_rate': float((crossings == 0).mean()),
        }
        print(f"  {label:22} USABLE {100*out['usable_rate']:5.1f}%   "
              f"joined {100*out['joined_rate']:5.1f}%   "
              f"both valid {100*out['both_cells_valid_rate']:5.1f}%   "
              f"crossings {out['min_crossings_mean']:5.1f}")
        return out

    print(f'interface band {band_rows} rows, sigma {int(sigma.sum())} px metal')
    print(f'{len(names)} targets: {", ".join(names)}\n')
    arms = [
        arm_report(free, 'free, same target', False),
        arm_report(free, 'free, cross target', True),
        arm_report(pinned, 'pinned, same target', False),
        arm_report(pinned, 'pinned, cross target', True),
    ]

    # Does pinning cost validity? That is the price of the constraint.
    keep = {}
    for label, pool in (('free', free), ('pinned', pinned)):
        ok = sum(validity.check(UnitCell(a)).ok for n in names for a in pool[n])
        tot = sum(len(pool[n]) for n in names)
        keep[label] = ok / tot
        print(f'\n  {label:7} single-cell validity {100*ok/tot:.1f}%  ({ok}/{tot})')

    report = {
        'band_rows': band_rows,
        'sigma_metal_px': int(sigma.sum()),
        'targets': names,
        'arms': arms,
        'single_cell_validity': keep,
    }
    (RESULTS_DIR / f'{out_stem}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cross_pinned = [a for a in arms if a['arm'] == 'pinned, cross target'][0]
    cross_free = [a for a in arms if a['arm'] == 'free, cross target'][0]
    print("\nVerdict (cross-target, the case a graded stent needs):")
    print(f"  usable {100*cross_free['usable_rate']:.1f}% free "
          f"-> {100*cross_pinned['usable_rate']:.1f}% pinned")
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Axial interface compatibility')
    ap.add_argument('phase', choices=('sample', 'analyze'))
    ap.add_argument('--per-target', type=int, default=N_PER_TARGET)
    ap.add_argument('--band', type=int, default=BAND_ROWS)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--resample', type=int, default=RESAMPLE)
    args = ap.parse_args()
    if args.phase == 'sample':
        sample_phase(n_per_target=args.per_target, band_rows=args.band, steps=args.steps,
                     resample=args.resample)
    else:
        analyze()
