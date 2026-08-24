"""
Sample at chosen targets, and check the targets are askable before asking.

Support is counted in a neighborhood of the target rather than a single point, because a
target is only askable if the model saw designs near it.
"""

import argparse
import json
import time

import numpy as np

import config
from diffusion import dataset

Y_KEYS = dataset.Y_KEYS
RESULTS_DIR = config.PROJECT_ROOT / 'diffusion' / 'results'

# From S8.4: best fidelity, and diversity is flat across guidance so one-to-many pays nothing.
GUIDANCE = 5.0
N_PER_TARGET = 64

# The desirable scope is high radial support, low area over the fatigue limit.
# eps_a_max and f_metal are filled from the conditional median of the supporting cells.
TARGET_LADDER = (
    {'name': 'K100_A25', 'K_radial': 100.0, 'A_over_lim': 0.25},
    {'name': 'K200_A25', 'K_radial': 200.0, 'A_over_lim': 0.25},
    {'name': 'K300_A25', 'K_radial': 300.0, 'A_over_lim': 0.25},
    {'name': 'K100_A10', 'K_radial': 100.0, 'A_over_lim': 0.10},
    {'name': 'K200_A10', 'K_radial': 200.0, 'A_over_lim': 0.10},
    {'name': 'K300_A10', 'K_radial': 300.0, 'A_over_lim': 0.10},
    # Controls: mid-distribution, where support is thick and conditioning should be easy.
    {'name': 'control_mid', 'K_radial': 58.0, 'A_over_lim': 0.54},
    {'name': 'control_soft', 'K_radial': 20.0, 'A_over_lim': 0.70},
)

def support(frame, target, rel=0.25, abs_tol=0.08):
    """
    How many training cells sit near this target.

    A relative window on `K_radial` (it spans decades) and an absolute one on the fraction
    metrics.
    """
    mask = np.ones(len(frame), dtype=bool)
    for key, value in target.items():
        if key not in Y_KEYS:
            continue
        col = frame[key].to_numpy()
        if key in dataset.LOG_KEYS:
            mask &= np.abs(np.log10(np.maximum(col, 1e-12)) - np.log10(max(value, 1e-12))) \
                <= np.log10(1 + rel)
        else:
            mask &= np.abs(col - value) <= abs_tol
    return int(mask.sum()), mask

def complete_target(frame, target, rel=0.25, abs_tol=0.08):
    """
    Fill unspecified `y` components from the conditional median of the supporting cells.

    Pinning them to the global median would ask for a combination the data doesn't contain
    and the model would then be asked to satisfy a vector no design realizes.
    """
    n, mask = support(frame, target, rel, abs_tol)
    source = frame[mask] if n >= 8 else frame
    full = {}
    for key in Y_KEYS:
        full[key] = float(target[key]) if key in target else float(source[key].median())
    return full, n

def build_targets(frame, ladder=TARGET_LADDER, min_support=8):
    """Complete every target and record its support. Under-supported ones are flagged."""
    out = []
    for spec in ladder:
        asked = {k: v for k, v in spec.items() if k in Y_KEYS}
        full, n = complete_target(frame, asked)
        out.append({
            'name': spec['name'],
            'asked': asked,
            'target': full,
            'support': n,
            'well_supported': bool(n >= min_support),
        })
    return out

def sample_phase(guidance=GUIDANCE, n_per_target=N_PER_TARGET, steps=50, seed=0, ckpt=None, out_stem='s9_1_generated'):
    """Torch only. Sample every target and save the raw fields."""
    from diffusion.sample import load_model, sample_conditional

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ddpm, norm, cfg = load_model(ckpt)
    _, frame = dataset.load()

    targets = build_targets(frame)
    print(f'{len(targets)} targets, {n_per_target} samples each at guidance {guidance}',
          flush=True)
    for t in targets:
        flag = '' if t['well_supported'] else '   <-- THIN SUPPORT'
        print(f"  {t['name']:14} support {t['support']:5d}{flag}")

    t0 = time.time()
    fields, which = sample_conditional(ddpm, norm, [t['target'] for t in targets],
                                       n_per_target=n_per_target, guidance=guidance,
                                       steps=steps, seed=seed, progress=False)
    print(f'  sampled {len(fields)} in {(time.time() - t0) / 60:.1f} min', flush=True)

    np.savez_compressed(RESULTS_DIR / f'{out_stem}_samples.npz',
                        fields=np.packbits(fields, axis=None),
                        shape=np.array(fields.shape), which=which)
    (RESULTS_DIR / f'{out_stem}_targets.json').write_text(
        json.dumps({'targets': targets, 'guidance': guidance,
                    'n_per_target': n_per_target, 'steps': steps,
                    'normalizer_std': norm.std.tolist()}, indent=2), encoding='utf-8')
    print(f'  Wrote {out_stem}_samples.npz', flush=True)

def load_samples(out_stem='s9_1_generated'):
    blob = np.load(RESULTS_DIR / f'{out_stem}_samples.npz')
    shape = tuple(blob['shape'])
    fields = np.unpackbits(blob['fields'],
                           count=int(np.prod(shape))).reshape(shape).astype(bool)
    meta = json.loads((RESULTS_DIR / f'{out_stem}_targets.json').read_text(encoding='utf-8'))
    return fields, blob['which'], meta

def screen_phase(out_stem='s9_1_generated'):
    """No torch. Clean, validate, label, and record achieved vs asked per target."""
    from diffusion.fidelity import label_fields

    fields, which, meta = load_samples(out_stem)
    targets = meta['targets']
    rows, dropped = label_fields(fields)

    per_target = []
    for j, t in enumerate(targets):
        idx = np.flatnonzero(which == j)
        got = [rows[i] for i in idx if rows[i] is not None]
        entry = {
            'name': t['name'], 'asked': t['asked'], 'target': t['target'],
            'support': t['support'], 'well_supported': t['well_supported'],
            'n_sampled': int(len(idx)), 'n_valid': len(got),
            'valid_rate': len(got) / max(len(idx), 1),
        }
        for key in Y_KEYS:
            if got:
                vals = np.array([g[key] for g in got], float)
                entry[f'{key}_achieved_median'] = float(np.median(vals))
                entry[f'{key}_achieved_p10'] = float(np.percentile(vals, 10))
                entry[f'{key}_achieved_p90'] = float(np.percentile(vals, 90))
            else:
                entry[f'{key}_achieved_median'] = None
        per_target.append(entry)

    result = {'guidance': meta['guidance'], 'n_per_target': meta['n_per_target'],
              'steps': meta['steps'], 'total_sampled': int(len(fields)),
              'total_valid': int(len(fields) - dropped),
              'overall_valid_rate': 1.0 - dropped / max(len(fields), 1),
              'per_target': per_target}
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Generate at chosen targets')
    ap.add_argument('phase', choices=('sample', 'screen'),
                    help='sample needs torch; screen must run in a separate process')
    ap.add_argument('--per-target', type=int, default=N_PER_TARGET)
    ap.add_argument('--guidance', type=float, default=GUIDANCE)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--stem', default='s9_1_generated')
    args = ap.parse_args()

    if args.phase == 'sample':
        sample_phase(guidance=args.guidance, n_per_target=args.per_target,
                     steps=args.steps, seed=args.seed, ckpt=args.ckpt,
                     out_stem=args.stem)
    else:
        out = screen_phase(out_stem=args.stem)
        print(f"{out['total_valid']}/{out['total_sampled']} valid after cleanup "
              f"({100 * out['overall_valid_rate']:.1f}%)")
        print()
        print(f"{'target':14} {'supp':>6} {'valid':>7}   "
              f"{'K_radial asked/got':>26}   {'A_over_lim asked/got':>24}")
        for t in out['per_target']:
            ka = t['asked'].get('K_radial')
            aa = t['asked'].get('A_over_lim')
            kg = t['K_radial_achieved_median']
            ag = t['A_over_lim_achieved_median']
            flag = '' if t['well_supported'] else '  THIN'
            print(f"  {t['name']:12} {t['support']:6d} {t['n_valid']:4d}/{t['n_sampled']:<3d}"
                  f"   {ka:9.1f} -> {kg:9.1f}        {aa:7.3f} -> {ag:7.3f}{flag}")
