"""
Are the designs at one target actually different, or the same design reshuffled?

This is the main argument for using a generative method at all. An optimizer returns one
answer; the claim here is that the inverse problem is one-to-many and a diffusion model
surfaces several distinct topologies that meet the same performance target.

Distinctness has to be translation-invariant. The cell lives on a torus, so a shifted copy is
the same design, and pixel-wise Hamming distance would score it as maximally different,
turning a trivial re-indexing into apparent novelty. The descriptor is therefore the
magnitude of the low-frequency 2D FFT, which translation leaves unchanged.

A raw distance means nothing on its own, so it is calibrated against two references measured
on the same designs:
  Shift floor  - A design against its own shifted copies.
  Cross-target - Designs drawn from different targets. If within-target distances approach
                 this, then designs meeting the same target are about as varied as designs
                 meeting different ones, which is the strong form of the claim.
"""

import argparse
import json

import numpy as np

import config
from diffusion import dataset

RESULTS_DIR = config.PROJECT_ROOT / 'screen' / 'results'
MODES = 10

def descriptor(fields, modes=MODES):
    """
    Translation-invariant structural descriptor, one row per cell.

    |FFT| is unchanged by translation on a torus. The DC term is dropped because it is
    `f_metal`, and each row is L2-normalized so the descriptor compares structure rather
    than density.
    """
    arrs = np.asarray(fields, dtype=np.float32)
    spec = np.abs(np.fft.rfft2(arrs, axes=(1, 2)))[:, :modes, :modes]
    flat = spec.reshape(len(arrs), -1)
    flat = flat[:, 1:]                                   # Drop DC == f_metal
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norm, 1e-12)

def pairwise(desc):
    """Euclidean distances between normalized descriptors. 0 means indistinguishable."""
    d = np.sqrt(np.maximum(
        ((desc[:, None, :] - desc[None, :, :]) ** 2).sum(axis=-1), 0.0))
    return d

def shift_floor(fields, n_shifts=8, seed=0, modes=MODES):
    """Distance between a design and its own torus shifts."""
    rng = np.random.default_rng(seed)
    arrs = np.asarray(fields, bool)
    out = []
    for arr in arrs:
        shifted = [np.roll(np.roll(arr, int(rng.integers(1, arr.shape[0])), axis=0), int(rng.integers(1, arr.shape[1])), axis=1)
                   for _ in range(n_shifts)]
        d = descriptor(np.stack([arr] + shifted), modes=modes)
        out.extend(pairwise(d)[0, 1:].tolist())
    return float(np.mean(out)), float(np.max(out))

def farthest_first(desc, k, seed=0):
    """
    Pick k mutually distant designs, greedily.

    A gallery of the k highest-scoring designs would show near-copies if the model happened
    to sample one region densely; farthest-first picks exemplars that span the variety
    actually present.
    """
    k = min(k, len(desc))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(desc)))]
    d = np.linalg.norm(desc - desc[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(desc - desc[nxt], axis=1))
    return np.array(chosen)

def close_to_target(rows, pool, target, keep_fraction=0.5, min_keep=6, keys=('K_radial', 'A_over_lim')):
    """
    The half of a target's designs that land closest to what was asked.

    Distance is per-component relative error, so `K_radial` (which spans decades) and
    `A_over_lim` (a fraction) contribute comparably instead of the larger number dominating.
    Returns positions within `pool`, not pool indices.
    """
    err = np.zeros(len(pool))
    for key in keys:
        want = float(target[key])
        got = np.array([rows[i][key] for i in pool], float)
        err += np.abs(got - want) / max(abs(want), 1e-9)
    keep = max(min_keep, int(round(keep_fraction * len(pool))))
    return np.argsort(err, kind='mergesort')[:min(keep, len(pool))]


def analyze(source_stem='s9_1_generated', min_valid=8, n_exemplars=6,
            out_stem='s9_3_onetomany'):
    """Per-target distinctness, calibrated against the shift floor and cross-target spread."""
    from diffusion import generate

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields, which, meta = generate.load_samples(source_stem)
    rows, _ = generate.cached_labels(fields, source_stem)
    targets = meta['targets']

    valid = np.array([i for i, r in enumerate(rows) if r is not None])
    floor_mean, floor_max = shift_floor(fields[valid[:40]])

    # Cross-target reference: one design from each target, pairwise.
    reps = []
    for j in range(len(targets)):
        pool = [i for i in valid if which[i] == j]
        if pool:
            reps.append(pool[0])
    cross = pairwise(descriptor(fields[np.array(reps)]))
    cross_mean = float(cross[np.triu_indices(len(reps), 1)].mean()) if len(reps) > 1 else 0.0

    per_target = []
    for j, t in enumerate(targets):
        pool = np.array([i for i in valid if which[i] == j])
        if len(pool) < min_valid:
            per_target.append({'name': t['name'], 'n_valid': int(len(pool)),
                               'note': 'too few valid designs'})
            continue
        desc = descriptor(fields[pool])
        d = pairwise(desc)
        iu = np.triu_indices(len(pool), 1)
        within = d[iu]
        ys = {key: np.array([rows[i][key] for i in pool], float)
              for key in dataset.Y_KEYS}
        on_target = pool[close_to_target(rows, pool, t['target'])]
        near_desc = descriptor(fields[on_target])
        exemplars = on_target[farthest_first(near_desc, n_exemplars)]
        per_target.append({
            'name': t['name'],
            'n_valid': int(len(pool)),
            'within_mean': float(within.mean()),
            'within_p10': float(np.percentile(within, 10)),
            'within_max': float(within.max()),
            'fraction_above_floor': float((within > floor_max).mean()),
            'ratio_to_cross_target': float(within.mean() / cross_mean) if cross_mean else None,
            'y_spread': {key: [float(np.percentile(v, 10)), float(np.percentile(v, 90))]
                         for key, v in ys.items()},
            'exemplar_pool_indices': exemplars.tolist(),
        })

    result = {
        'source': source_stem,
        'shift_floor_mean': floor_mean,
        'shift_floor_max': floor_max,
        'cross_target_mean': cross_mean,
        'modes': MODES,
        'per_target': per_target,
    }
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='One-to-many check')
    ap.add_argument('--source', default='s9_1_generated')
    ap.add_argument('--exemplars', type=int, default=6)
    ap.add_argument('--stem', default='s9_3_onetomany')
    args = ap.parse_args()
    out = analyze(source_stem=args.source, n_exemplars=args.exemplars, out_stem=args.stem)

    print(f"shift floor (a design vs its own shifts): mean {out['shift_floor_mean']:.4f}, "
          f"max {out['shift_floor_max']:.4f}")
    print(f"cross-target mean distance: {out['cross_target_mean']:.4f}")
    print()
    print(f"{'target':14} {'n':>4} {'within':>8} {'p10':>7} {'max':>7} "
          f"{'>floor':>7} {'vs cross':>9}")
    for t in out['per_target']:
        if 'note' in t:
            print(f"  {t['name']:12} {t['n_valid']:4d}   {t['note']}")
            continue
        print(f"  {t['name']:12} {t['n_valid']:4d} {t['within_mean']:8.4f} "
              f"{t['within_p10']:7.4f} {t['within_max']:7.4f} "
              f"{100 * t['fraction_above_floor']:6.1f}% {t['ratio_to_cross_target']:9.2f}")
