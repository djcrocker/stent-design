"""
Train/val/test splits.

Dataset has 4,348 exact duplicate cells (8.7 %). Mostly from the family source, whose 
randomly drawn widths and amplitudes snap to the same pixel grid and rasterize to
identical arrays. A random row-wise split would put identical designs in both train and
validation, and the val loss would then be measuring memorization.

So cells are grouped first and the groups are split. Grouping uses a translation-invariant
signature: the cell lives on a torus, so a shifted copy is the same design, and |FFT| is
unchanged by translation. Quantizing a low-frequency corner of the spectrum gives a cheap
bucket key that catches exact duplicates and shifted copies together, without the 4,096
shift comparisons per pair that a direct canonical form would need.

Measured here, translation adds little over exact matching (4,456 grouped vs 4,348 exact),
but the invariance costs nothing and the guarantee is worth stating.
"""

import json
import pathlib

import numpy as np

import config
from diffusion import dataset

SPLIT_NAMES = ('train', 'val', 'test')
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
# Low-frequency corner of the spectrum, quantized. Coarse enough to group near-identical
# designs, fine enough not to collapse different ones.
SIG_MODES = 6
SIG_QUANT = 0.002

def signature(arrs, modes=SIG_MODES, quant=SIG_QUANT):
    """Translation-invariant signature per cell. Returns an (N, modes*modes) int array."""
    arrs = np.asarray(arrs)
    spec = np.abs(np.fft.rfft2(arrs.astype(np.float32), axes=(1, 2)))
    sig = spec[:, :modes, :modes].reshape(len(arrs), -1)
    return np.round(sig / (arrs[0].size * quant)).astype(np.int32)

def group_ids(arrs, modes=SIG_MODES, quant=SIG_QUANT):
    """Assign each cell a design-group id. Identical and shifted cells share an id."""
    sig = signature(arrs, modes, quant)
    seen, out = {}, np.empty(len(sig), dtype=np.int64)
    for i, row in enumerate(sig):
        key = row.tobytes()
        if key not in seen:
            seen[key] = len(seen)
        out[i] = seen[key]
    return out

def make_splits(arrs, fractions=DEFAULT_FRACTIONS, seed=0, modes=SIG_MODES, quant=SIG_QUANT):
    """
    Split whole design-groups into train/val/test.

    Groups are shuffled and filled greedily toward the target fractions rather than sliced by count: 
    group sizes are uneven, so assigning by group count would miss the intended row proportions.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f'fractions must sum to 1, got {sum(fractions)}')
    groups = group_ids(arrs, modes, quant)
    order = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    members = {g: np.flatnonzero(groups == g) for g in order}
    targets = [f * len(groups) for f in fractions]
    counts = [0, 0, 0]
    assignment = np.empty(len(groups), dtype=object)

    for g in order:
        # Whichever split is furthest below its target takes the next group whole.
        deficit = [t - c for t, c in zip(targets, counts)]
        k = int(np.argmax(deficit))
        idx = members[g]
        assignment[idx] = SPLIT_NAMES[k]
        counts[k] += len(idx)
    return assignment.astype(str), groups

def coverage_by_split(frame, split, n_bins=8):
    """Occupied `y`-bins and per-component range, per split."""
    out = {}
    for name in SPLIT_NAMES:
        mask = (split == name)
        sub = frame.loc[mask]
        labels = sub[list(dataset.Y_KEYS)].to_dict('records')
        occupied = len({tuple(r) for r in dataset.bin_indices(labels, n_bins=n_bins)})
        out[name] = {
            'n': int(mask.sum()),
            'fraction': float(mask.mean()),
            'occupied_bins': occupied,
            'ranges': {k: [float(sub[k].min()), float(sub[k].max())]
                       for k in dataset.Y_KEYS},
        }
    return out

def leakage_check(split, groups):
    """No design-group can appear in more than one split."""
    bad = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(set(split[idx])) > 1:
            bad.append(int(g))
    return bad

def build(fractions=DEFAULT_FRACTIONS, seed=0, out_dir=None, stem='dataset'):
    """Assign splits, verify no leakage, and record coverage per split."""
    out_dir = config.DATA_DIR if out_dir is None else pathlib.Path(out_dir)
    arrs, frame = dataset.load(out_dir=out_dir, stem=stem)

    split, groups = make_splits(arrs, fractions=fractions, seed=seed)
    leaks = leakage_check(split, groups)
    if leaks:
        raise RuntimeError(f'{len(leaks)} design-groups straddle splits')

    frame = frame.copy()
    frame['split'] = split
    frame['design_group'] = groups
    frame.to_parquet(out_dir / f'{stem}.parquet', index=False)

    n_groups = int(len(np.unique(groups)))
    info = {
        'fractions_requested': list(fractions),
        'n_cells': int(len(frame)),
        'n_design_groups': n_groups,
        'n_duplicate_cells': int(len(frame) - n_groups),
        'leakage_groups': 0,
        'coverage': coverage_by_split(frame, split),
    }
    (out_dir / f'{stem}_splits.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description='Train/val/test splits by design')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stem', default='dataset')
    args = ap.parse_args()
    result = build(seed=args.seed, stem=args.stem)

    print(f"{result['n_cells']:,} cells in {result['n_design_groups']:,} design groups "
          f"({result['n_duplicate_cells']:,} duplicates collapsed)")
    print(f"leakage: {result['leakage_groups']} groups straddle splits")
    print()
    for name, d in result['coverage'].items():
        print(f"  {name:6} {d['n']:6,} cells ({100 * d['fraction']:4.1f}%)  "
              f"{d['occupied_bins']:4d} occupied y-bins")
    print()
    for key in dataset.Y_KEYS:
        spans = '   '.join(
            f"{name}: {d['ranges'][key][0]:.4f}-{d['ranges'][key][1]:.4f}"
            for name, d in result['coverage'].items())
        print(f"  {key:11} {spans}")
