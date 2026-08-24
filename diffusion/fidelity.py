"""
Does the conditioning vector actually steer the output?

We know the model emits valid topology. Nothing there showed `y` does any work: an
unconditional model would have scored the same. This step samples at explicit targets `y*`,
re-labels the results with the 2D screen, and asks whether achieved `y` tracks `y*`.

Measured per component, never pooled. `f_metal` is the array mean, so it is easily
readable from the field and a model can nail it while ignoring everything else; a pooled
score would be lifted by the easy component and hide whether `K_radial` or `eps_a_max`
are steered at all.

Two statistics, because they fail differently:
  Tracking - Spearman/Pearson of achieved vs target across targets. Does the ordering follow?
  Landing  - Median |achieved - target| normalized by the train standard deviation. A model
             can track a target perfectly while sitting at a constant offset from it.

Targets come from the test split rather than a grid over objective space. A grid would
contain physically impossible combinations and a miss there would be ambiguous between weak 
conditioning and an impossible ask. Test-split targets are real `y` vectors the model never 
trained on. Out-of-range asks are probed separately and reported as extrapolation.
"""

import argparse
import json
import time

import numpy as np

import config
from diffusion import dataset

Y_KEYS = dataset.Y_KEYS
DEFAULT_GUIDANCE = (1.0, 2.0, 3.0, 5.0)
RESULTS_DIR = config.PROJECT_ROOT / 'diffusion' / 'results'

def _rankdata(values):
    """Average ranks, ties shared."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_vals = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks

def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def _spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))

def pick_targets(frame, n_targets=40, seed=0, split='test'):
    """Real `y` vectors from a split the model never trained on."""
    sub = frame[frame['split'] == split] if 'split' in frame.columns else frame
    labels = sub[list(Y_KEYS)].to_dict('records')
    keep = dataset.coverage_subsample(labels, min(n_targets, len(sub)), n_bins=4, seed=seed)
    picked = sub.iloc[keep]
    return [{k: float(row[k]) for k in Y_KEYS} for _, row in picked.iterrows()]

def out_of_range_targets(frame):
    """
    Deliberately impossible asks.

    Each pushes one component past what the training set contains while holding the rest at
    the training median, so a failure is because of a component.
    """
    med = {k: float(frame[k].median()) for k in Y_KEYS}
    spec = (('K_radial', 'K_radial 2x the max', float(frame['K_radial'].max()) * 2.0),
            ('eps_a_max', 'eps_a_max 0.4x the min', float(frame['eps_a_max'].min()) * 0.4),
            ('A_over_lim', 'A_over_lim = 0', 0.0))
    probes = []
    for key, label, value in spec:
        target = dict(med)
        target[key] = value
        probes.append({'label': label, 'component': key, 'target': target})
    return probes

def label_fields(fields, clean_first=True):
    """Re-label generated cells with the 2D screen."""
    from geom import cleanup
    from geom.cell import UnitCell

    rows, dropped = [], 0
    for arr in fields:
        cell = UnitCell(arr)
        if clean_first:
            result = cleanup.clean(arr)
            if not result.fixed or result.cell is None:
                rows.append(None)
                dropped += 1
                continue
            cell = result.cell
        rows.append(dataset.label_one(cell.to_array()))
    return rows, dropped

def fidelity_stats(targets, achieved, train_std):
    """Per-component tracking and landing error. `achieved` may hold None for drops."""
    out = {}
    for i, key in enumerate(Y_KEYS):
        pairs = [(t[key], a[key]) for t, a in zip(targets, achieved) if a is not None]
        if len(pairs) < 4:
            out[key] = {'n': len(pairs), 'note': 'too few labeled samples'}
            continue
        want = np.array([p[0] for p in pairs], float)
        got = np.array([p[1] for p in pairs], float)
        # K_radial spans decades; compare in the space the model was conditioned in.
        if key in dataset.LOG_KEYS:
            want, got = (np.log10(np.maximum(want, 1e-12)),
                         np.log10(np.maximum(got, 1e-12)))
        out[key] = {
            'n': len(pairs),
            'spearman': _spearman(want, got),
            'pearson': _pearson(want, got),
            'median_abs_err_sd': float(np.median(np.abs(got - want)) / train_std[i]),
            'bias_sd': float(np.median(got - want) / train_std[i]),
        }
    return out

def diversity(fields, which, n_targets):
    """Mean pairwise Hamming distance within each target group."""
    flat = fields.reshape(len(fields), -1)
    scores = []
    for j in range(n_targets):
        idx = np.flatnonzero(which == j)
        if len(idx) < 2:
            continue
        block = flat[idx].astype(np.float64)
        # Unbiased mean pairwise disagreement, without materializing every pair.
        p = block.mean(axis=0)
        scores.append(float(2.0 * np.mean(p * (1.0 - p)) * len(idx) / (len(idx) - 1)))
    return float(np.mean(scores)) if scores else float('nan')

def sample_phase(n_targets=40, n_per_target=16, guidance=DEFAULT_GUIDANCE, steps=50,
                 seed=0, ckpt=None, out_stem='s8_4_fidelity'):
    """Torch only. Sample every target at every guidance, save raw fields."""
    from diffusion.sample import load_model, sample_conditional

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ddpm, norm, cfg = load_model(ckpt)
    _, frame = dataset.load()

    targets = pick_targets(frame, n_targets=n_targets, seed=seed)
    probes = out_of_range_targets(frame)
    print(f'{len(targets)} targets from the test split, {n_per_target} samples each',
          flush=True)

    blobs = {}
    for g in guidance:
        t0 = time.time()
        fields, which = sample_conditional(ddpm, norm, targets, n_per_target=n_per_target,
                                           guidance=g, steps=steps, seed=seed, progress=False)
        blobs[f'fields_{g}'] = np.packbits(fields, axis=None)
        blobs[f'shape_{g}'] = np.array(fields.shape)
        blobs[f'which_{g}'] = which
        print(f'  guidance {g}: {len(fields)} sampled in '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)

    best_g = max(guidance)
    ef, ew = sample_conditional(ddpm, norm, [p['target'] for p in probes],
                                n_per_target=n_per_target, guidance=best_g, steps=steps,
                                seed=seed, progress=False)
    blobs['extrap_fields'] = np.packbits(ef, axis=None)
    blobs['extrap_shape'] = np.array(ef.shape)
    blobs['extrap_which'] = ew

    np.savez_compressed(RESULTS_DIR / f'{out_stem}_samples.npz', **blobs)
    (RESULTS_DIR / f'{out_stem}_targets.json').write_text(
        json.dumps({'targets': targets, 'probes': probes, 'guidance': list(guidance),
                    'n_per_target': n_per_target, 'steps': steps,
                    'extrap_guidance': best_g,
                    'normalizer_std': norm.std.tolist()}, indent=2), encoding='utf-8')
    print(f'  Wrote {out_stem}_samples.npz', flush=True)

def _unpack(blob, fields_key, shape_key):
    shape = tuple(blob[shape_key])
    return np.unpackbits(blob[fields_key], count=int(np.prod(shape))).reshape(shape).astype(bool)

def analyze_phase(out_stem='s8_4_fidelity'):
    """No torch. Label the saved fields and compute per-component fidelity."""
    blob = np.load(RESULTS_DIR / f'{out_stem}_samples.npz')
    meta = json.loads((RESULTS_DIR / f'{out_stem}_targets.json').read_text(encoding='utf-8'))
    _, frame = dataset.load()
    targets = meta['targets']
    train_std = np.array(meta['normalizer_std'], float)

    result = {'n_targets': len(targets), 'n_per_target': meta['n_per_target'],
              'steps': meta['steps'], 'guidance': meta['guidance'], 'by_guidance': {}}

    for g in meta['guidance']:
        fields = _unpack(blob, f'fields_{g}', f'shape_{g}')
        which = blob[f'which_{g}']
        rows, dropped = label_fields(fields)
        entry = {
            'labeled': int(sum(r is not None for r in rows)),
            'dropped_unfixable': int(dropped),
            'valid_rate_after_cleanup': 1.0 - dropped / max(len(fields), 1),
            'diversity_hamming': diversity(fields, which, len(targets)),
            'components': fidelity_stats([targets[j] for j in which], rows, train_std),
        }
        result['by_guidance'][str(g)] = entry
        print(f'  guidance {g}: {entry["labeled"]}/{len(fields)} labeled, '
              f'valid {100 * entry["valid_rate_after_cleanup"]:.1f}%, '
              f'diversity {entry["diversity_hamming"]:.4f}', flush=True)
        for key in Y_KEYS:
            c = entry['components'][key]
            if 'spearman' in c:
                print(f'      {key:11} rho {c["spearman"]:+.3f}  '
                      f'err {c["median_abs_err_sd"]:.3f} sd  '
                      f'bias {c["bias_sd"]:+.3f} sd')

    # EXTRAPOLATION #
    ef = _unpack(blob, 'extrap_fields', 'extrap_shape')
    ew = blob['extrap_which']
    rows, _ = label_fields(ef)
    extrap = []
    for j, probe in enumerate(meta['probes']):
        key = probe['component']
        got = [rows[i] for i in np.flatnonzero(ew == j) if rows[i] is not None]
        extrap.append({
            'label': probe['label'], 'component': key,
            'target': probe['target'][key],
            'achieved_median': float(np.median([r[key] for r in got])) if got else None,
            'training_range': [float(frame[key].min()), float(frame[key].max())],
            'n_labeled': len(got),
        })
    result['extrapolation'] = {'guidance': meta['extrap_guidance'], 'probes': extrap}

    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Conditioning fidelity')
    ap.add_argument('phase', choices=('sample', 'analyze'),
                    help='sample needs torch; analyze must run in its own process')
    ap.add_argument('--targets', type=int, default=40)
    ap.add_argument('--per-target', type=int, default=16)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--stem', default='s8_4_fidelity')
    args = ap.parse_args()

    if args.phase == 'sample':
        sample_phase(n_targets=args.targets, n_per_target=args.per_target,
                     steps=args.steps, seed=args.seed, ckpt=args.ckpt, out_stem=args.stem)
    else:
        out = analyze_phase(out_stem=args.stem)
        print()
        print('Extrapolation probes:')
        for p in out['extrapolation']['probes']:
            lo, hi = p['training_range']
            got = ('none labeled' if p['achieved_median'] is None
                   else f"{p['achieved_median']:.4f}")
            print(f"  {p['label']:24} target {p['target']:.4f}  achieved {got}  "
                  f"(training {lo:.4f}-{hi:.4f})")
