"""
Sample from the trained DDPM, then measure how much of it is usable.

The valid-generation rate is the Spike C gate: the fraction of raw samples that pass
validity after cleanup. Reported at two points:
  Raw    - Straight from the model, thresholded at 0
  Cleaned- After the same repair pipeline every other cell in this project goes through
because a model that produces almost-valid topology which cleanup can fix is a usable model,
while one whose output cleanup has to rebuild is not, and a single number hides the
difference.

Unconditional first, because if the model can't produce valid topology at all then
conditioning fidelity is meaningless.
"""

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import torch

import config
from diffusion.ddpm import DDPM, Normalizer
from diffusion.model import UNet

CKPT_DIR = config.PROJECT_ROOT / 'diffusion' / 'checkpoints'
RESULTS_DIR = config.PROJECT_ROOT / 'diffusion' / 'results'

def load_model(path=None, device=None):
    """Restore the EMA weights and the y-normalizer saved alongside them."""
    path = (CKPT_DIR / 'dataset_ddpm.pt') if path is None else pathlib.Path(path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob['config']
    model = UNet(y_dim=cfg['y_dim'], base=cfg['base']).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    ddpm = DDPM(model, timesteps=cfg['timesteps'], device=device)
    return ddpm, Normalizer.from_dict(blob['normalizer']), cfg

def to_fields(x):
    """Model output in [-1, 1] to a binary field. Threshold at 0, the data midpoint."""
    return (x.detach().float().cpu().numpy()[:, 0] > 0.0)

def validity_report(fields):
    """Raw and post-cleanup validity, with the reasons raw samples fail."""
    from geom import cleanup, validity
    from geom.cell import UnitCell

    raw_ok = 0
    fixed = 0
    reasons = collections.Counter()
    cleaned, change = [], []
    for arr in fields:
        v = validity.check(UnitCell(arr))
        raw_ok += bool(v.ok)
        if not v.ok:
            for r in v.reasons:
                reasons[r] += 1
        result = cleanup.clean(arr)
        if result.fixed and result.cell is not None:
            fixed += 1
            cleaned.append(result.cell)
            change.append(result.change_fraction)
    n = len(fields)
    return {
        'n': n,
        'raw_valid': raw_ok,
        'raw_rate': raw_ok / max(n, 1),
        'cleaned_valid': fixed,
        'cleaned_rate': fixed / max(n, 1),
        'mean_change_fraction': float(np.mean(change)) if change else None,
        'raw_failure_reasons': dict(reasons.most_common()),
    }, cleaned

def sample_unconditional(ddpm, norm, n=256, batch=128, seed=0, steps=50, progress=True):
    """
    Sample with the conditioning switched off.

    guidance=0 uses only the unconditional branch, which is what the CFG dropout trained.
    DDIM by default: measured against the full 1000-step chain it gives the same raw
    validity (90.6 %) at 0.182 s/cell instead of 5.3, so the gate can be measured on a
    sample large enough to have a usable confidence interval.
    """
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        y = torch.zeros(k, len(norm.keys), device=ddpm.device)
        x = (ddpm.ddim_sample(k, y, guidance=0.0, steps=steps) if steps
             else ddpm.sample(k, y, guidance=0.0))
        out.append(to_fields(x))
        if progress:
            print(f'    sampled {i + k}/{n}', flush=True)
    return np.concatenate(out)

def sample_conditional(ddpm, norm, targets, n_per_target=32, guidance=2.0, seed=0, steps=50, progress=True):
    """Sample at explicit y* targets. `targets` is a list of {key: value}."""
    torch.manual_seed(seed)
    fields, which = [], []
    for j, target in enumerate(targets):
        z = torch.tensor(norm.transform_dict(target), dtype=torch.float32)
        y = z[None].repeat(n_per_target, 1).to(ddpm.device)
        x = (ddpm.ddim_sample(n_per_target, y, guidance=guidance, steps=steps) if steps
             else ddpm.sample(n_per_target, y, guidance=guidance))
        fields.append(to_fields(x))
        which.extend([j] * n_per_target)
        if progress:
            print(f'    target {j + 1}/{len(targets)} done', flush=True)
    return np.concatenate(fields), np.array(which)

def run(n=1024, batch=128, seed=0, steps=50, ckpt=None, out_stem='s8_2_unconditional'):
    """Unconditional samples, validity report, and the fields saved for inspection."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ddpm, norm, cfg = load_model(ckpt)
    print(f'model on {ddpm.device}, {cfg["timesteps"]} timesteps', flush=True)

    t0 = time.time()
    fields = sample_unconditional(ddpm, norm, n=n, batch=batch, seed=seed, steps=steps)
    elapsed = time.time() - t0
    print(f'  sampled {len(fields)} in {elapsed / 60:.1f} min '
          f'({elapsed / max(len(fields), 1):.1f} s/cell)', flush=True)

    report, cleaned = validity_report(fields)
    report['guidance'] = 0.0
    report['steps'] = steps
    report['seconds_per_cell'] = elapsed / max(len(fields), 1)
    report['f_metal_mean'] = float(fields.mean())

    np.savez_compressed(RESULTS_DIR / f'{out_stem}_fields.npz',
                        packed=np.packbits(fields, axis=None),
                        shape=np.array(fields.shape))
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='S8.2 unconditional sampling + validity')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=50, help='DDIM steps; 0 = full DDPM chain')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()
    result = run(n=args.n, batch=args.batch, seed=args.seed, steps=args.steps,
                 ckpt=args.ckpt)

    print()
    print(f"  raw valid      {result['raw_valid']:4d}/{result['n']} "
          f"= {100 * result['raw_rate']:.1f}%")
    print(f"  after cleanup  {result['cleaned_valid']:4d}/{result['n']} "
          f"= {100 * result['cleaned_rate']:.1f}%")
    if result['mean_change_fraction'] is not None:
        print(f"  cleanup changed {100 * result['mean_change_fraction']:.2f}% of pixels "
              f"on average")
    print(f"  mean f_metal   {result['f_metal_mean']:.4f}")
    if result['raw_failure_reasons']:
        print('  raw failures:')
        for reason, count in result['raw_failure_reasons'].items():
            print(f'    {reason:16} {count}')
