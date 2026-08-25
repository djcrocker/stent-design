"""
Does the valid-generation rate actually improve during training?

Two phases:
    python figures/scripts/s8_3_valid_rate.py sample
    python figures/scripts/s8_3_valid_rate.py plot
"""

import argparse
import json
import re

import numpy as np

import config

RESULTS = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_3_valid_rate.npz'
REPORT = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_3_valid_rate.json'
OUT = config.FIG_DEV_DIR / 's8_3_valid_rate.png'
CKPT_DIR = config.PROJECT_ROOT / 'diffusion' / 'checkpoints'

RAW = '#2a78d6'
CLEANED = '#2e9e5b'
TARGET = '#e0a83a'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def checkpoints():
    """Epoch checkpoints, in order, as (epoch, path)."""
    found = []
    for p in sorted(CKPT_DIR.glob('dataset_ddpm_epoch*.pt')):
        m = re.search(r'epoch(\d+)', p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)

def sample_phase(n=256, batch=128, steps=50, seed=0):
    """Torch only. Unconditional samples from every epoch checkpoint."""
    from diffusion.sample import load_model, sample_unconditional

    blobs = {}
    for epoch, path in checkpoints():
        ddpm, norm, cfg = load_model(path)
        fields = sample_unconditional(ddpm, norm, n=n, batch=batch, seed=seed,
                                      steps=steps, progress=False)
        blobs[f'fields_{epoch}'] = np.packbits(fields, axis=None)
        blobs[f'shape_{epoch}'] = np.array(fields.shape)
        print(f'  epoch {epoch:3d}: {len(fields)} sampled', flush=True)
    blobs['epochs'] = np.array([e for e, _ in checkpoints()])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS, **blobs)
    print(f'  Wrote {RESULTS.name}', flush=True)

def plot_phase():
    """No torch. Validity per epoch, plotted against the S8.3 target band."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from geom import cleanup, validity
    from geom.cell import UnitCell

    blob = np.load(RESULTS)
    epochs = blob['epochs']
    rows = []
    for epoch in epochs:
        shape = tuple(blob[f'shape_{epoch}'])
        fields = np.unpackbits(blob[f'fields_{epoch}'], count=int(np.prod(shape))).reshape(shape).astype(bool)
        raw = sum(validity.check(UnitCell(a)).ok for a in fields)
        fixed = sum(cleanup.clean(a).fixed for a in fields)
        rows.append({'epoch': int(epoch), 'n': len(fields),
                     'raw_rate': raw / len(fields), 'cleaned_rate': fixed / len(fields),
                     'f_metal_mean': float(fields.mean())})
        print(f"  epoch {epoch:3d}: raw {100 * rows[-1]['raw_rate']:5.1f}%  "
              f"cleaned {100 * rows[-1]['cleaned_rate']:5.1f}%", flush=True)

    REPORT.write_text(json.dumps(rows, indent=2), encoding='utf-8')

    e = np.array([r['epoch'] for r in rows])
    raw = 100 * np.array([r['raw_rate'] for r in rows])
    cln = 100 * np.array([r['cleaned_rate'] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.axhspan(70, 80, color=TARGET, alpha=0.18, zorder=1)
    ax.text(e[0], 75, ' S8.3 target band', va='center', fontsize=9, color='#8a6520')
    ax.plot(e, cln, color=CLEANED, lw=2.2, marker='s', ms=6, zorder=3,
            label='valid after S2.4 cleanup')
    ax.plot(e, raw, color=RAW, lw=2.2, marker='o', ms=6, zorder=3,
            label='valid raw, straight from the model')
    for x, y in ((e[-1], raw[-1]), (e[-1], cln[-1])):
        ax.annotate(f'{y:.1f}%', (x, y), xytext=(8, -3), textcoords='offset points',
                    fontsize=9, color=TEXT_PRIMARY)

    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel('training epoch', color=TEXT_SECONDARY)
    ax.set_ylabel('valid-generation rate (%)', color=TEXT_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, e[-1] + 4)
    ax.set_title(f"Valid-generation rate over training  ({rows[0]['n']} unconditional "
                 f"samples per checkpoint)",
                 color=TEXT_PRIMARY, fontsize=12.5, loc='left')
    ax.legend(fontsize=9.5, frameon=False, loc='lower right')

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')
    print('Wrote', OUT)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Valid rate over training')
    ap.add_argument('phase', choices=('sample', 'plot'))
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--steps', type=int, default=50)
    args = ap.parse_args()
    if args.phase == 'sample':
        sample_phase(n=args.n, steps=args.steps)
    else:
        plot_phase()
