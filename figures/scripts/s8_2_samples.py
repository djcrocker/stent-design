"""
What the generated cells actually look like.

An 85% valid-generation rate is a number the model could in principle reach by finding an
unhonest way to satisfy connectivity and wrapping. Only looking at the samples can rule
that out.

Top     - Generated cells, bordered by outcome: valid raw, fixed by S2.4 cleanup, unfixable.
Middle  - The three failure modes side by side, to show whether failures are near-misses or
          nonsense.
Bottom  - Generated against real training cells at matched f_metal, unlabeled row order, plus
          the f_metal distributions. "Indistinguishable from training data" is the informal
          bar and this is where it either holds or does not.

Usage: python figures/scripts/s8_2_samples.py
"""

import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config

FIELDS = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_2_unconditional_fields.npz'
REPORT = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_2_unconditional.json'
OUT = config.FIG_DEV_DIR / 's8_2_samples.png'

OK = '#2e9e5b'
FIXED = '#e0a83a'
BAD = '#e34948'
REAL = '#2a78d6'
GEN = '#7a3fbf'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

N_COLS = 12

def load_fields():
    blob = np.load(FIELDS)
    return np.unpackbits(blob['packed'], count=int(np.prod(blob['shape']))).reshape(tuple(blob['shape'])).astype(bool)

def classify(fields):
    """Outcome per sample, plus the raw failure reasons."""
    from geom import cleanup, validity
    from geom.cell import UnitCell

    status, reasons = [], []
    for arr in fields:
        v = validity.check(UnitCell(arr))
        if v.ok:
            status.append('valid')
            reasons.append(None)
            continue
        result = cleanup.clean(arr)
        status.append('fixed' if result.fixed else 'unfixable')
        reasons.append(tuple(v.reasons))
    return np.array(status), reasons

def show(ax, arr, color, title=None):
    ax.imshow(arr, cmap='binary', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(color)
        side.set_linewidth(2.0)
    if title:
        ax.set_title(title, fontsize=7, color=TEXT_SECONDARY, pad=3)

def main():
    fields = load_fields()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    status, reasons = classify(fields)
    rng = np.random.default_rng(0)

    from diffusion import dataset
    real_arrs, frame = dataset.load()
    real_fm = frame['f_metal'].to_numpy()
    gen_fm = fields.reshape(len(fields), -1).mean(axis=1)

    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(5, N_COLS, height_ratios=[1, 1, 1, 1, 1.25], hspace=0.45,
                          wspace=0.12)

    # GENERATED, BY OUTCOME #
    picks = []
    for label, color in (('valid', OK), ('fixed', FIXED), ('unfixable', BAD)):
        idx = np.flatnonzero(status == label)
        if len(idx):
            picks.append((label, color, rng.choice(idx, min(N_COLS, len(idx)), replace=False)))
    for row, (label, color, idx) in enumerate(picks[:2]):
        for col in range(N_COLS):
            ax = fig.add_subplot(gs[row, col])
            if col < len(idx):
                show(ax, fields[idx[col]], color,
                     f'f={gen_fm[idx[col]]:.2f}' if row == 0 else None)
            else:
                ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=TEXT_PRIMARY)

    # FAILURE MODES #
    by_reason = collections.defaultdict(list)
    for i, rs in enumerate(reasons):
        if rs:
            for r in rs:
                by_reason[r].append(i)
    top = sorted(by_reason.items(), key=lambda kv: -len(kv[1]))[:3]
    for col in range(N_COLS):
        ax = fig.add_subplot(gs[2, col])
        group = col // 4
        if group < len(top):
            reason, idx = top[group]
            k = idx[(col % 4) % len(idx)]
            show(ax, fields[k], BAD, reason if col % 4 == 0 else None)
        else:
            ax.axis('off')

    # GENERATED vs REAL, MATCHED f_metal #
    order = rng.permutation(N_COLS)
    lo, hi = 0.30, 0.42
    gen_pool = np.flatnonzero((gen_fm > lo) & (gen_fm < hi) & (status == 'valid'))
    real_pool = np.flatnonzero((real_fm > lo) & (real_fm < hi))
    gen_pick = rng.choice(gen_pool, N_COLS, replace=False)
    real_pick = rng.choice(real_pool, N_COLS, replace=False)
    for col in range(N_COLS):
        ax = fig.add_subplot(gs[3, col])
        # Row order shuffled: half generated, half real, unlabeled.
        use_gen = order[col] % 2 == 0
        arr = fields[gen_pick[col]] if use_gen else real_arrs[real_pick[col]]
        show(ax, arr, TEXT_SECONDARY)
        if col == 0:
            ax.set_ylabel('mixed', fontsize=9, color=TEXT_PRIMARY)

    bottom = gs[4, :].subgridspec(1, 2, wspace=0.22)

    # f_metal DISTRIBUTIONS #
    ax = fig.add_subplot(bottom[0, 0])
    bins = np.linspace(0.05, 0.6, 50)
    ax.hist(real_fm, bins=bins, color=REAL, alpha=0.55, label='training set', zorder=3)
    ax.hist(gen_fm, bins=bins, color=GEN, alpha=0.55,
            weights=np.full(len(gen_fm), len(real_fm) / len(gen_fm)),
            label='generated (rescaled)', zorder=3)
    ax.axvline(config.F_METAL_MIN, color=BAD, lw=1, ls='--', zorder=4)
    ax.axvline(config.F_METAL_MAX, color=BAD, lw=1, ls='--', zorder=4,
               label='validity band')
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel('f_metal', color=TEXT_SECONDARY)
    ax.set_title(f'density learned unprompted: generated mean {gen_fm.mean():.4f} '
                 f'vs training {real_fm.mean():.4f}',
                 color=TEXT_PRIMARY, fontsize=11, loc='left')
    ax.legend(fontsize=8, frameon=False)

    # OUTCOME BREAKDOWN #
    ax = fig.add_subplot(gs[4, 6:])
    counts = collections.Counter(status)
    labels = ['valid', 'fixed', 'unfixable']
    vals = [counts.get(k, 0) for k in labels]
    ax.barh(labels, vals, color=[OK, FIXED, BAD], zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + len(fields) * 0.01, i, f'{v}  ({100 * v / len(fields):.1f}%)',
                va='center', fontsize=9, color=TEXT_PRIMARY)
    ax.grid(True, axis='x', color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlim(-len(fields) * 0.02, len(fields) * 1.25)
    ax.set_xlabel('samples', color=TEXT_SECONDARY)
    ax.set_title(f'cleanup moved only {100 * report["mean_change_fraction"]:.2f}% of pixels',
                 color=TEXT_PRIMARY, fontsize=11, loc='left')

    fig.suptitle(f'Unconditional samples - {len(fields):,} cells, '
                 f'{100 * report["raw_rate"]:.1f}% valid raw, '
                 f'{100 * report["cleaned_rate"]:.1f}% after cleanup '
                 f'({report["steps"]}-step DDIM)',
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor='white')

    print(f'{len(fields):,} samples')
    for k in labels:
        print(f'  {k:11} {counts.get(k, 0):5d}  ({100 * counts.get(k, 0) / len(fields):5.1f}%)')
    print(f'  f_metal generated {gen_fm.mean():.4f}  training {real_fm.mean():.4f}')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
