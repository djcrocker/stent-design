"""
This figure shows the process of the diffusion model, which is what makes it different 
from an optimizer: it doesn't refine one candidate toward an optimum, it denoises a 
sample drawn from pure Gaussian noise, and a different noise draw lands on a different 
valid design.

Three things are recorded at each DDIM step:
  x_t     - What the chain actually holds; mostly noise until late
  x0_hat  - The model's current guess at the finished cell, which is where structure
            becomes visible long before x_t looks like anything
  commit  - The fraction of pixels whose binarized x0_hat already matches the final
            design, which turns "structure appears early" into a number

Two phases:
    python figures/scripts/s8_traj.py sample
    python figures/scripts/s8_traj.py plot
"""

import argparse
import json

import numpy as np

import config

RESULTS = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_traj.npz'
REPORT = config.PROJECT_ROOT / 'diffusion' / 'results' / 's8_traj.json'
OUT_PNG = config.FIG_DEV_DIR / 's8_traj.png'
OUT_GIF = config.FIG_DEV_DIR / 's8_traj.gif'

TARGET = {'K_radial': 200.0, 'A_over_lim': 0.10}
N_SAMPLES = 8
STEPS = 50
GUIDANCE = 5.0

ACCENT = '#0E7C86'
COMMIT = '#2a78d6'
MARK = '#eb6834'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def sample_phase(n=N_SAMPLES, steps=STEPS, guidance=GUIDANCE, seed=3):
    """Torch only. One conditional batch, with every step retained."""
    import torch

    from diffusion import dataset, generate
    from diffusion.sample import load_model

    ddpm, norm, _ = load_model()
    _, frame = dataset.load()
    full, support = generate.complete_target(frame, TARGET)
    print(f'target {TARGET}  ->  {full}   ({support} supporting cells)', flush=True)

    torch.manual_seed(seed)
    z = torch.tensor(norm.transform_dict(full), dtype=torch.float32)
    y = z[None].repeat(n, 1).to(ddpm.device)
    x, (t_seq, xt, x0) = ddpm.ddim_sample(n, y, guidance=guidance, steps=steps, trajectory=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS,
        t=t_seq,
        xt=xt[:, :1, 0].numpy().astype(np.float16),
        x0=x0[:, :, 0].numpy().astype(np.float16),
        final=x.detach().float().cpu().numpy()[:, 0].astype(np.float16),
        target=np.array([full[k] for k in norm.keys]),
        target_keys=np.array(list(norm.keys)),
        guidance=np.array(guidance),
    )
    print(f'  Wrote {RESULTS.name}  ({RESULTS.stat().st_size / 1e6:.1f} MB)', flush=True)

def commitment(x0, final_bin):
    """Fraction of pixels already at their final value, per step, per sample."""
    guess = x0 > 0.0
    return (guess == final_bin[None]).mean(axis=(2, 3))

def plot_phase():
    """No torch. Contact sheet, commitment curve, and the animation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blob = np.load(RESULTS, allow_pickle=False)
    t = blob['t']
    xt = blob['xt'].astype(np.float32)
    x0 = blob['x0'].astype(np.float32)
    final = blob['final'].astype(np.float32)
    final_bin = final > 0.0
    n_steps, n = x0.shape[:2]

    frac = commitment(x0, final_bin)
    mean = frac.mean(axis=1)
    hit = int(np.argmax(mean >= 0.95))
    settle_step = hit if mean[hit] >= 0.95 else None

    labels = _labels(final_bin)
    report = {
        'target': {str(k): float(v) for k, v in zip(blob['target_keys'], blob['target'])},
        'guidance': float(blob['guidance']),
        'n_samples': int(n),
        'n_steps': int(n_steps),
        'commitment_mean': mean.round(4).tolist(),
        'settle_step_95pct': settle_step,
        'settle_fraction_of_run': None if settle_step is None else round(settle_step / n_steps, 3),
        'labels': labels,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')

    _contact_sheet(t, xt, x0, final_bin, labels, mean, settle_step, plt)
    _animate(x0)

    print(f'{n} samples, {n_steps} DDIM steps')
    if settle_step is not None:
        print(f'  95% of pixels settled by step {settle_step}/{n_steps} '
              f'({100 * settle_step / n_steps:.0f}% of the run, t={t[settle_step]})')
    for i, lab in enumerate(labels):
        flag = '' if lab['valid'] else '   invalid'
        print(f"  sample {i}: K {lab['K_radial']:7.1f}  A {lab['A_over_lim']:.3f}"
              f"  f_metal {lab['f_metal']:.3f}{flag}")
    print('Wrote', OUT_PNG)
    print('Wrote', OUT_GIF)

def _labels(final_bin):
    """Label the finished designs so the sheet ends on real numbers, not just pictures."""
    from diffusion.dataset import label_one
    from geom import validity
    from geom.cell import UnitCell

    out = []
    for arr in final_bin:
        cell = UnitCell(arr)
        v = validity.check(cell)
        row = {'valid': bool(v.ok), 'f_metal': float(arr.mean()),
               'K_radial': float('nan'), 'A_over_lim': float('nan')}
        if v.ok:
            try:
                lab = label_one(arr)
                row['K_radial'] = lab['K_radial']
                row['A_over_lim'] = lab['A_over_lim']
            except Exception as exc:                    # A label failure isn't fatal here
                row['error'] = str(exc)
        out.append(row)
    return out

def _contact_sheet(t, xt, x0, final_bin, labels, mean, settle_step, plt):
    """The noisy chain, the model's running guess, and the finished designs."""
    n_steps = len(t)
    cols = np.linspace(0, n_steps - 1, 9).round().astype(int)

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(4, 9, height_ratios=[1, 1, 1, 1.25], hspace=0.34, wspace=0.06)

    lim = float(np.abs(xt[:, 0]).max())
    for c, s in enumerate(cols):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(xt[s, 0], cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        ax.set_title(f't = {t[s]}', fontsize=8.5, color=TEXT_SECONDARY, pad=3)
        _bare(ax)
        if c == 0:
            _rowlabel(ax, 'x_t\nthe noisy chain')

        ax = fig.add_subplot(gs[1, c])
        ax.imshow(x0[s, 0], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        _bare(ax)
        if c == 0:
            _rowlabel(ax, 'x0 estimate\nwhere it is heading')

        ax = fig.add_subplot(gs[2, c])
        ax.imshow(x0[s, 0] > 0.0, cmap='binary', interpolation='nearest')
        ax.set_xlabel(f'{100 * mean[s]:.0f}% settled', fontsize=8, color=TEXT_SECONDARY,
                      labelpad=2)
        _bare(ax)
        if c == 0:
            _rowlabel(ax, 'thresholded\nthe design so far')

    # FINISHED DESIGNS #
    bottom = gs[3, :].subgridspec(1, 2, width_ratios=[5.0, 3.4], wspace=0.20)
    tiles = bottom[0, 0].subgridspec(1, 5, wspace=0.08)
    n_show = min(5, len(final_bin))
    for i in range(n_show):
        ax = fig.add_subplot(tiles[0, i])
        ax.imshow(final_bin[i], cmap='binary', interpolation='nearest')
        lab = labels[i]
        title = (f"K {lab['K_radial']:.0f}   A {lab['A_over_lim']:.3f}"
                 if lab['valid'] else 'invalid')
        ax.set_title(title, fontsize=8, color=TEXT_SECONDARY, pad=3)
        _bare(ax, hide_spines=False)
        for side in ax.spines.values():
            side.set_color(ACCENT)
            side.set_linewidth(1.6)
        if i == 0:
            _rowlabel(ax, 'finished\nsame target,\nsame model,\ndifferent noise')

    # COMMITMENT CURVE #
    ax = fig.add_subplot(bottom[0, 1])
    ax.plot(np.arange(n_steps), 100 * mean, color=COMMIT, lw=2.0, zorder=3)
    if settle_step is not None:
        ax.axvline(settle_step, color=MARK, lw=1.5, ls='--', zorder=2)
        ax.annotate(f'95% by step {settle_step}', (settle_step, 60), xytext=(6, 0),
                    textcoords='offset points', fontsize=8.5, color=MARK, va='center')
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.set_ylim(40, 102)
    ax.set_xlabel('DDIM step', color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel('% of pixels at\ntheir final value', color=TEXT_SECONDARY, fontsize=9)

    fig.suptitle('Noise to topology: one conditional DDIM run, every step retained',
                 color=TEXT_PRIMARY, fontsize=13, x=0.012, ha='left')
    fig.subplots_adjust(left=0.105, right=0.99, top=0.935, bottom=0.055)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor='white')
    plt.close(fig)

def _bare(ax, hide_spines=True):
    ax.set_xticks([])
    ax.set_yticks([])
    if hide_spines:
        for side in ax.spines.values():
            side.set_visible(False)

def _rowlabel(ax, text):
    ax.set_ylabel(text, fontsize=8.5, color=TEXT_PRIMARY, rotation=0, ha='right', va='center', labelpad=10)

def _animate(x0, hold=12):
    """All 8 samples denoising together, so variety and emergence read at the same time."""
    import imageio.v2 as imageio

    n_steps, n = x0.shape[:2]
    rows, cols = 2, (n + 1) // 2
    pad = 3
    h = w = x0.shape[-1]
    frames = []
    for s in list(range(n_steps)) + [n_steps - 1] * hold:
        sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad),
                        235, dtype=np.uint8)
        for i in range(n):
            r, c = divmod(i, cols)
            tile = np.where(x0[s, i] > 0.0, 20, 255).astype(np.uint8)
            y0 = pad + r * (h + pad)
            x_0 = pad + c * (w + pad)
            sheet[y0:y0 + h, x_0:x_0 + w] = tile
        frames.append(np.kron(sheet, np.ones((4, 4), dtype=np.uint8)))
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT_GIF, frames, duration=0.09, loop=0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='denoising trajectory')
    ap.add_argument('phase', choices=('sample', 'plot'))
    ap.add_argument('--n', type=int, default=N_SAMPLES)
    ap.add_argument('--steps', type=int, default=STEPS)
    args = ap.parse_args()
    if args.phase == 'sample':
        sample_phase(n=args.n, steps=args.steps)
    else:
        plot_phase()
