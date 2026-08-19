"""
Grid is amplitude across, strut width down, at a fixed period count. Rejected corners are
drawn greyed with their reason.

Usage: python figures/scripts/s3_1_parametric.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import validity as V
from geom.parametric import DEFAULT_AMPLITUDES, achievable_widths, crown, sweep
from geom.render import plot_cell

OUT = config.FIG_DEV_DIR / 's3_1_parametric.png'
PERIODS = 2

def main():
    widths = achievable_widths()
    amps = DEFAULT_AMPLITUDES

    fig, axes = plt.subplots(len(widths), len(amps),
                             figsize=(2.6 * len(amps), 2.6 * len(widths)))
    for row, w in enumerate(widths):
        for col, a in enumerate(amps):
            ax = axes[row][col]
            cell = crown(strut_width_mm=w, crown_amplitude=a, n_periods=PERIODS)
            result = V.check(cell)
            plot_cell(cell, ax=ax,
                      title=(f'w={w:.3f} amp={a:.2f}\n'
                             f'f={result.metrics["f_metal"]:.3f}'
                             + ('' if result.ok else '\n' + ','.join(result.reasons))))
            ax.title.set_color('black' if result.ok else 'tab:red')
            ax.title.set_fontsize(9)
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'Parametric crown family, {PERIODS} periods '
                 f'(link length = 0.5 - amplitude, forced by tiling)', fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)

    valid, rejected = sweep()
    coverage = np.array([p['f_metal'] for p, _ in valid])
    print(f'sweep            {len(valid)} valid / {len(valid) + len(rejected)} combinations')
    print(f'widths (mm)      {[round(w, 4) for w in widths]}  ({len(widths)} pixel-exact levels)')
    print(f'amplitudes       {list(amps)}')
    print(f'periods          {list(range(1, 4))}')
    print(f'f_metal range    {coverage.min():.3f} - {coverage.max():.3f}'
          f'   (conventional stents ~0.19-0.26)')
    codes = {}
    for _, reasons in rejected:
        for c in reasons:
            codes[c] = codes.get(c, 0) + 1
    print(f'rejections       {codes}')
    print(f'\nWrote {OUT}')

if __name__ == '__main__':
    main()
