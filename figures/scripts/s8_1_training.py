"""
The DDPM training curve.

Left  - Train and val loss. Val sits slightly below train, which is expected: the train 
        figure is an epoch average over a model that improved during the epoch, dropout 
        is active in training and off at eval, and 10% of training samples have `y` 
        dropped for classifier-free guidance, which is a harder prediction than the 
        always-conditioned val pass.
Right - The same val curve on a log axis with the last-10-epoch improvement annotated,
        which is what says whether 60 epochs was enough or the run stopped early.

Usage: python figures/scripts/s8_1_training.py
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config

HISTORY = config.PROJECT_ROOT / 'diffusion' / 'checkpoints' / 'dataset_history.json'
OUT = config.FIG_DEV_DIR / 's8_1_training.png'

TRAIN = '#2a78d6'
VAL = '#eb6834'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def main():
    history = json.loads(HISTORY.read_text(encoding='utf-8'))
    epoch = np.array([h['epoch'] for h in history])
    train = np.array([h['train_loss'] for h in history])
    val = np.array([h['val_loss'] for h in history])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(epoch, train, color=TRAIN, lw=2, label='train (epoch average)', zorder=3)
    ax.plot(epoch, val, color=VAL, lw=2, label='val (fixed noise)', zorder=3)
    # A model predicting nothing scores 1.0 against unit-variance noise.
    ax.axhline(1.0, color=TEXT_SECONDARY, lw=1, ls=':', zorder=2,
               label='predicting nothing (1.0)')
    style(ax)
    ax.set_yscale('log')
    ax.set_xlabel('epoch', color=TEXT_SECONDARY)
    ax.set_ylabel('epsilon-prediction MSE', color=TEXT_SECONDARY)
    ax.set_title('training curve', color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    ax.plot(epoch, val, color=VAL, lw=2, marker='o', ms=3, zorder=3)
    style(ax)
    ax.set_xlabel('epoch', color=TEXT_SECONDARY)
    ax.set_ylabel('val loss', color=TEXT_SECONDARY)
    tail = 100 * (val[-10] - val[-1]) / val[-10]
    ax.axvspan(epoch[-10], epoch[-1], color=TEXT_SECONDARY, alpha=0.10, zorder=1)
    ax.set_title(f'converged: last 10 epochs improved val by {tail:.1f} %',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.annotate(f'final {val[-1]:.5f}', xy=(epoch[-1], val[-1]),
                xytext=(-90, 26), textcoords='offset points', fontsize=9,
                color=TEXT_PRIMARY,
                arrowprops=dict(arrowstyle='->', color=TEXT_SECONDARY, lw=1))

    explained = 100 * (1 - val[-1])
    fig.suptitle(f'Conditional DDPM - 18.60 M parameters, {len(history)} epochs, '
                 f'{history[-1]["minutes"]:.0f} min on an RTX 5080   |   '
                 f'final val {val[-1]:.5f} (~{explained:.1f} % of noise variance explained)',
                 color=TEXT_PRIMARY, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')

    print(f'{len(history)} epochs, {history[-1]["minutes"]:.1f} min')
    print(f'  train {train[0]:.5f} -> {train[-1]:.5f}')
    print(f'  val   {val[0]:.5f} -> {val[-1]:.5f}')
    print(f'  last 10 epochs improved val by {tail:.1f} %')
    print(f'  val below train by {100 * (train[-1] - val[-1]) / train[-1]:.1f} % '
          f'(dropout + CFG-dropout gap)')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
