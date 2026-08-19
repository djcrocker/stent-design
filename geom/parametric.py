"""
Parametric baseline family (sine/crown), approximating Kamenskiy 2026.

This is what the generative model has to beat.

A crown cell is one axial period of a zigzag ring plus the link that joins it to the next
ring. Within our fixed cell envelope the free parameters are:

    `strut_width_mm`:     The strut's in-plane width
    `crown_amplitude`:    How far the zigzag swings axially, as a fraction of the cell
    `link_length`:        Axial length of the connector, as a fraction of the cell

Two of Kamenskiy's five parameters are absent:

    Strut thickness:    Radial / out-of-plane.
    #struts per circ:   That is N_CIRC, which defines the cell's physical width. Held fixed
                        at 12 for everything the generative model uses: the diffusion model
                        sees only pixels, so if physical scale varied across the dataset,
                        identical images would map to different y and the conditioning would
                        stop being a function. The baseline may sweep #struts separately, at
                        its own scale, for the Pareto comparison.

Widths snap to the pixel lattice. One pixel is ~0.0245 mm at GRID_N=64, so a
0.10 mm strut can only render as 4 or 5 px, a 10-25% quantisation across the useful range,
and distinct requests collapse to identical geometry (0.14 and 0.16 mm render the same
cell). Snapping makes the recorded parameter equal to the geometry actually produced, so
nothing downstream is labelled with a precision the pixels do not have. `snap_width_mm`
reports the achievable value; `achievable_widths` enumerates the sweep levels.
"""

import numpy as np

import config
from geom.cell import UnitCell

def snap_width_mm(width_mm, n=None):
    """
    Round a requested strut width to the nearest achievable pixel width.

    Never returns less than MIN_FEATURE_MM.
    """
    n = config.GRID_N if n is None else n
    px_mm = config.cell_extent_mm()[0] / n
    px = max(1, int(round(width_mm / px_mm)))
    if px * px_mm < config.MIN_FEATURE_MM:
        px = int(np.ceil(config.MIN_FEATURE_MM / px_mm))
    return px * px_mm


def achievable_widths(lo_mm=None, hi_mm=None, n=None):
    """Every distinct strut width the grid can actually represent in [lo, hi]."""
    n = config.GRID_N if n is None else n
    px_mm = config.cell_extent_mm()[0] / n
    lo_mm = config.MIN_FEATURE_MM if lo_mm is None else lo_mm
    hi_mm = 0.25 if hi_mm is None else hi_mm
    lo_px = int(np.ceil(lo_mm / px_mm))
    hi_px = int(np.floor(hi_mm / px_mm))
    return [px * px_mm for px in range(lo_px, hi_px + 1)]


def crown(strut_width_mm=None, crown_amplitude=0.30, n_periods=2, n=None):
    """
    One sine/crown unit cell: two phase-shifted zigzag rings joined by axial links.

    With identical tiling a single-ring cell can't carry a distinct link:
    a vertical connector would have to meet a peak at its lower end and a trough at its
    upper end at the same circumferential position, which no zigzag does, and a diagonal
    peak-to-trough connector is another zigzag strut, which brings the family back to
    the diamond. Real stents resolve this by phase-shifting alternate rings, so the true
    repeating unit spans two of them. Layout up the cell:

        [0, A]          ring 1, troughs at the bottom edge, peaks at A
        [A, A+L]        links, at ring 1's peak columns
        [A+L, 2A+L]     ring 2, phase-shifted half a period
        [2A+L, n]       links, at ring 2's peak columns -> meets the next cell's troughs

    Closing that loop forces 2(A + L) = n, so link length isn't free: it's
    `0.5 - crown_amplitude`. The free parameters are strut width, crown amplitude, and the
    number of zigzag periods around the cell, the last standing in for Kamenskiy's
    #struts-per-circumference within the fixed envelope, so N_CIRC and the physical scale
    never change.
    """
    n = config.GRID_N if n is None else n
    strut_width_mm = (config.STRUT_WIDTH_MM if strut_width_mm is None
                      else strut_width_mm)
    width_mm = snap_width_mm(strut_width_mm, n)

    if not 0.0 < crown_amplitude < 0.5:
        raise ValueError(
            f'crown_amplitude must be in (0, 0.5) - two rings and two link bands share the '
            f'cell height, so A + L = n/2. Got {crown_amplitude}'
        )
    if n_periods < 1:
        raise ValueError(f'n_periods must be >= 1, got {n_periods}')

    circ_mm, _ = config.cell_extent_mm()
    px_mm = circ_mm / n
    half_w = (width_mm / px_mm) / 2.0

    A = crown_amplitude * n              # Axial extent of one ring
    L = (0.5 - crown_amplitude) * n      # Link length, forced by the tiling
    p = float(n_periods)

    jj, ii = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing='ij')

    def zigzag(phase_shift, base):
        """Band of half-width half_w following a triangular wave starting at `base`."""
        frac = (ii * p / n + phase_shift) % 1.0
        tri = 2.0 * np.abs(frac - 0.5)                 # 1 at peaks, 0 at troughs
        ridge = base + A * tri
        slope = 2.0 * A * p / n
        return np.abs(jj - ridge) / np.hypot(1.0, slope) <= half_w

    def links(peak_phase, lo, hi):
        """Axial bars at the columns where that ring peaks."""
        frac = (ii * p / n + peak_phase) % 1.0
        at_peak = np.minimum(frac, 1.0 - frac) * (n / p) <= half_w
        return at_peak & (jj >= lo - half_w) & (jj <= hi + half_w)

    field = zigzag(0.0, 0.0)                       # ring 1: troughs at j=0, peaks at A
    field |= links(0.0, A, A + L)                  # links at ring 1's peaks
    field |= zigzag(0.5, A + L)                    # ring 2, half a period out of phase
    field |= links(0.5, 2 * A + L, n)              # links at ring 2's peaks

    return UnitCell(field)

# Default sweep axes. Widths come from the pixel lattice, so every level is distinct
# geometry rather than a distinct label on the same picture.
DEFAULT_AMPLITUDES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_PERIODS = (1, 2, 3)

def sweep(widths_mm=None, amplitudes=DEFAULT_AMPLITUDES, periods=DEFAULT_PERIODS, n=None):
    """
    Generate the family across the parameter grid.

    Returns (valid, rejected). `valid` is a list of (params, cell) for combinations that
    pass geom.validity; `rejected` is a list of (params, reasons). Rejection is expected at
    the corners of the grid - wide struts with many crown periods fill the cell
    with metal - and is reported, so the family's usable region is a measured fact.
    """
    from geom import validity

    widths_mm = achievable_widths(n=n) if widths_mm is None else widths_mm
    valid, rejected = [], []
    for w in widths_mm:
        for a in amplitudes:
            for p in periods:
                params = {'strut_width_mm': round(w, 4),
                          'crown_amplitude': a,
                          'n_periods': p,
                          'link_length': round(0.5 - a, 4)}
                cell = crown(strut_width_mm=w, crown_amplitude=a, n_periods=p, n=n)
                result = validity.check(cell)
                if result.ok:
                    valid.append((params | {'f_metal': round(result.metrics['f_metal'], 4)},
                                  cell))
                else:
                    rejected.append((params, result.reasons))
    return valid, rejected
