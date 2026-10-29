"""
Parametric baseline family (sine/crown), approximating Kamenskiy 2026.

This is what the generative model has to beat.

A crown cell is one axial period of a zigzag ring plus the link that joins it to the next
ring. Within our fixed cell envelope the free parameters are:

    `strut_width_mm`:     The strut's in-plane width
    `crown_amplitude`:    How far the zigzag swings axially, as a fraction of the cell
    `link_length`:        Axial length of the connector, as a fraction of the cell
    `band_width_mm`:      Width of a continuous circumferential band at the cell seam
    `width_taper`:        How strut width varies between crown apex and mid-span

Two of Kamenskiy's five parameters are absent:

    Strut thickness:    Radial / out-of-plane.
    #struts per circ:   That's N_CIRC, which defines the cell's physical width. Held fixed
                        at 12 for everything the generative model uses: the diffusion model
                        sees only pixels, so if physical scale varied across the dataset,
                        identical images would map to different y and the conditioning would
                        stop being a function. The baseline may sweep #struts separately, at
                        its own scale, for the Pareto comparison.

Widths snap to the pixel lattice. One pixel is ~0.0245 mm at GRID_N=64, so a
0.10 mm strut can only render as 4 or 5 px, a 10-25% quantization across the useful range,
and distinct requests collapse to identical geometry (0.14 and 0.16 mm render the same
cell). Snapping makes the recorded parameter equal to the geometry actually produced, so
nothing downstream is labeled with a precision the pixels do not have. `snap_width_mm`
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


def crown(strut_width_mm=None, crown_amplitude=0.30, n_periods=2, n=None, band_width_mm=0.0, width_taper=0.0):
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

    `band_width_mm` lays a continuous circumferential band across the cell seam, which welds
    ring 1's troughs into an unbroken hoop.

    `width_taper` in (-1, 1) scales strut half-width by `1 + taper * (2*apexness - 1)`, with
    apexness `|cos(2*pi*phase)|` so it's 1 at both turns of the wave and 0 at mid-span.
    Positive thickens the apexes and thins the mid-spans, negative the reverse, and 0 is
    exactly the uniform strut this family had before. Applied to the zigzag rings only, not
    the axial links: one member with a varying width is all the claim needs, and a second
    taper law would be more code expressing the same capability.
    """
    n = config.GRID_N if n is None else n
    strut_width_mm = (config.STRUT_WIDTH_MM if strut_width_mm is None else strut_width_mm)
    width_mm = snap_width_mm(strut_width_mm, n)

    if not 0.0 < crown_amplitude < 0.5:
        raise ValueError(
            f'crown_amplitude must be in (0, 0.5) - two rings and two link bands share the '
            f'cell height, so A + L = n/2. Got {crown_amplitude}'
        )
    if n_periods < 1:
        raise ValueError(f'n_periods must be >= 1, got {n_periods}')
    if not -1.0 < width_taper < 1.0:
        raise ValueError(
            f'width_taper must be in (-1, 1) - at +-1 the thin end of the strut reaches '
            f'zero width. Got {width_taper}'
        )
    if band_width_mm < 0.0:
        raise ValueError(f'band_width_mm must be >= 0, got {band_width_mm}')

    circ_mm, _ = config.cell_extent_mm()
    px_mm = circ_mm / n
    half_w = (width_mm / px_mm) / 2.0

    A = crown_amplitude * n              # Axial extent of one ring
    L = (0.5 - crown_amplitude) * n      # Link length, forced by the tiling
    p = float(n_periods)

    jj, ii = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing='ij')

    def zigzag(phase_shift, base):
        """Band following a triangular wave from `base`, tapered if asked."""
        frac = (ii * p / n + phase_shift) % 1.0
        tri = 2.0 * np.abs(frac - 0.5)                 # 1 at peaks, 0 at troughs
        ridge = base + A * tri
        slope = 2.0 * A * p / n
        # 1 at both turns of the wave, 0 at mid-span, so the taper is symmetric about
        # each apex rather than favoring peaks over troughs.
        apexness = np.abs(np.cos(2.0 * np.pi * frac))
        hw = half_w * (1.0 + width_taper * (2.0 * apexness - 1.0))
        return np.abs(jj - ridge) / np.hypot(1.0, slope) <= hw

    def links(peak_phase, lo, hi):
        """Axial bars at the columns where that ring peaks."""
        frac = (ii * p / n + peak_phase) % 1.0
        at_peak = np.minimum(frac, 1.0 - frac) * (n / p) <= half_w
        return at_peak & (jj >= lo - half_w) & (jj <= hi + half_w)

    field = zigzag(0.0, 0.0)                       # ring 1: troughs at j=0, peaks at A
    field |= links(0.0, A, A + L)                  # links at ring 1's peaks
    field |= zigzag(0.5, A + L)                    # ring 2, half a period out of phase
    field |= links(0.5, 2 * A + L, n)              # links at ring 2's peaks

    if band_width_mm > 0.0:
        half_b = (snap_width_mm(band_width_mm, n) / px_mm) / 2.0
        field |= np.minimum(jj, n - jj) <= half_b

    return UnitCell(field)

# Default sweep axes. Widths come from the pixel lattice, so every level is distinct
# geometry rather than a distinct label on the same picture.
DEFAULT_AMPLITUDES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_PERIODS = (1, 2, 3)

# Extended axes for the fair baseline (S10.1 Part 2).
DEFAULT_BANDS_MM = (0.0, 0.10, 0.15)
DEFAULT_TAPERS = (0.0, 0.25, 0.45, -0.25, -0.45)

def sweep(widths_mm=None, amplitudes=DEFAULT_AMPLITUDES, periods=DEFAULT_PERIODS, n=None, bands_mm=(0.0,), tapers=(0.0,)):
    """
    Generate the family across the parameter grid.

    Returns (valid, rejected). `valid` is a list of (params, cell) for combinations that
    pass geom.validity; `rejected` is a list of (params, reasons). Rejection is expected at
    the corners of the grid - wide struts with many crown periods fill the cell
    with metal - and is reported, so the family's usable region is a measured fact.

    `bands_mm` and `tapers` default to the single no-op level, so calling `sweep()` returns
    exactly the family that Stages 3-10.3 were measured on. `extended_sweep()` is the opt-in.
    """
    from geom import validity

    widths_mm = achievable_widths(n=n) if widths_mm is None else widths_mm
    valid, rejected = [], []
    for w in widths_mm:
        for a in amplitudes:
            for p in periods:
                for b in bands_mm:
                    for t in tapers:
                        params = {'strut_width_mm': round(w, 4),
                                  'crown_amplitude': a,
                                  'n_periods': p,
                                  'link_length': round(0.5 - a, 4)}
                        # Only recorded when they are doing something, so a default sweep's
                        # params are byte-identical to what S10.3 and earlier stages stored.
                        if b or t:
                            params |= {'band_width_mm': round(b, 4),
                                       'width_taper': round(t, 4)}
                        cell = crown(strut_width_mm=w, crown_amplitude=a, n_periods=p, n=n,
                                     band_width_mm=b, width_taper=t)
                        result = validity.check(cell)
                        if result.ok:
                            valid.append(
                                (params | {'f_metal': round(result.metrics['f_metal'], 4)},
                                 cell))
                        else:
                            rejected.append((params, result.reasons))
    return valid, rejected

def extended_sweep(n=None):
    """
    The fair baseline family: the original grid crossed with hoop bands and width tapers.

    This is the population S10.1 (Part 3) runs in 3D and S10.1 (Part 4) re-measures S10.3's
    exclusion against. `sweep()` stays the published family so the two can be compared.
    """
    return sweep(n=n, bands_mm=DEFAULT_BANDS_MM, tapers=DEFAULT_TAPERS)
