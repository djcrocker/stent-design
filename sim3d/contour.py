"""
Binary field -> smooth boundary polygons.

In a voxel tube, stair-stepped strut edges create artificial stress concentrations at crowns 
and necks, which is where eps_a_max is read, so meshing that way would bias the objective.

Method: Gaussian-blur the binary field, then take the 0.5 level set. Smoothing the field
rather than the extracted polygon is the standard topology-optimisation -> CAD route; it
produces smooth contours by construction instead of trying to repair a staircase after the
fact, and it can't fold a polygon through itself.

Smoothing thins struts, and MIN_FEATURE_MM is only ~4 px, so sigma is not a free parameter:
`measure_min_width` re-measures the smoothed geometry and the caller is expected to check it
against the manufacturing floor rather than assume.

Everything here is periodic: the field is blurred with wrap mode and contours are extracted
from a tiled, padded image so a strut crossing the cell boundary is one continuous curve
rather than two that stop at the edge.
"""

import numpy as np
from scipy import ndimage
from skimage import measure

import config

LEVEL = 0.5

def smoothed_field(cell, sigma_px=0.8, n_circ=None, n_axial=1):
    """
    Tiled, periodically-blurred field ready for contouring.

    Returns (field, pad_px) where pad_px is the padding added on each side; contours are
    extracted from the padded field and then cropped back, so curves crossing the wrap are
    continuous rather than clipped.
    """
    n_circ = config.N_CIRC if n_circ is None else n_circ
    tiled = cell.tile(n_circ, n_axial).astype(float)

    pad = max(4, int(np.ceil(4 * sigma_px)))
    padded = np.pad(tiled, pad, mode='wrap')
    blurred = ndimage.gaussian_filter(padded, sigma_px, mode='wrap')
    return blurred, pad


def contours(cell, sigma_px=0.8, n_circ=None, n_axial=1):
    """
    Boundary polygons of the smoothed field, in PIXEL coordinates of the tiled cell.

    Coordinates are (row, col) = (axial, circumferential), matching the array convention
    used everywhere else. Polygons crossing the wrap continue past the tile extent; the
    caller maps them onto the cylinder, where the wrap closes them.
    """
    field, pad = smoothed_field(cell, sigma_px, n_circ, n_axial)
    polys = measure.find_contours(field, LEVEL)
    return [p - pad for p in polys]

def ridge_widths(cell, sigma_px=0.8, n_circ=None, n_axial=1, upsample=4):
    """
    Local thickness (mm) sampled along the medial ridge of the smoothed geometry.

    Measured on an upsampled raster: at native resolution the distance transform is
    pixel-quantised, so the answer snaps to multiples of a pixel and is blind to exactly the
    sub-pixel change that smoothing causes.
    """
    field, pad = smoothed_field(cell, sigma_px, n_circ, n_axial)
    inner = field[pad:-pad, pad:-pad]
    if not (inner >= LEVEL).any():
        return np.array([])

    fine = ndimage.zoom(inner, upsample, order=1, mode='grid-wrap')
    solid = fine >= LEVEL
    half = min(solid.shape) // 2
    dist = ndimage.distance_transform_edt(np.pad(solid, half, mode='wrap'))
    dist = dist[half:-half, half:-half]

    ridge = (dist > 0) & (dist >= ndimage.maximum_filter(dist, size=3) - 1e-9)
    if not ridge.any():
        return np.array([])
    return 2.0 * dist[ridge] * (config.mm_per_px()[0] / upsample)

def measure_min_width(cell, sigma_px=0.8, n_circ=None, n_axial=1, upsample=4, percentile=1.0):
    """
    Narrowest feature of the smoothed geometry, in mm.

    The raw minimum isn't trustworthy here. Medial-ridge detection throws off isolated
    spurious points, and the wider the field the more of them appear. The 1st percentile
    agrees between the 2 to 4 decimal places at every smoothing level, so that's what 
    gets reported and checked against MIN_FEATURE_MM.
    """
    widths = ridge_widths(cell, sigma_px, n_circ, n_axial, upsample)
    return 0.0 if widths.size == 0 else float(np.percentile(widths, percentile))

def metal_fraction(cell, sigma_px=0.8, n_circ=None, n_axial=1, upsample=4):
    """
    Metal area fraction of the smoothed geometry.

    Taken from the rasterised level set: find_contours returns outer boundaries and holes,
    so summing |area| adds the holes instead of subtracting them.
    """
    field, pad = smoothed_field(cell, sigma_px, n_circ, n_axial)
    fine = ndimage.zoom(field[pad:-pad, pad:-pad], upsample, order=1, mode='grid-wrap')
    return float((fine >= LEVEL).mean())

def polygon_area_mm2(poly):
    """Signed area of a pixel-coordinate polygon, in mm^2 (shoelace)."""
    mm_px = config.mm_per_px()[0]
    r, c = poly[:, 0] * mm_px, poly[:, 1] * mm_px
    return 0.5 * float(np.sum(r * np.roll(c, -1) - np.roll(r, -1) * c))

def summarise(cell, sigma_px=0.8, n_circ=None, n_axial=1):
    """Contour statistics, for choosing sigma."""
    polys = contours(cell, sigma_px, n_circ, n_axial)
    smoothed = metal_fraction(cell, sigma_px, n_circ, n_axial)
    return {
        'sigma_px': sigma_px,
        'sigma_mm': sigma_px * config.mm_per_px()[0],
        'n_polygons': len(polys),
        'n_points': int(sum(len(p) for p in polys)),
        'min_width_mm': measure_min_width(cell, sigma_px, n_circ, n_axial),
        'min_width_raw_mm': float(ridge_widths(cell, sigma_px, n_circ, n_axial).min()),
        'min_feature_mm': config.MIN_FEATURE_MM,
        'f_metal_smoothed': smoothed,
        'f_metal_voxel': cell.f_metal,
    }
