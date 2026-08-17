"""
Project-wide constants.

UNITS: mm / N / MPa / tonne — length mm, force N, stress MPa, mass tonne, density tonne/mm^3, time s. 
Ansys has to be set to the mm-t-s system too.

Values marked PROVISIONAL are placeholders that need a sourced value.
"""

import math
from pathlib import Path

UNITS_BANNER = (
    "UNITS mm-t-s: length mm | force N | stress MPa | mass tonne | "
    "density tonne/mm^3 | time s"
)

# PATHS #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
FIG_DEV_DIR = PROJECT_ROOT / 'figures' / 'dev'
FIG_PAPER_DIR = PROJECT_ROOT / 'figures' / 'paper'

# REPRESENTATION #
GRID_N = 64  # unit cell is a GRID_N x GRID_N binary material/void field

# GEOMETRY ENVELOPE (all provisional) #
D_DEPLOYED_MM = 6.0          # deployed stent diameter (SFA range 5-7)
D_CRIMPED_MM = 2.0           # crimped diameter
STRUT_THICKNESS_MM = 0.22    # radial thickness t
STRUT_WIDTH_MM = 0.20        # in-plane strut width w (also the min feature size)

N_CIRC = 12

AXIAL_PITCH_MM = math.pi * D_DEPLOYED_MM / N_CIRC  # 1.5708 mm at D=6, n_circ=12

# OBJECTIVE/CONDITIONING VECTOR #
OBJECTIVE_KEYS = ('K_radial', 'eps_a_max', 'A_over_lim', 'f_metal')

# Alternating-strain fatigue limit. 
# Pelton et al. 2008 measured a 10^7-cycle strain-amplitude limit of +/-0.4% and found 
# that mean strain does not cap fatigue life. Kamenskiy 2026 uses the same 0.4% and 
# reports >97% of optimized elements below it.
EPS_A_LIM = 0.004

# VALIDITY THRESHOLDS #
MIN_FEATURE_MM = STRUT_WIDTH_MM  # min strut width a valid cell must sustain
F_METAL_MIN = 0.15  # PROVISIONAL
F_METAL_MAX = 0.60  # PROVISIONAL

# NITINOL SUPERELASTIC (Auricchio) PARAMETERS #
# Sources from the baseline paper's own parameter set, so using it makes our
# Pareto comparison a like-for-like material comparison, so any difference comes 
# from topology, not material constants.
NITINOL = {
    'E_austenite_MPa': 65_000.0,
    'E_martensite_MPa': 23_500.0,
    'poisson_austenite': 0.33,
    'poisson_martensite': 0.33,
    'eps_transformation': 0.046,
    'eps_volumetric_transformation': 0.046,
    'sigma_start_loading_MPa': 465.0,
    'sigma_finish_loading_MPa': 535.0,
    'sigma_start_unloading_MPa': 227.0,
    'sigma_finish_unloading_MPa': 187.0,
    'sigma_start_loading_compression_MPa': 582.0,
    'density_tonne_per_mm3': 6.5e-9,  # 6.5 g/cm^3 converted to the mm-t-s system
}

# SOURCES #
SOURCES = {
    'NITINOL': (
        "Kamenskiy A, MacTaggart J, Desyatova A. Computational Optimization of a Stent "
        "for the Femoropopliteal Artery. Ann Biomed Eng. 2026;54(5):1287-1305. "
        "doi:10.1007/s10439-025-03968-9 (Table 2; values adopted there from Gokgol et al.)"
    ),
    'EPS_A_LIM': (
        "Pelton AR, Schroeder V, Mitchell MR, Gong X-Y, Barney M, Robertson SW. Fatigue "
        "and durability of Nitinol stents. J Mech Behav Biomed Mater. 2008;1(2):153-164. "
        "doi:10.1016/j.jmbbm.2007.08.001 (10^7-cycle amplitude limit +/-0.4%; mean strain "
        "does not cap life). Same 0.4% used by Kamenskiy 2026."
    ),
}

# PROVISIONAL VALUES #
PROVISIONAL = {
    'D_DEPLOYED_MM': (
        "SFA range 5-7 mm; confirm at G2."
    ),
    'D_CRIMPED_MM': "confirm against delivery-system sizing",
    'STRUT_THICKNESS_MM': "confirm at G2; sits inside Kamenskiy 2026 DOE1 (100-250 um)",
    'STRUT_WIDTH_MM': "confirm at G2; inside Kamenskiy DOE1 (100-250 um); sets MIN_FEATURE_MM",
    'F_METAL_MIN': "no source yet; tune against the S3 parametric family",
    'F_METAL_MAX': "no source yet; tune against the S3 parametric family",
}

def cell_extent_mm():
    """Physical size (circumferential, axial) of one unit cell, in mm."""
    if N_CIRC is None or AXIAL_PITCH_MM is None:
        raise ValueError("N_CIRC or AXIAL_PITCH_MM are unset")
    return (math.pi * D_DEPLOYED_MM / N_CIRC, AXIAL_PITCH_MM)


def mm_per_px():
    """
    Physical size of one grid pixel, in mm, as (circumferential, axial).
    Equal in both directions while AXIAL_PITCH_MM is left at its square-cell default.
    """
    circ_mm, axial_mm = cell_extent_mm()
    return (circ_mm / GRID_N, axial_mm / GRID_N)

def min_feature_px():
    """MIN_FEATURE_MM expressed in pixels (circumferential direction)."""
    return MIN_FEATURE_MM / mm_per_px()[0]

def summary():
    """Print the full configuration."""
    def mark(name):
        if name in PROVISIONAL:
            return ' [PROVISIONAL]'
        return ' [SOURCED]' if name in SOURCES else ""

    print(UNITS_BANNER)
    print(f"project root        {PROJECT_ROOT}")
    print(f"data dir            {DATA_DIR}  (exists={DATA_DIR.is_dir()})")
    print()
    print(f"GRID_N              {GRID_N}")
    print(f"D_DEPLOYED_MM       {D_DEPLOYED_MM}{mark('D_DEPLOYED_MM')}")
    print(f"D_CRIMPED_MM        {D_CRIMPED_MM}{mark('D_CRIMPED_MM')}")
    print(f"STRUT_THICKNESS_MM  {STRUT_THICKNESS_MM}{mark('STRUT_THICKNESS_MM')}")
    print(f"STRUT_WIDTH_MM      {STRUT_WIDTH_MM}{mark('STRUT_WIDTH_MM')}")
    print(f"N_CIRC              {N_CIRC}")
    print(f"AXIAL_PITCH_MM      {AXIAL_PITCH_MM:.4f}  (square-cell default)")
    circ_mm, axial_mm = cell_extent_mm()
    px_c, px_a = mm_per_px()
    print(f"cell extent (mm)    {circ_mm:.4f} circ x {axial_mm:.4f} axial")
    print(f"mm per pixel        {px_c:.5f} circ x {px_a:.5f} axial")
    print(f"min feature (px)    {min_feature_px():.1f}")
    print()
    print(f"OBJECTIVE_KEYS      {OBJECTIVE_KEYS}")
    print(f"EPS_A_LIM           {EPS_A_LIM}{mark('EPS_A_LIM')}")
    print(f"MIN_FEATURE_MM      {MIN_FEATURE_MM}")
    print(f"F_METAL range       [{F_METAL_MIN}, {F_METAL_MAX}]{mark('F_METAL_MIN')}")
    print()
    print(f"NITINOL{mark('NITINOL')}")
    for k, v in NITINOL.items():
        print(f"  {k:<28} {v}")
    print()
    print(f"{len(PROVISIONAL)} provisional values need a sourced number:")
    for name, why in PROVISIONAL.items():
        print(f"  {name:<20} {why}")
    print()
    print(f"{len(SOURCES)} sourced values:")
    for name, cite in SOURCES.items():
        print(f"  {name}")
        print(f"    {cite}")

if __name__ == "__main__":
    summary()