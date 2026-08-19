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

# LIMB-FLEXION LOADING #
# The cyclic deformation the fatigue surrogate imposes. Segment choice matters: the FPA 
# isn't one loading environment. Default is the proximal SFA, matching where D_DEPLOYED_MM
# came from; FLEX_AXIAL_COMPRESSION_BY_SEGMENT holds the others so the harsher popliteal
# case can be run.
FLEX_SEGMENT = 'SFA'
FLEX_AXIAL_COMPRESSION_BY_SEGMENT = {   # (walking, most-flexed) peak axial compression
    'SFA': (0.09, 0.15),
    'AH': (0.11, 0.19),                 # adductor hiatus
    'PA': (0.13, 0.25),                 # popliteal
}
# Walking is the high-cycle case, and Pelton's limit is a 10^7-cycle limit, so the fatigue
# screen uses the walking value. Sitting and gardening are low-cycle and belong to a
# separate, static-strength question.
FLEX_AXIAL_COMPRESSION = FLEX_AXIAL_COMPRESSION_BY_SEGMENT[FLEX_SEGMENT][0]

# Fatigue is driven by the alternating component. The artery cycles between the standing
# baseline and the flexed state, so the amplitude is half the peak-to-peak range and the
# mean is compressive at the same magnitude. EPS_A_LIM is an amplitude limit, so this is
# the quantity that must be compared against it.
FLEX_AXIAL_AMPLITUDE = FLEX_AXIAL_COMPRESSION / 2.0
FLEX_AXIAL_MEAN = -FLEX_AXIAL_COMPRESSION / 2.0

# VALIDITY THRESHOLDS #
MIN_FEATURE_MM = 0.10
# Degeneracy guards, not design targets. Conventional self-expanding nitinol stents cover
# ~19-26% of the vessel wall (see sources); these span roughly half to twice that band so
# generative designs may leave conventional practice, while near-empty and near-solid
# fields are still rejected.
F_METAL_MIN = 0.10
F_METAL_MAX = 0.50

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
    'F_METAL_MIN': (
        "Vessel wall coverage of conventional self-expanding nitinol stents: SX 19.2+/-2.9%, "
        "Micro-SX 25.9+/-2.9%. Histopathologic evaluation of nitinol self-expanding stents in "
        "an animal model of advanced atherosclerotic lesions. EuroIntervention 2010; "
        "PMID 20142227. Rabbit aortic model, not femoropopliteal - used as an order-of-"
        "magnitude anchor for the guard rails, not as a design target."
    ),
    'MIN_FEATURE_MM': (
        "Conservative manufacturing floor. Modern nitinol stent struts run ~60-110 um "
        "(industry trend is 110 um down to 60-85 um) and laser kerf is 12-50 um "
        "(femtosecond ~12 um, fiber 15-50 um), so 100 um is comfortably producible. Also "
        "the lower bound of Kamenskiy 2026 DOE1 (100-250 um); their DOE2 explores to 50 um."
    ),
    'FLEX_AXIAL_COMPRESSION': (
        "Poulson W, Kamenskiy A, Seas A, Deegan P, Lomneth C, MacTaggart J. Limb "
        "flexion-induced axial compression and bending in human femoropopliteal artery "
        "segments. J Vasc Surg. 2018;67(2):607-613. doi:10.1016/j.jvs.2017.01.071 "
        "(PMID 28526560). Nitinol markers in 28 in-situ FPAs from 14 human cadavers, CT at "
        "standing 180deg / walking 110deg / sitting 90deg / gardening 60deg. Axial "
        "compression: SFA 9-15%, adductor hiatus 11-19%, popliteal 13-25%. Bending sphere "
        "radii 21-27 / 10-18 / 8-19 mm. Same group as the Kamenskiy 2026 baseline."
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