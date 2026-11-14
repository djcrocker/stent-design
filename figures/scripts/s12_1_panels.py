"""Figure 1 panels."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import config
from geom import tube
from geom.handmade import diamond

OUT = config.FIG_DEV_DIR
METAL = '#1a1a1a'
SURFACE = '#ffffff'
TEXT = '#0b0b0b'
STRUT = '#b0b6bd'

# The campaign bends 4 cells to R = 54 mm and looks straight.
# LONG_CELLS shows the SAME curvature over a longer segment.
TRUE_CELLS = 4
LONG_CELLS = 16

def flex(points, radius_mm, compression=0.0):
    """
    The deck's Euler-Bernoulli map (sim3d/apdl.py), applied to every node.

    A plane section at offset zeta from mid-span goes to a radial plane of the arc, so
    the displacement depends on the node's own y.
    """
    out = np.array(points, float)
    y, z = out[:, 1], out[:, 2]
    zmid = 0.5 * (z.min() + z.max())
    zeta = z - zmid
    sarc = zeta * (1.0 - compression)
    arm = radius_mm - y
    out[:, 1] = radius_mm - arm * np.cos(sarc / radius_mm)
    out[:, 2] = zmid + arm * np.sin(sarc / radius_mm)
    return out

def render_tube(mesh, path, zoom=1.25):
    """
    One tube on white, laid out for a pipeline row.

    The camera sits out on +x with y up, which puts the tube horizontal and arcs it
    upward.
    """
    center = np.array(mesh.center, float)
    plotter = pv.Plotter(off_screen=True, window_size=(1700, 850))
    plotter.background_color = SURFACE
    plotter.add_mesh(mesh, color=STRUT, specular=0.4, smooth_shading=False, show_edges=False)
    plotter.camera_position = [(center[0] + 1.0, center[1], center[2]), tuple(center), (0, 1, 0)]
    plotter.camera.azimuth = 10
    plotter.camera.elevation = 12
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    plotter.screenshot(str(path))
    plotter.close()
    print(f'  wrote {path.name}')

def tubes(cell):
    """Straight, flexed at the campaign's geometry, and flexed over a longer segment."""
    radius = config.FLEX_BEND_AMPLITUDE_RADIUS_MM
    for n_axial, tag in ((TRUE_CELLS, ''), (LONG_CELLS, '_long')):
        mesh = tube.tube_mesh(cell, n_axial=n_axial)
        length = mesh.bounds[5] - mesh.bounds[4]
        arc = np.degrees(length / radius)
        print(f'{n_axial} cells: {length:.2f} mm, arc {arc:.1f} deg at R = {radius} mm')
        render_tube(mesh, OUT / f's12_1_tube_straight{tag}.png')
        bent = mesh.copy()
        bent.points = flex(mesh.points, radius)
        render_tube(bent, OUT / f's12_1_tube_flexed{tag}.png')

def _panel(ax, field, extent_mm, bar_mm, label):
    ax.imshow(np.asarray(field, float).T, cmap=matplotlib.colors.ListedColormap(
        [SURFACE, METAL]), origin='lower', vmin=0, vmax=1,
        extent=(0, extent_mm[0], 0, extent_mm[1]), interpolation='nearest')
    margin = 0.22 * extent_mm[1]
    ax.set_xlim(0, extent_mm[0])
    ax.set_ylim(-margin, extent_mm[1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    y = -0.40 * margin
    ax.plot([0, bar_mm], [y, y], color=METAL, lw=3.0, solid_capstyle='butt')
    ax.text(bar_mm / 2, y - 0.13 * margin, label, ha='center', va='top',
            fontsize=10, color=TEXT)

def cell_and_tiling(cell):
    """The representation panel."""
    circ, axial = cell.extent_mm
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), facecolor=SURFACE)
    _panel(axes[0], cell.to_array(), (circ, axial), 0.5, '0.5 mm')
    _panel(axes[1], cell.tile(3, 3), (circ * 3, axial * 3), 1.0, '1 mm')
    for i in range(1, 3):
        axes[1].vlines(i * circ, 0, axial * 3, color='#d4562f', lw=0.7, alpha=0.75)
        axes[1].hlines(i * axial, 0, circ * 3, color='#d4562f', lw=0.7, alpha=0.75)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    path = OUT / 's12_1_cell_and_tiling.png'
    fig.savefig(path, dpi=400, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {path.name}  ({cell.n}x{cell.n}, f_metal {cell.f_metal:.3f})')

def check_flex():
    radius = config.FLEX_BEND_AMPLITUDE_RADIUS_MM
    z = np.linspace(0.0, 25.0, 41)
    axis = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    bent = flex(axis, radius)
    mid = len(z) // 2
    assert np.allclose(bent[mid], axis[mid]), 'mid-span moved'
    # Algebraic circle fit: the center is equidistant from every point on the arc.
    y, zz = bent[:, 1], bent[:, 2]
    A = np.column_stack([y, zz, np.ones_like(y)])
    sol, *_ = np.linalg.lstsq(A, y ** 2 + zz ** 2, rcond=None)
    cy, cz = sol[0] / 2, sol[1] / 2
    got = np.sqrt(sol[2] + cy ** 2 + cz ** 2)
    assert abs(got - radius) < 1e-6, f'curvature is {got:.4f} mm, not {radius}'
    print(f'flex map OK: mid-span fixed, arc radius {got:.4f} mm')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_flex()
    cell = diamond()
    cell_and_tiling(cell)
    tubes(cell)

if __name__ == '__main__':
    main()
