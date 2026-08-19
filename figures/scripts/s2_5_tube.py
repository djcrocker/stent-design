"""
pyvista shows a closed tubular stent built from a valid cell.

Also writes a standalone interactive viewer when the trame backend is available.

Usage: python figures/scripts/s2_5_tube.py
"""

import numpy as np
import pyvista as pv

import config
from geom import tube, validity
from geom.handmade import diamond

PNG = config.FIG_DEV_DIR / 's2_5_tube.png'
HTML = config.FIG_DEV_DIR / 's2_5_tube.html'
N_AXIAL = 6

def main():
    cell = diamond()
    ok, reasons = validity.is_valid(cell)
    print(f'Source cell valid={ok} {reasons}')

    mesh = tube.tube_mesh(cell, n_axial=N_AXIAL)
    sized = mesh.compute_cell_sizes()
    volume = float(sized['Volume'].sum())
    expected = tube.expected_volume_mm3(cell, n_axial=N_AXIAL)
    radial = np.hypot(mesh.points[:, 0], mesh.points[:, 1])

    print(f'hexahedra        {mesh.n_cells}')
    print(f'points           {mesh.n_points}')
    print(f'volume           {volume:.5f} mm^3  (analytic {expected:.5f}, '
          f'ratio {volume / expected:.4f})')
    print(f'inner radius     {radial.min():.4f} mm  (D/2 = {config.D_DEPLOYED_MM / 2})')
    print(f'outer radius     {radial.max():.4f} mm  '
          f'(+t = {config.D_DEPLOYED_MM / 2 + config.STRUT_THICKNESS_MM})')
    print(f'length           {mesh.bounds[5] - mesh.bounds[4]:.4f} mm '
          f'({N_AXIAL} cells)')
    print(f'open edges       {mesh.extract_surface(algorithm="dataset_surface").n_open_edges}'
          f'  (0 = closed ring)')

    PNG.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.add_mesh(mesh, color='#b0b6bd', specular=0.5, smooth_shading=False,
                     show_edges=False)
    plotter.add_axes()
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 20
    try:
        plotter.screenshot(str(PNG))
        print(f'\nWrote {PNG}')
    except Exception as exc:
        print(f'\nScreenshot unavailable ({type(exc).__name__}: {exc})')

    try:
        plotter.export_html(str(HTML))
        print(f'Wrote {HTML}')
    except Exception as exc:
        print(f'export_html unavailable ({type(exc).__name__}: {exc})')
    plotter.close()

if __name__ == "__main__":
    main()
