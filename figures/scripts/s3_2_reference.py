"""
Writes data/reference_cell.npy (the array), data/reference_cell.json (the recipe and the
config fingerprint) and figures/dev/reference_cell.png (cell, 3x3 tiling, and the tube it
wraps onto).

Usage: python figures/scripts/s3_2_reference.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import reference, tube
from geom import validity as V
from geom.render import plot_cell, plot_tiling

def main():
    cell = reference.save()
    result = V.check(cell)

    print('Reference cell')
    for k, v in reference.PARAMS.items():
        print(f'  {k:20} {v}')
    print(f'  {"link_length":20} {0.5 - reference.PARAMS["crown_amplitude"]}  (derived)')
    print(f'  {"f_metal":20} {result.metrics["f_metal"]:.4f}   '
          f'(conventional stents ~0.19-0.26)')
    print(f'  {"thin_fraction":20} {result.metrics["thin_fraction"]:.4f}')
    print(f'  {"valid":20} {result.ok} {result.reasons}')
    print('  Config fingerprint')
    for k, v in reference.fingerprint().items():
        print(f'    {k:18} {v}')

    mesh = tube.tube_mesh(cell, n_axial=6)
    volume = float(mesh.compute_cell_sizes()['Volume'].sum())
    print(f'  {"tube (6 cells)":20} {mesh.n_cells} hexes, {volume:.4f} mm^3')

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_cell(cell, ax=ax1, title=f'reference cell  f_metal={cell.f_metal:.3f}')
    ax2 = fig.add_subplot(1, 3, 2)
    plot_tiling(cell, 3, 3, ax=ax2, title='3 x 3 tiling')

    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.add_mesh(mesh, color='#b0b6bd', specular=0.5)
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 20
    shot = plotter.screenshot(return_img=True)
    plotter.close()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(shot)
    ax3.set_title('wrapped to tube (6 cells)')
    ax3.axis('off')

    fig.tight_layout(pad=1.6)
    reference.PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(reference.PNG_PATH, dpi=130, bbox_inches='tight')

    print(f'\nWrote {reference.NPY_PATH}')
    print(f'Wrote {reference.JSON_PATH}')
    print(f'Wrote {reference.PNG_PATH}')

if __name__ == "__main__":
    main()
