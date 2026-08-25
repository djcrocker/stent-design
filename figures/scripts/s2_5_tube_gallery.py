"""
S2.5 rendered one handmade cell. Here, we can see the handmade baseline next to designs
the model actually produced at different corners of the trade-off, all wrapped to the 
same 6 mm deployed tube so the difference is structural.

Usage: python figures/scripts/s2_5_tube_gallery.py
"""

import json

import numpy as np
import pyvista as pv

import config
from geom import tube, validity
from geom.cell import UnitCell
from geom.handmade import diamond

SHORTLIST = config.PROJECT_ROOT / 'screen' / 'results' / 's9_2_shortlist.json'
CELLS = config.PROJECT_ROOT / 'screen' / 'results' / 's9_2_shortlist_cells.npz'
PNG = config.FIG_DEV_DIR / 's2_5_tube_gallery.png'
HTML = config.FIG_DEV_DIR / 's2_5_tube_gallery.html'

N_AXIAL = 4
SPACING_MM = 8.2
DECIMATE = 0.85
COLORS = ('#9aa1a9', '#0E7C86', '#2a78d6', '#eb6834')

def pick():
    """
    The baseline plus three shortlist designs at distinct corners of the trade-off.

    Stiffest and most fatigue-resistant are the ends of the front; the balanced pick is
    the front design closest to the middle of it, so the row spans the trade rather than
    showing three variations on one end of it.
    """
    chosen = [{'label': 'handmade diamond\n(the S2.5 baseline)',
               'cell': diamond().to_array(), 'metrics': None}]
    if not (SHORTLIST.exists() and CELLS.exists()):
        print('no shortlist on disk; showing the baseline only')
        return chosen

    report = json.loads(SHORTLIST.read_text(encoding='utf-8'))
    blob = np.load(CELLS)
    fields = np.unpackbits(blob['fields'], count=int(np.prod(blob['shape']))).reshape(tuple(blob['shape'])).astype(bool)
    rows = report['shortlist']
    front = [i for i, r in enumerate(rows) if r['layer'] == 1]
    if not front:
        front = list(range(len(rows)))

    k = np.array([rows[i]['K_radial'] for i in front])
    a = np.array([rows[i]['A_over_lim'] for i in front])

    mid = np.abs(np.argsort(np.argsort(k)) / max(len(front) - 1, 1) - 0.5)

    picks = [(front[int(np.argmax(k))], 'stiffest on the front'),
             (front[int(np.argmin(a))], 'most fatigue-resistant'),
             (front[int(np.argmin(mid))], 'balanced')]

    seen = set()
    for i, why in picks:
        if i in seen:
            continue
        seen.add(i)
        r = rows[i]
        chosen.append({
            'label': f"{why}\nK {r['K_radial']:.0f}   A {r['A_over_lim']:.3f}",
            'cell': fields[i],
            'metrics': r,
        })
    return chosen

def main():
    picks = pick()
    plotter = pv.Plotter(off_screen=True, window_size=(2000, 560))
    labels, anchors = [], []
    span = SPACING_MM * (len(picks) - 1)

    for j, item in enumerate(picks):
        cell = UnitCell(np.asarray(item['cell'], dtype=bool))
        ok, reasons = validity.is_valid(cell)
        mesh = tube.tube_mesh(cell, n_axial=N_AXIAL)
        x = j * SPACING_MM - span / 2
        mesh = mesh.translate((x, 0.0, 0.0), inplace=False)
        # Only the outer shell is ever seen, and the shell is mostly coplanar voxel
        # faces: 288k triangles per tube collapse to 43k with the silhouette intact.
        shell = mesh.extract_surface(algorithm='dataset_surface').triangulate()
        shell = shell.decimate(DECIMATE)
        plotter.add_mesh(shell, color=COLORS[j % len(COLORS)], specular=0.4,
                         smooth_shading=False, show_edges=False)
        # Anchor the label above the tube so it stays put as the scene is rotated.
        labels.append(item['label'].replace('\n', '   '))
        anchors.append((x, 0.0, mesh.bounds[5] + 0.7))

        length = mesh.bounds[5] - mesh.bounds[4]
        print(f"{item['label'].splitlines()[0]:26}  valid={ok!s:5}  "
              f"{mesh.n_cells:6d} hexes  f_metal {cell.f_metal:.3f}  "
              f"length {length:.3f} mm")
        if not ok:
            print(f'    reasons: {reasons}')

    plotter.add_point_labels(np.array(anchors), labels, font_size=13,
                             text_color='#0b0b0b', shape=None, show_points=False,
                             always_visible=True, justification_horizontal='center')
    plotter.add_axes()
    plotter.camera_position = 'xz'
    plotter.camera.azimuth = -22
    plotter.camera.elevation = 14
    plotter.reset_camera()
    plotter.camera.zoom(2.3)

    PNG.parent.mkdir(parents=True, exist_ok=True)
    for path, fn in ((PNG, plotter.screenshot), (HTML, plotter.export_html)):
        try:
            fn(str(path))
            print(f'Wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)')
        except Exception as exc:
            print(f'{path.name} unavailable ({type(exc).__name__}: {exc})')
    plotter.close()

if __name__ == "__main__":
    main()
