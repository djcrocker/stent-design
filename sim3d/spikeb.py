"""
Label the same cells with both tiers, so ranking can be tested.

The cheap 2D screen only has to rank designs like 3D truth, not agree with it. S5.4 made
the distinction load-bearing: on the reference cell the three 2D/3D ratios came out 0.764x,
1.127x and 0.692x, so no single calibration offset can fix the screen.

The cell set is two populations:
  - a stratified sample of the crown parametric family (the screen's home ground), and
  - the handmade diamond/grid cells, which are different topology families entirely.
If the rank correlation holds only within the crown family, that is a finding about
generalization.
"""

import json
import pathlib

import numpy as np

import config
from geom import handmade, parametric
from sim2d.fatigue import fatigue
from sim2d.homogenize import homogenize

MESH = {'n_circ': 1, 'n_axial': 2, 'layers': 4, 'limit': 0.20}
MAX_FACE_ANGLE_DEG = 155.0

# 54 crown + 5 handmade + 1 reference = 60 cells
N_PARAMETRIC = 54
DECK_DIR = config.PROJECT_ROOT / 'sim3d' / 'decks'
RESULTS_DIR = config.PROJECT_ROOT / 'sim3d' / 'results'
ANSYS_EXE = (r'C:\Program Files\ANSYS Inc\ANSYS Student\v261' r'\ansys\bin\winx64\ANSYS261.exe')

def _param_matrix(entries):
    """Parameters normalized to [0, 1] so distances weight each axis equally."""
    raw = np.array([[p['strut_width_mm'], p['crown_amplitude'], float(p['n_periods'])] for p, _ in entries])
    lo, hi = raw.min(axis=0), raw.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (raw - lo) / span

def stratify(entries, n):
    """
    Farthest-point sample of the parameter grid.

    Taking the first n of a sorted sweep would cluster in one corner and understate the
    diversity the correlation is meant to be tested over. This starts from the cell nearest
    the centroid, then repeatedly adds whichever remaining cell is farthest from everything
    already chosen.
    """
    if n >= len(entries):
        return list(range(len(entries)))
    X = _param_matrix(entries)
    chosen = [int(np.argmin(np.linalg.norm(X - X.mean(axis=0), axis=1)))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return sorted(chosen)

def select_cells(n_parametric=N_PARAMETRIC, include_handmade=True, include_reference=True):
    """Cell set. Returns a list of spec dicts, in stable order."""
    valid, _ = parametric.sweep()
    specs = []
    for i, idx in enumerate(stratify(valid, n_parametric)):
        params, cell = valid[idx]
        specs.append({'name': f's6_cell{i:02d}', 'family': 'crown', 'params': dict(params), 'cell': cell})
    if include_handmade:
        for j, (label, factory) in enumerate(sorted(handmade.VALID_CELLS.items())):
            specs.append({'name': f's6_hand{j:02d}', 'family': 'handmade', 'params': {'label': label}, 'cell': factory()})
    if include_reference:
        from geom import reference
        specs.append({'name': 's6_ref', 'family': 'reference', 'params': dict(reference.PARAMS), 'cell': reference.build()})
    return specs

def label_2d(cell):
    """The 2D screen's labels for one cell."""
    h = homogenize(cell)
    f = fatigue(cell, h)
    return {'K_radial_2D': float(h.K_radial),
            'eps_a_max_2D': float(f.eps_a_max),
            'eps_a_p99_2D': float(f.eps_a_p99),
            'A_over_lim_2D': float(f.A_over_lim),
            'f_metal': float(cell.f_metal)}

def build(specs, directory=None, mesh=None, nsubst=None):
    """Write a mesh and a deck per cell, and attach the 2D labels."""
    from sim3d import apdl, mesh3d

    directory = DECK_DIR if directory is None else pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mesh = MESH if mesh is None else mesh
    for spec in specs:
        stem = spec['name']
        info, q = mesh3d.build_and_write(spec['cell'], directory / f'{stem}_mesh.inp',
                                         n_circ=mesh['n_circ'], n_axial=mesh['n_axial'],
                                         layers=mesh['layers'], limit=mesh['limit'])
        if q['max_face_angle_deg'] > MAX_FACE_ANGLE_DEG or q['n_nonpositive']:
            raise ValueError(f"{stem}: mesh would be rejected by Ansys "
                             f"({q['max_face_angle_deg']:.1f} deg, "
                             f"{q['n_nonpositive']} non-positive)")
        deck = directory / f'{stem}.inp'
        kwargs = {} if nsubst is None else {'nsubst': nsubst}
        deck.write_text(apdl.spike_a_loadsteps(mesh=f'{stem}_mesh', deck_dir=directory,
                                               out_stem=stem, outres='ALL,LAST', **kwargs),
                        encoding='utf-8')
        spec['deck'] = str(deck)
        spec['quality'] = q
        spec['labels_2d'] = label_2d(spec['cell'])
    return specs

def write_batch_script(specs, path=None, out_dir=None, np_cores=4, exe=None):
    """A PowerShell script that runs every deck unattended."""
    path = (DECK_DIR / 'run_spikeb.ps1') if path is None else pathlib.Path(path)
    out_dir = (config.PROJECT_ROOT / 'ansys_output') if out_dir is None else out_dir
    exe = ANSYS_EXE if exe is None else exe
    names = ', '.join(chr(34) + s['name'] + chr(34) for s in specs)
    L = ['# Spike B batch. Generated by spikeb.py.',
         '# Each job is independent.',
         f'$exe = "{exe}"',
         f'$out = "{out_dir}"',
         f'$decks = "{DECK_DIR}"',
         'if (-not (Test-Path $out)) { New-Item -ItemType Directory $out | Out-Null }',
         '',
         '# Refuse to start if another instance is mid-solve.',
         '$busy = Get-ChildItem (Join-Path $out "s6_*.lock") -ErrorAction SilentlyContinue',
         'if ($busy) {',
         '  Write-Host "ABORT: a Spike B job is already running:" -ForegroundColor Red',
         '  $busy | ForEach-Object { Write-Host "  $($_.Name)" }',
         '  Write-Host "Wait for it to finish, or delete the lock if the job is dead."',
         '  exit 1',
         '}',
         '',
         '# Timestamped.',
         '$log = Join-Path $out ("spikeb_timing_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv")',
         '"name,seconds,exit" | Out-File -FilePath $log -Encoding utf8',
         f'$names = @({names})',
         'foreach ($n in $names) {',
         '  $deck = Join-Path $decks "$n.inp"',
         '  $lock = Join-Path $out "$n.lock"',
         '  if (Test-Path $lock) {',
         '    Write-Host "=== $n === SKIPPED (stale lock)" -ForegroundColor Yellow',
         '    "$n,0,locked" | Out-File -FilePath $log -Append -Encoding utf8',
         '    continue',
         '  }',
         '  Write-Host "=== $n ==="',
         '  $t0 = Get-Date',
         f'  & $exe -b -np {np_cores} -dir $out -j $n -i $deck -o (Join-Path $out "$n.out")',
         '  $code = $LASTEXITCODE',
         '  $secs = [int]((Get-Date) - $t0).TotalSeconds',
         '  "$n,$secs,$code" | Out-File -FilePath $log -Append -Encoding utf8',
         '  Write-Host "    $secs s (exit $code)"',
         '}',
         'Write-Host "Timing written to $log"']
    path.write_text(chr(10).join(L) + chr(10), encoding='utf-8')
    return path

def write_manifest(specs, path=None):
    """Record which cells were chosen and their 2D labels, before any 3D run."""
    path = (RESULTS_DIR / 's6_1_manifest.json') if path is None else pathlib.Path(path)
    payload = {'mesh': MESH,
               'cells': [{'name': s['name'], 'family': s['family'], 'params': s['params'],
                          'labels_2d': s.get('labels_2d'),
                          'quality': s.get('quality')} for s in specs]}
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path

# Finer substepping for cells that stalled. The default 20,200,10 is generous for a clean
# solve but leaves little room in the reverse transformation, which is where most failures
# happened.
RETRY_NSUBST = '100,2000,50'

def failed_cells(results_dir=None, specs=None):
    """Names of cells whose monitor file shows they never reached the final load step."""
    results_dir = (config.PROJECT_ROOT / 'ansys_output') if results_dir is None         else pathlib.Path(results_dir)
    specs = select_cells() if specs is None else specs
    return [s['name'] for s in specs
            if not read_convergence(results_dir / f"{s['name']}.mntr")['converged']]

def write_retry(names=None, results_dir=None, nsubst=RETRY_NSUBST, path=None):
    """Regenerate just the failed cells' decks with finer substepping, plus a batch script."""
    names = failed_cells(results_dir) if names is None else names
    specs = [s for s in select_cells() if s['name'] in set(names)]
    build(specs, nsubst=nsubst)
    path = (DECK_DIR / 'run_spikeb_retry.ps1') if path is None else path
    return write_batch_script(specs, path=path), [s['name'] for s in specs]

# The deck runs four load steps, so a finished solve ends at TIME 4.0.
FINAL_TIME = 4.0

def read_convergence(mntr_path, final_time=FINAL_TIME, tol=1e-6):
    """
    Determine if a cell's simulation converged and how many substeps were needed.

    Returns reached_time, converged, retries (substeps needing more than one attempt), and
    max_attempts, the last two being a difficulty signal worth carrying into S6.2, since a
    cell that needed heavy bisection is a lower-confidence data point.
    """
    path = pathlib.Path(mntr_path)
    if not path.exists():
        return {'reached_time': None, 'converged': False, 'retries': 0, 'max_attempts': 0}
    reached, retries, max_attempts = 0.0, 0, 0
    for line in path.read_text(errors='ignore').splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            attempt = int(parts[2])
            total = float(parts[6])
        except ValueError:
            continue            # header or banner
        reached = max(reached, total)
        max_attempts = max(max_attempts, attempt)
        if attempt > 1:
            retries += 1
    return {'reached_time': reached,
            'converged': reached >= final_time - tol,
            'retries': retries,
            'max_attempts': max_attempts}

def collect(specs=None, results_dir=None, mesh=None, out_stem='s6_1_labels', out_dir=None):
    """
    Join each cell's 2D labels with its 3D results into one table.

    Cells whose run didn't finish are kept with 3D fields set to None and `converged`
    False; a topology the 3D tier can't solve is data about the envelope.
    """
    from sim3d import loadsteps, mesh3d

    specs = select_cells() if specs is None else specs
    results_dir = (config.PROJECT_ROOT / 'ansys_output') if results_dir is None \
        else pathlib.Path(results_dir)
    mesh = MESH if mesh is None else mesh

    rows = []
    for spec in specs:
        stem = spec['name']
        row = {'name': stem, 'family': spec['family']}
        row.update(spec.get('labels_2d') or label_2d(spec['cell']))
        status = results_dir / f'{stem}.txt'
        amps = results_dir / f'{stem}_amp.txt'
        conv = read_convergence(results_dir / f'{stem}.mntr')
        row.update(conv)
        if not (status.exists() and amps.exists() and conv['converged']):
            row.update({'converged': False, 'K_radial_3D': None, 'eps_a_max_3D': None, 'A_over_lim_3D': None})
            rows.append(row)
            continue
        summary = loadsteps.summarize(status, n_axial=mesh['n_axial'], n_circ=mesh['n_circ'])
        points, hexes, _ = mesh3d.tube_hex_mesh(
            spec['cell'], n_circ=mesh['n_circ'], n_axial=mesh['n_axial'],
            layers=mesh['layers'], limit=mesh['limit'])
        amp = loadsteps.read_node_amplitudes(amps)
        row.update({
            'converged': True,
            'K_radial_3D': summary['K_radial_3D'],
            'eps_a_max_3D': summary['eps_a_max_3D'],
            'A_over_lim_3D': loadsteps.a_over_lim_3d(amp, points, hexes),
            'eps_etable_over_nodal': summary.get('eps_etable_over_nodal'),
        })
        rows.append(row)

    out_dir = RESULTS_DIR if out_dir is None else pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{out_stem}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    cols = ['name', 'family', 'converged', 'reached_time', 'retries', 'max_attempts',
            'f_metal', 'K_radial_2D', 'K_radial_3D', 'eps_a_max_2D', 'eps_a_max_3D',
            'eps_a_p99_2D', 'A_over_lim_2D', 'A_over_lim_3D', 'eps_etable_over_nodal']
    lines = [','.join(cols)]
    for row in rows:
        lines.append(','.join('' if row.get(c) is None else str(row.get(c, ''))  for c in cols))
    (out_dir / f'{out_stem}.csv').write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    return rows
