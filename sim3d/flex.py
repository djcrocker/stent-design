"""
S9.4 loading audit: does the axial screen's ranking survive a change of load mode?

The 2D tier is an axial cyclic-strain surrogate. It is plane-stress and structurally can't
represent out-of-plane bending, so the question of whether it ranks designs the way bending
would can only be answered in 3D.

Expensive because:
  Full ring    Bending is not axisymmetric.
  L/D >= 1     At n_axial=1 the ring is 1.57 mm long, L/D 0.26, which measures end effects
               rather than bending.

Affordable because:
  layers=2     s5_3_strain{1..4} measured nodal strain sampling at 1.000000 against the
               closed form at every layer count, and cantilever deflection at 0.9956 with 2
               layers vs 0.9949 with 4. Both quantities this reads are converged at 2.
  Window       The driven end faces are an artificial grip, so strain is read from the
               central half and the end cells are discarded.
"""

import argparse
import json
import pathlib

import numpy as np

import config
from sim3d import apdl, mesh3d

DECK_DIR = config.PROJECT_ROOT / 'sim3d' / 'decks' / 'flex'
RESULTS_DIR = config.PROJECT_ROOT / 'sim3d' / 'results'
STAGE_DIR = config.PROJECT_ROOT / 'sim3d' / 'decks' / 'vm_stage'

# Full ring, L/D 1.05, 2 layers.
MESH = {'n_circ': None, 'n_axial': 4, 'layers': 2, 'limit': 0.20}

# Reject anything Ansys would reject.
MAX_FACE_ANGLE_DEG = 155.0
RELEASE = False

N_SUBSET = 8                    # cells, each run under two load modes
ARMS = ('bend', 'comb')         # bending only, and combined bending + axial

VM_EXE = r'C:\Program Files\ANSYS Inc\v261\ansys\bin\winx64\ANSYS261.exe'
VM_CORES = 4

def stratify(fields, n=N_SUBSET, seed=0):
    """Topology-stratified subset, spread over structure."""
    from screen import onetomany

    desc = onetomany.descriptor(fields)
    return np.sort(onetomany.farthest_first(desc, n, seed=seed))

def build(indices=None, fields=None, directory=None, mesh=None, arms=ARMS):
    """Mesh each selected cell once and write one deck per load arm against it."""
    from geom.cell import UnitCell
    from screen import shortlist

    directory = DECK_DIR if directory is None else pathlib.Path(directory)
    mesh = MESH if mesh is None else mesh
    directory.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields, _ = shortlist.load()
    if indices is None:
        indices = stratify(fields)

    built = []
    for i in indices:
        i = int(i)
        stem = f's9_4_cell{i:02d}'
        mesh_path = directory / f'{stem}_mesh.inp'
        info, q = mesh3d.build_and_write(UnitCell(fields[i]), mesh_path,
                                         n_circ=mesh['n_circ'], n_axial=mesh['n_axial'],
                                         layers=mesh['layers'], limit=mesh['limit'])
        if q['max_face_angle_deg'] > MAX_FACE_ANGLE_DEG or q['n_nonpositive']:
            raise ValueError(f"{stem}: mesh would be rejected by Ansys "
                             f"({q['max_face_angle_deg']:.1f} deg, "
                             f"{q['n_nonpositive']} non-positive)")
        decks = {}
        for arm in arms:
            compression = config.FLEX_AXIAL_AMPLITUDE if arm == 'comb' else 0.0
            out_stem = f'{stem}_{arm}'
            path = directory / f'{out_stem}.inp'
            path.write_text(apdl.flex_cycle(mesh=f'{stem}_mesh', deck_dir='',
                                            out_stem=out_stem, compression=compression,
                                            release=RELEASE),
                            encoding='utf-8')
            decks[arm] = str(path)
        built.append({'pool_index': i, 'stem': stem, 'f_metal': float(fields[i].mean()),
                      'n_nodes': info['n_points'], 'n_hexes': info['n_hexes'],
                      'entities': info['n_points'] + info['n_hexes'],
                      'scaled_jacobian_min': q['scaled_jacobian_min'],
                      'max_face_angle_deg': q['max_face_angle_deg'],
                      'mesh': str(mesh_path), 'decks': decks})

    manifest = {'mesh': {k: v for k, v in mesh.items()},
                'arms': list(arms),
                'bend_radius_mm': config.FLEX_BEND_AMPLITUDE_RADIUS_MM,
                'bend_radius_full_excursion_mm': config.FLEX_BEND_RADIUS_MM,
                'axial_compression': config.FLEX_AXIAL_AMPLITUDE,
                'axial_compression_full_excursion': config.FLEX_AXIAL_COMPRESSION,
                'sample_fraction': apdl.BEND_SAMPLE_FRACTION,
                'release': RELEASE,
                'length_mm': mesh['n_axial'] * config.cell_extent_mm()[1],
                'cells': built}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 's9_4_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

VM_BACKUP = r'\\Client\C$\Users\djcro\Projects\stent-design\ansys_output'

def write_vm_script(manifest, path=None, np_cores=VM_CORES, backup=VM_BACKUP):
    """A PowerShell script that runs every deck on the VM."""

    path = (DECK_DIR / 'run_flex.ps1') if path is None else pathlib.Path(path)
    jobs = [f'{c["stem"]}_{arm}' for c in manifest['cells'] for arm in manifest['arms']]
    L = ['# S9.4 loading audit. Generated by sim3d/flex.py.',
         f'# {len(jobs)} jobs: {len(manifest["cells"])} cells x {len(manifest["arms"])} load arms.',
         f'$exe = "{VM_EXE}"',
         '$dir = $PSScriptRoot',
         'if (-not $dir) { $dir = (Get-Location).Path }',
         f'$backup = "{backup}"' if backup else '$backup = ""',
         'Set-Location $dir',
         'Write-Host "Running in $dir" -ForegroundColor Yellow',
         f'$log = Join-Path $dir ("s9_4_timing_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv")',
         '"job,seconds,exit" | Out-File -FilePath $log -Encoding utf8',
         f'$jobs = @({", ".join(chr(34) + j + chr(34) for j in jobs)})',
         'foreach ($j in $jobs) {',
         '  if (Test-Path (Join-Path $dir "$j.ok")) {',
         '    Write-Host "=== $j  already done, skipping" -ForegroundColor DarkGray',
         '    continue',
         '  }',
         '  Write-Host "=== $j ===" -ForegroundColor Cyan',
         '  $t0 = Get-Date',
         # -o shouldn't collide with the deck's own /OUTPUT,<stem>,txt.
         '  & $exe -b -np ' + str(np_cores) + ' -dir $dir -j $j -i "$j.inp" -o "$j`_run.out"',
         '  $code = $LASTEXITCODE',
         '  $secs = [int]((Get-Date) - $t0).TotalSeconds',
         '  "$j,$secs,$code" | Out-File -FilePath $log -Append -Encoding utf8',
         '  if ($code -eq 0) {',
         '    "exit 0" | Out-File -FilePath (Join-Path $dir "$j.ok") -Encoding utf8',
         '    if ($backup -and (Test-Path $backup)) {',
         '      try {',
         '        Copy-Item (Join-Path $dir "$j.txt"), (Join-Path $dir "$j`_amp.txt"), '
         '(Join-Path $dir "$j`_zloc.txt"), (Join-Path $dir "$j.ok") $backup -Force '
         '-ErrorAction Stop',
         '        Write-Host "    backed up" -ForegroundColor DarkGray',
         '      } catch { Write-Host "    BACKUP FAILED: $_" -ForegroundColor Yellow }',
         '    } else {',
         '      Write-Host "    no backup path - copy results off by hand" '
         '-ForegroundColor Yellow',
         '    }',
         '  } else {',
         '    Remove-Item (Join-Path $dir "$j.ok") -ErrorAction SilentlyContinue',
         '    Write-Host "    FAILED - any results file this job wrote is NOT valid" '
         '-ForegroundColor Red',
         '  }',
         '  Write-Host "    $secs s (exit $code)"',
         '}',
         '$left = @($jobs | Where-Object { -not (Test-Path (Join-Path $dir "$_.ok")) })',
         'Write-Host "Timings in $log"',
         'Write-Host "$($jobs.Count - $left.Count)/$($jobs.Count) complete; '
         're-run this script to retry the rest."']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(L) + chr(10), encoding='utf-8')
    return path

AUDIT_K = 3

def rank_audit(rows, arms, metrics=('A_over_lim', 'eps_a_p99', 'eps_a_max'), k=AUDIT_K, seed=0):
    """
    Does the axial screen's ranking survive the change of load mode?

    The false-negative rate is `1 - top-k retention`. Retention counts how many of bending's 
    best k the axial best k captured, so its complement counts the designs that are good under 
    bending and which axial rejected.
    """
    from sim3d import correlate

    out = {'k': k, 'gate_rho': correlate.GATE_RHO, 'arms': {}}
    for arm in arms:
        done = [r for r in rows if r.get(f'{arm}_converged')]
        per = {'n': len(done), 'n_missing': len(rows) - len(done), 'metrics': {}}
        for m in metrics:
            if len(done) < 3:
                continue
            x = [r[f'axial_{m}'] for r in done]
            y = [r[f'{arm}_{m}'] for r in done]
            rho, p = correlate.spearman(x, y)
            lo, hi = correlate.bootstrap_ci(x, y, seed=seed)
            kk = min(k, len(done))
            ret = correlate.topk_retention(x, y, kk, correlate.HIGHER_IS_BETTER[m])
            per['metrics'][m] = {'rho': rho, 'p': p, 'ci95': [lo, hi], 'n': len(done),
                                 'top_k': kk, 'retention': ret,
                                 'false_negative_rate': 1.0 - ret,
                                 'passes': rho >= correlate.GATE_RHO}
        out['arms'][arm] = per
    return out

def grip_vs_density(rows, arms):
    """
    Is the end-grip artifact correlated with what's being ranked?

    `grip_inflation` is the whole-ring strain peak over the window peak, so it measures how
    far the artificial grip inflates strain outside the measurement window. A roughly
    constant inflation is a shared offset and cancels in a rank correlation. But, one
    that tracks `f_metal` doesn't: it would bias the ranking along the same axis as the
    metric. So this decides how a FAIL gets read.
    """
    from sim3d import correlate

    out = {}
    for arm in arms:
        done = [r for r in rows if r.get(f'{arm}_converged')
                and r.get(f'{arm}_grip_inflation') is not None]
        if len(done) < 3:
            out[arm] = {'n': len(done)}
            continue
        f = np.array([r['f_metal'] for r in done], float)
        g = np.array([r[f'{arm}_grip_inflation'] for r in done], float)
        rho, p = correlate.spearman(f, g)
        slope, intercept = np.polyfit(f, g, 1)
        out[arm] = {'n': len(done), 'rho': rho, 'p': p,
                    'slope_per_f_metal': float(slope), 'intercept': float(intercept),
                    'min': float(g.min()), 'max': float(g.max()),
                    'spread': float(g.max() - g.min())}
    return out

def analyze(manifest_path=None, results_dir=None, out_stem='s9_4_loading_audit'):
    """Rank correlation of the axial screen against each 3D load arm, and the false-negative rate."""
    from geom.cell import UnitCell
    from screen import shortlist
    from sim3d import loadsteps

    results_dir = (config.PROJECT_ROOT / 'ansys_output') if results_dir is None \
        else pathlib.Path(results_dir)
    manifest_path = (RESULTS_DIR / 's9_4_manifest.json') if manifest_path is None \
        else pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fields, summary = shortlist.load()
    mesh = manifest['mesh']
    axial = summary['shortlist']

    rows = []
    for c in manifest['cells']:
        i = c['pool_index']
        points, hexes, _ = mesh3d.tube_hex_mesh(UnitCell(fields[i]), n_circ=mesh['n_circ'],
                                                n_axial=mesh['n_axial'],
                                                layers=mesh['layers'], limit=mesh['limit'])
        a = axial[i]
        if abs(a['f_metal'] - c['f_metal']) > 1e-9:
            raise ValueError(f"{c['stem']} joined to the wrong axial row: f_metal "
                             f"{c['f_metal']:.6f} vs shortlist[{i}] {a['f_metal']:.6f}")
        row = {'pool_index': i, 'stem': c['stem'], 'f_metal': c['f_metal'],
               'axial_A_over_lim': a['A_over_lim'], 'axial_eps_a_max': a['eps_a_max'],
               'axial_eps_a_p99': a['eps_a_p99']}
        for arm in manifest['arms']:
            stem = f'{c["stem"]}_{arm}'
            status = results_dir / f'{stem}.txt'
            amp = results_dir / f'{stem}_amp.txt'
            if not ((results_dir / f'{stem}.ok').exists()
                    and status.exists() and amp.exists()):
                row[f'{arm}_converged'] = False
                continue
            zloc = results_dir / f'{stem}_zloc.txt'
            # halve == release: with a release state the dump is a range, without one the
            # deck imposed the amplitude curvature and the dump is already an amplitude.
            s = loadsteps.flex_scalars(status, amp, points, hexes,
                                       zloc_path=zloc if zloc.exists() else None,
                                       fraction=manifest['sample_fraction'],
                                       halve=manifest.get('release', False))
            row[f'{arm}_converged'] = True
            row[f'{arm}_eps_a_max'] = s['eps_a_max_bend']
            row[f'{arm}_eps_a_p99'] = s['eps_a_p99_bend']
            row[f'{arm}_eps_a_p95'] = s['eps_a_p95_bend']
            row[f'{arm}_over_eps_L'] = s['over_eps_L_bend']
            row[f'{arm}_A_over_lim'] = s['A_over_lim_bend']
            row[f'{arm}_grip_inflation'] = s['grip_inflation']
        rows.append(row)

    report = {'source': str(manifest_path), 'n_cells': len(rows),
              'bend_radius_mm': manifest['bend_radius_mm'],
              'axial_compression': manifest['axial_compression'],
              'rows': rows,
              'rank_audit': rank_audit(rows, manifest['arms']),
              'grip_vs_f_metal': grip_vs_density(rows, manifest['arms'])}
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(report, indent=2),
                                                  encoding='utf-8')
    return report

def stage(manifest, stage_dir=None):
    """Copy the meshes, decks, and runner into vm_stage for transfer to the VM."""
    import shutil

    stage_dir = (STAGE_DIR / 'flex') if stage_dir is None else pathlib.Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for c in manifest['cells']:
        for src in [c['mesh'], *c['decks'].values()]:
            dst = stage_dir / pathlib.Path(src).name
            shutil.copy2(src, dst)
            copied.append(dst)
    script = write_vm_script(manifest)
    shutil.copy2(script, stage_dir / script.name)
    copied.append(stage_dir / script.name)
    return copied

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='S9.4 loading audit')
    ap.add_argument('phase', choices=('build', 'analyze'))
    ap.add_argument('--n', type=int, default=N_SUBSET)
    ap.add_argument('--one', action='store_true',
                    help='the densest shortlist cell only, to time a single solve')
    ap.add_argument('--no-stage', action='store_true')
    args = ap.parse_args()

    if args.phase == 'analyze':
        rep = analyze()
        print(f"{rep['n_cells']} cells, bend radius {rep['bend_radius_mm']:.0f} mm")
        for r in rep['rows']:
            parts = [f"{r['stem']}  f_metal {r['f_metal']:.3f}"]
            for arm in ARMS:
                if r.get(f'{arm}_converged'):
                    parts.append(f"{arm} eps_a {r[f'{arm}_eps_a_max']:.5f} "
                                 f"A/lim {r[f'{arm}_A_over_lim']:.3f}")
                else:
                    parts.append(f'{arm} MISSING')
            print('  ' + '  |  '.join(parts))

        ra = rep['rank_audit']
        print()
        print(f"Rank audit, axial screen vs each 3D arm (gate rho >= {ra['gate_rho']}, "
              f"top-{ra['k']}):")
        for arm, per in ra['arms'].items():
            miss = f", {per['n_missing']} MISSING" if per['n_missing'] else ''
            print(f"  {arm}  n={per['n']}{miss}")
            for m, s in per['metrics'].items():
                print(f"    {m:11} rho {s['rho']:+.3f} "
                      f"CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] p {s['p']:.3f}  "
                      f"retention {s['retention']:.2f}  "
                      f"false-neg {s['false_negative_rate']:.2f}  "
                      f"{'PASS' if s['passes'] else 'FAIL'}")

        print()
        print("Grip inflation vs f_metal (constant = cancels in a ranking; "
              "sloped = a FAIL may be the BC):")
        for arm, s in rep['grip_vs_f_metal'].items():
            if 'rho' not in s:
                print(f"  {arm}  n={s['n']}, too few to regress")
                continue
            print(f"  {arm}  {s['min']:.3f}-{s['max']:.3f} (spread {s['spread']:.3f})  "
                  f"rho {s['rho']:+.3f} p {s['p']:.3f}  "
                  f"slope {s['slope_per_f_metal']:+.3f} per unit f_metal")
        raise SystemExit(0)

    from screen import shortlist as _sl

    fields, _ = _sl.load()
    if args.one:
        # Densest, so the measured wall clock is an upper bound for budgeting the rest.
        indices = [int(np.array([f.mean() for f in fields]).argmax())]
    else:
        indices = stratify(fields, args.n)

    man = build(indices=indices, fields=fields)
    L = man['length_mm']
    print(f"mesh: full ring, n_axial={man['mesh']['n_axial']}, "
          f"layers={man['mesh']['layers']}, L={L:.2f} mm, L/D={L / config.D_DEPLOYED_MM:.2f}")
    print(f"load: bend radius {man['bend_radius_mm']:.0f} mm "
          f"(curvature {1 / man['bend_radius_mm']:.5f} 1/mm), "
          f"axial compression {man['axial_compression']} on the combined arm")
    print(f"strain window: central {man['sample_fraction']:.0%} of the length\n")
    print(f"{'stem':16} {'f_metal':>8} {'nodes':>9} {'hexes':>9} {'entities':>10} "
          f"{'sJac':>6} {'angle':>7}")
    for c in man['cells']:
        print(f"{c['stem']:16} {c['f_metal']:8.3f} {c['n_nodes']:9,} {c['n_hexes']:9,} "
              f"{c['entities']:10,} {c['scaled_jacobian_min']:6.3f} "
              f"{c['max_face_angle_deg']:7.1f}")
    print(f"\n{len(man['cells'])} cells x {len(man['arms'])} arms = "
          f"{len(man['cells']) * len(man['arms'])} solves")
    if not args.no_stage:
        copied = stage(man)
        print(f"staged {len(copied)} files in {copied[0].parent}")
