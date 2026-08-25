"""
Browse the figures and open one.

Usage:
    python figures/show.py            # interactive
    python figures/show.py --list     # print the catalog and exit
    python figures/show.py s8_traj    # open the best match directly
"""

import argparse
import os
import re
import subprocess
import sys

import config

DEV = config.FIG_DEV_DIR
SUFFIXES = ('.png', '.gif', '.html')

# stem -> (step, title, what it shows, how to rebuild it)
CATALOG = {
    's0_2_sma_spike': ('S0.2', 'Nitinol hysteresis loop',
                       'The Ansys superelastic material card.',
                       'python figures/scripts/s0_2_sma_spike.py'),
    's2_1_cell_and_tiling': ('S2.1', 'Unit cell and 3x3 tiling',
                             'One cell beside its own tiling.',
                             'python figures/scripts/s2_1_cell_and_tiling.py'),
    's2_2_periodicity': ('S2.2', 'Shift equivariance',
                         'Morphology on the torus.',
                         'python figures/scripts/s2_2_periodicity.py'),
    's2_3_validity': ('S2.3', 'Validity failures',
                      'Each way a cell can be invalid, and what the checker says about it.',
                      'python figures/scripts/s2_3_validity.py'),
    's2_4_cleanup': ('S2.4', 'Cleanup pipeline',
                     'Broken cell to repaired cell, with the change fraction that repair cost.',
                     'python figures/scripts/s2_4_cleanup.py'),
    's2_5_tube': ('S2.5', 'Tube mesh (single design)',
                  'The handmade diamond wrapped to a closed 6 mm ring.',
                  'python figures/scripts/s2_5_tube.py'),
    's2_5_tube_gallery': ('S2.5', 'Tube gallery (four designs)',
                          'The baseline next to the stiffest, most fatigue-resistant, and most '
                          'balanced designs on the S9.2 front.',
                          'python figures/scripts/s2_5_tube_gallery.py'),
    's3_1_parametric': ('S3.1', 'Parametric crown family',
                        'The hand-built family the dataset sampler was seeded from.',
                        'python figures/scripts/s3_1_parametric.py'),
    'reference_cell': ('S3.2', 'Reference cell',
                       'The single design every comparison is based on.',
                       'python figures/scripts/s3_2_reference.py'),
    's4_1_stiffness': ('S4.1', 'Homogenized stiffness',
                       'Computational homogenization against its closed-form check.',
                       'python figures/scripts/s4_1_stiffness.py'),
    's4_2_fatigue': ('S4.2', 'Strain amplitude field',
                     'Where strain concentrates under limb flexion.',
                     'python figures/scripts/s4_2_fatigue.py'),
    's4_4_throughput': ('S4.4', 'Labeling throughput',
                        'Cost per labeled cell, which is what made a 50,000-cell dataset possible.',
                        'python figures/scripts/s4_4_throughput.py'),
    's5_1_contour': ('S5.1', 'Contour mesh',
                     'Marching-squares boundary replacing the stair-stepped voxel edge.',
                     'python figures/scripts/s5_1_contour.py'),
    's5_3_loadsteps': ('S5.3', '3D load steps',
                       'The Ansys bending run, step by step.',
                       'python figures/scripts/s5_3_loadsteps.py'),
    's6_2_correlation': ('S6.2', '2D vs 3D rank correlation',
                         'Does the cheap 2D screen order designs the way 3D does?',
                         'python figures/scripts/s6_2_correlation.py'),
    's6_2_contact_sheet': ('S6.2', 'Spike B contact sheet',
                           'Every cell in the correlation study, with its 2D and 3D numbers.',
                           'python figures/scripts/s6_2_contact_sheet.py'),
    's6_2_coverage': ('S6.2', 'Spike B coverage',
                      'Where the 60 chosen cells sit in the space they are meant to represent.',
                      'python figures/scripts/s6_2_coverage.py'),
    's6_2_residuals': ('S6.2', 'Spike B residuals',
                       '2D-3D disagreement against solver difficulty.',
                       'python figures/scripts/s6_2_residuals.py'),
    's6_4_resolution': ('S6.4', '64 vs 128 resolution',
                        'Refinement moves the values hugely and the ranking barely changes.',
                        'python figures/scripts/s6_4_resolution.py'),
    's7_1_coverage': ('S7.1', 'Dataset coverage',
                      'What the 50,000-cell training set actually spans.',
                      'python figures/scripts/s7_1_coverage.py'),
    's8_1_training': ('S8.1', 'Training curves',
                      'Train and validation loss over 60 epochs.',
                      'python figures/scripts/s8_1_training.py'),
    's8_2_samples': ('S8.2', 'Unconditional samples',
                     'What the model produces with the conditioning switched off.',
                     'python figures/scripts/s8_2_samples.py'),
    's8_3_valid_rate': ('S8.3', 'Valid rate over training',
                        'Flat from epoch 10 - validity is learned early, fidelity is what the rest bought.',
                        'python figures/scripts/s8_3_valid_rate.py sample && '
                        'python figures/scripts/s8_3_valid_rate.py plot'),
    's8_traj': ('S8', 'Denoising trajectory',
                'Noise to topology, every DDIM step. The design is settled by step 7 of 50.',
                'python figures/scripts/s8_traj.py sample && '
                'python figures/scripts/s8_traj.py plot'),
    's9_3_onetomany': ('S9.3', 'One-to-many gallery',
                       'Several distinct topologies meeting the same target.',
                       'python figures/scripts/s9_3_onetomany.py'),
    'explorer': ('S9.3', 'Strut Topology Explorer',
                 'Interactive. 50,000 training cells, 650 generated designs, the shortlist, '
                 'and the distinct exemplars per target, all on one pair of axes.',
                 'python figures/scripts/explorer_data.py && '
                 'python figures/scripts/explorer_page.py'),
}

def ansys_meta(stem):
    """
    Nine Ansys captures share one description but not one identity.

    The stage and the run live in the filename (`ansys_output_5.1_fullring-A`), so read
    them back out rather than listing nine near-identical rows called "Ansys screenshot".
    """
    rest = stem[len('ansys_output_'):]
    m = re.match(r'(\d+(?:\.\d+)?)_?(.*)', rest)
    stage = f'S{m.group(1)}' if m else 'Ansys'
    what = (m.group(2) if m else rest).replace('_', ' ').replace('-', ' ') or 'capture'
    return (stage, f'Ansys: {what}',
            'Captured from the Ansys GUI, not generated by this repo.', None)

def stage_key(stage):
    """Sort S2.5 before S10.1 - a plain string sort puts S10 in the middle of S1."""
    digits = ''.join(c if c.isdigit() or c == '.' else ' ' for c in stage).split()
    return tuple(int(d) for d in digits[0].split('.')) if digits else (99,)

def catalog():
    """Everything in figures/dev, annotated where we have an annotation for it."""
    found = {}
    for path in sorted(DEV.glob('*')):
        if path.suffix.lower() not in SUFFIXES:
            continue
        stem = path.stem
        meta = CATALOG.get(stem) or (ansys_meta(stem) if stem.startswith('ansys_output') else None)
        if meta is None:
            meta = ('--', stem.replace('_', ' '), 'No description recorded yet.', None)
        found.setdefault(stem, {'stem': stem, 'stage': meta[0], 'title': meta[1],
                                'blurb': meta[2], 'build': meta[3], 'files': []})
        found[stem]['files'].append(path)

    for stem, meta in CATALOG.items():
        if stem not in found:
            found[stem] = {'stem': stem, 'stage': meta[0], 'title': meta[1],
                           'blurb': meta[2], 'build': meta[3], 'files': []}

    rows = list(found.values())
    rows.sort(key=lambda r: (stage_key(r['stage']), r['stem']))
    return rows

def open_file(path):
    if sys.platform == 'win32':
        os.startfile(str(path))                                   # noqa: S606
    elif sys.platform == 'darwin':
        subprocess.run(['open', str(path)], check=False)
    else:
        subprocess.run(['xdg-open', str(path)], check=False)

def open_row(row):
    """Open every render for one entry."""
    if not row['files']:
        print(f"\n  Not rendered yet. Build it with:\n    {row['build'] or '(no build command)'}\n")
        return False
    # HTML first: it is the interactive one, and it should land in front.
    for path in sorted(row['files'], key=lambda p: p.suffix.lower() != '.html'):
        print(f'  opening {path.name}')
        open_file(path)
    return True

def build_row(row):
    if not row['build']:
        print('\n  Nothing to build - this one is not generated by the repo.\n')
        return
    print(f"\n  $ {row['build']}\n")
    for part in row['build'].split('&&'):
        cmd = part.strip()
        if not cmd:
            continue
        if subprocess.run(cmd, shell=True, cwd=str(config.PROJECT_ROOT)).returncode:
            print('  build failed\n')
            return
    print('  done\n')

# RENDERING #

BOLD, DIM, OFF = '\033[1m', '\033[2m', '\033[0m'
CYAN, YELLOW, GREY = '\033[36m', '\033[33m', '\033[90m'

def size_of(row):
    total = sum(p.stat().st_size for p in row['files'])
    if not total:
        return ''
    return f'{total / 1e6:.1f} MB' if total >= 1e6 else f'{total / 1e3:.0f} KB'

def draw(rows, cur):
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print(f'{BOLD}  Exploratory figures{OFF}  {GREY}figures/dev{OFF}')
    print(f'{GREY}  up/down move    enter open    b build    q quit{OFF}\n')
    for i, r in enumerate(rows):
        here = i == cur
        kinds = ' + '.join(sorted({p.suffix.lstrip('.') for p in r['files']}))
        mark = f'{CYAN}>{OFF}' if here else ' '
        stage = f"{r['stage']:>6}"
        title = r['title']
        if not r['files']:
            line = f"{GREY}{stage}  {title:<34}  not rendered{OFF}"
        else:
            tag = f'{YELLOW}{kinds}{OFF}' if 'html' in kinds else f'{GREY}{kinds}{OFF}'
            line = f"{stage}  {BOLD if here else ''}{title:<34}{OFF if here else ''}  {tag} {GREY}{size_of(r)}{OFF}"
        print(f' {mark} {line}')
    r = rows[cur]
    print(f'\n{GREY}  ' + '-' * 72 + OFF)
    print(f"  {r['blurb']}")
    if r['files']:
        print(f"{GREY}  " + '  '.join(p.name for p in sorted(r['files'])) + OFF)
    elif r['build']:
        print(f"{GREY}  build:  {r['build']}{OFF}")

def interactive(rows):
    try:
        import msvcrt
    except ImportError:
        return fallback(rows)

    cur = 0
    while True:
        draw(rows, cur)
        key = msvcrt.getch()
        if key in (b'\x00', b'\xe0'):                  # Arrow keys arrive as two bytes
            arrow = msvcrt.getch()
            if arrow == b'H':
                cur = (cur - 1) % len(rows)
            elif arrow == b'P':
                cur = (cur + 1) % len(rows)
            continue
        if key in (b'q', b'\x1b', b'\x03'):
            os.system('cls' if sys.platform == 'win32' else 'clear')
            return
        if key in (b'\r', b'\n'):
            print()
            open_row(rows[cur])
            msvcrt.getch()
        elif key in (b'b', b'B'):
            build_row(rows[cur])
            rows[:] = catalog()
            cur = min(cur, len(rows) - 1)
            input('  enter to continue ')
        elif key in (b'k',):
            cur = (cur - 1) % len(rows)
        elif key in (b'j',):
            cur = (cur + 1) % len(rows)

def fallback(rows):
    """Numbered prompt."""
    show_list(rows)
    while True:
        try:
            raw = input('\n  number to open, q to quit: ').strip()
        except EOFError:
            return
        if raw.lower() in ('q', ''):
            return
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            open_row(rows[int(raw) - 1])
        else:
            print('  not a listed number')

def show_list(rows):
    for i, r in enumerate(rows, 1):
        kinds = ' + '.join(sorted({p.suffix.lstrip('.') for p in r['files']})) or 'not rendered'
        print(f"  {i:>2}. {r['stage']:>6}  {r['title']:<36} {kinds}")

def main():
    ap = argparse.ArgumentParser(description='Browse figures/dev')
    ap.add_argument('match', nargs='?', help='open the first entry matching this text')
    ap.add_argument('--list', action='store_true', help='print the catalog and exit')
    args = ap.parse_args()

    rows = catalog()
    if args.list:
        show_list(rows)
        return
    if args.match:
        m = args.match.lower()
        hits = [r for r in rows if m in r['stem'].lower() or m in r['title'].lower()]
        if not hits:
            print(f'  nothing matches {args.match!r}')
            show_list(rows)
            return
        open_row(hits[0])
        return
    interactive(rows)

if __name__ == "__main__":
    main()
