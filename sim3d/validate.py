"""
Static checks on generated APDL decks.

This catches processor errors, and it can't catch semantic errors.
"""

import re

import config

# Commands valid only in the processor named. Not exhaustive.
PROCESSOR_ONLY = {
    'POST1': {'FSUM', 'NSORT', 'ESORT', 'ETABLE', 'SADD', 'SET', 'PLNSOL', 'PLESOL',
              'PRNSOL', 'PRESOL', 'NSEL_RESULT', 'SUMTYPE',
              'LCDEF', 'LCWRITE', 'LCASE', 'LCOPER', 'LCZERO', 'LCFILE',
              'FILE'},
    'POST26': {'NSOL', 'ESOL', 'RFORCE', 'PRVAR', 'PLVAR', 'NUMVAR', 'XVAR', 'STORE'},
    'SOLU': {'SOLVE', 'ANTYPE', 'NLGEOM', 'NSUBST', 'AUTOTS', 'CNVTOL', 'OUTRES',
             'LNSRCH', 'TIME', 'DELTIM', 'NEQIT', 'RESCONTROL'},
    'PREP7': {'ET', 'KEYOPT', 'MP', 'TB', 'TBDATA', 'N', 'EN', 'BLOCK', 'VMESH',
              'LESIZE', 'MSHAPE', 'MSHKEY', 'NROTAT', 'EMODIF', 'VSWEEP', 'ESIZE'},
}

PROCESSOR_STARTS = {'/PREP7': 'PREP7', '/SOLU': 'SOLU', '/POST1': 'POST1',
                    '/POST26': 'POST26'}

BEGIN_ONLY = {'/FILNAME', '/CLEAR', '/CWD', '/CONFIG', 'RESUME'}

def check(text):
    """Return a list of problems found in an APDL deck."""
    problems = []
    processor = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split('!')[0].strip()
        if not line:
            continue
        head = line.split(',')[0].strip().upper()

        if head in PROCESSOR_STARTS:
            processor = PROCESSOR_STARTS[head]
            continue
        if head == 'FINISH':
            processor = None
            continue

        if head in BEGIN_ONLY and processor is not None:
            problems.append(
                f'Line {lineno}: {head} is only valid at the Begin level but appears '
                f'in /{processor}.'
            )
            continue

        for owner, commands in PROCESSOR_ONLY.items():
            if head in commands and processor is not None and processor != owner:
                problems.append(
                    f'Line {lineno}: {head} is a /{owner} command but appears in /{processor}'
                )
            elif head in commands and processor is None and owner != 'PREP7':
                problems.append(
                    f'Line {lineno}: {head} is a /{owner} command but no processor is active'
                )
    return problems

def check_file(path):
    return check(path.read_text(encoding='utf-8', errors='ignore'))

def check_all(directory=None):
    """Check every generated deck. Returns {name: [problems]}."""
    directory = (config.PROJECT_ROOT / 'sim3d' / 'decks') if directory is None else directory
    return {p.name: check_file(p) for p in sorted(directory.glob('*.inp'))}

if __name__ == "__main__":
    results = check_all()
    total = 0
    for name, problems in results.items():
        if problems:
            total += len(problems)
            print(f'{name}:')
            for problem in problems:
                print(f'  {problem}')
        else:
            print(f'{name}: ok')
    print(f'\n{total} problem(s).')
