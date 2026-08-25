"""
Assemble the Strut Topology Explorer page: template + inlined payload.

Run `explorer_data.py` first.

Usage: python figures/scripts/explorer_page.py
"""

import config

TEMPLATE = config.PROJECT_ROOT / 'figures' / 'templates' / 'explorer.html'
PAYLOAD = config.PROJECT_ROOT / 'figures' / 'dev' / 'explorer_data.json'
OUT = config.PROJECT_ROOT / 'figures' / 'dev' / 'explorer.html'

MARKER = '/*__PAYLOAD__*/'

def main():
    html = TEMPLATE.read_text(encoding='utf-8')
    if MARKER not in html:
        raise SystemExit(f'{TEMPLATE.name} has no {MARKER} placeholder')
    data = PAYLOAD.read_text(encoding='utf-8')

    if '</script' in data.lower():
        raise SystemExit('payload contains a script-closing sequence')

    OUT.write_text(html.replace(MARKER, data), encoding='utf-8')
    mb = OUT.stat().st_size / 1e6
    print(f'Wrote {OUT}  ({mb:.2f} MB)')
    if mb > 15.0:
        print('  WARNING: stay under 16 MB')

if __name__ == "__main__":
    main()
