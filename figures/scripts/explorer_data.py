"""
Pack everything the Design Explorer page needs into one JSON.

The explorer merges two things: the S7 training dataset (what the model was shown) and 
the S9.1/9.3 generated pool (what it produced when asked for specific targets). They 
belong on the same axes so they go together.

Usage: python figures/scripts/explorer_data.py
"""

import base64
import json

import numpy as np

import config

OUT = config.PROJECT_ROOT / 'figures' / 'dev' / 'explorer_data.json'
SHORTLIST = config.PROJECT_ROOT / 'screen' / 'results' / 's9_2_shortlist.json'
ONETOMANY = config.PROJECT_ROOT / 'screen' / 'results' / 's9_3_onetomany.json'

N_THUMBS = 900
METRICS = ('K_radial', 'A_over_lim', 'f_metal', 'eps_a_max')

def b64f32(a):
    return base64.b64encode(np.asarray(a, dtype='<f4').tobytes()).decode('ascii')

def b64u8(a):
    return base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode('ascii')

def b64bits(fields):
    """Bit-pack a stack of boolean 64x64 fields: 512 bytes each, not 4096."""
    return base64.b64encode(np.packbits(np.asarray(fields, bool), axis=None).tobytes()).decode('ascii')

def dataset_block():
    from diffusion import dataset

    arrs, frame = dataset.load()
    print(f'dataset: {len(frame)} cells')

    labels = [{k: float(frame[k].iloc[i]) for k in dataset.Y_KEYS} for i in range(len(frame))]
    thumb_idx = np.sort(dataset.coverage_subsample(labels, N_THUMBS))
    print(f'  {len(thumb_idx)} thumbnails, spread round-robin over the y grid')

    splits = frame['split'].to_numpy()
    order = {'train': 0, 'val': 1, 'test': 2}
    block = {
        'n': int(len(frame)),
        'split': b64u8([order.get(s, 0) for s in splits]),
        'split_names': ['train', 'val', 'test'],
        'thumb_index': b64f32(thumb_idx),
        'thumb_bits': b64bits(arrs[thumb_idx]),
    }
    for k in METRICS:
        block[k] = b64f32(frame[k].to_numpy())
    return block

def generated_block():
    from diffusion import generate

    fields, which, meta = generate.load_samples()
    rows, _ = generate.cached_labels(fields, 's9_1_generated')
    targets = meta['targets']
    print(f'generated: {len(fields)} samples over {len(targets)} targets')

    valid = np.array([r is not None for r in rows], dtype=bool)
    block = {
        'n': int(len(fields)),
        'guidance': meta['guidance'],
        'n_per_target': meta['n_per_target'],
        'steps': meta['steps'],
        'which': b64u8(np.asarray(which, dtype=np.uint8)),
        'valid': b64u8(valid),
        'bits': b64bits(fields),
        'targets': [{
            'name': t['name'],
            'asked': t['asked'],
            'target': t['target'],
            'support': t['support'],
            'well_supported': t['well_supported'],
        } for t in targets],
    }

    for k in METRICS:
        v = np.array([np.nan if r is None else r.get(k, np.nan) for r in rows], dtype='<f4')
        block[k] = b64f32(np.nan_to_num(v, nan=0.0))
    return block

def screen_block():
    out = {}
    if SHORTLIST.exists():
        rep = json.loads(SHORTLIST.read_text(encoding='utf-8'))
        out['shortlist'] = {
            'k': rep['k'],
            'pool_sampled': rep['pool_sampled'],
            'pool_valid': rep['pool_valid'],
            'pool_distinct': rep['pool_distinct'],
            'layers_used': rep['layers_used'],
            'rows': [{'slot': r['slot'], 'pool_index': r['pool_index'], 'layer': r['layer'],
                      'target_name': r['target_name'],
                      **{k: r[k] for k in METRICS if k in r}}
                     for r in rep['shortlist']],
        }
        print(f"shortlist: {len(out['shortlist']['rows'])} designs")
    if ONETOMANY.exists():
        rep = json.loads(ONETOMANY.read_text(encoding='utf-8'))
        out['onetomany'] = {
            'shift_floor_max': rep['shift_floor_max'],
            'cross_target_mean': rep['cross_target_mean'],
            'per_target': [{
                'name': t['name'],
                'n_valid': t['n_valid'],
                'within_mean': t['within_mean'],
                'ratio_to_cross_target': t['ratio_to_cross_target'],
                'fraction_above_floor': t['fraction_above_floor'],
                'y_spread': t['y_spread'],
                'exemplar_pool_indices': t['exemplar_pool_indices'][:8],
            } for t in rep['per_target'] if 'note' not in t],
        }
        print(f"one-to-many: {len(out['onetomany']['per_target'])} targets")
    return out

def main():
    payload = {
        'grid_n': config.GRID_N,
        'limits': {
            'f_metal_min': config.F_METAL_MIN,
            'f_metal_max': config.F_METAL_MAX,
            'eps_a_lim': config.EPS_A_LIM,
            'min_feature_mm': config.MIN_FEATURE_MM,
        },
        'metrics': list(METRICS),
        # Enough to wrap a cell onto its tube in the browser. Same conventions as
        # geom/tube.py: array is [axial, circ], the cell tiles N_CIRC times around,
        # and metal sits outside the lumen radius.
        'tube': {
            'n_circ': config.N_CIRC,
            'radius_mm': config.D_DEPLOYED_MM / 2.0,
            'thickness_mm': config.STRUT_THICKNESS_MM,
            'cell_axial_mm': config.cell_extent_mm()[1],
        },
        'dataset': dataset_block(),
        'generated': generated_block(),
        **screen_block(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    print(f'Wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)')

if __name__ == "__main__":
    main()
