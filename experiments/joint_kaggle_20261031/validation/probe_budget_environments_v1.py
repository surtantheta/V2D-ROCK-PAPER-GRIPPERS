#!/usr/bin/env python3
"""Actual all30 mesh-budget cross-environment probe; no models or GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
KIT = Path('/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit')


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_oriented(triangles):
    result = []
    for tri in triangles:
        first = min(range(3), key=lambda i: tuple(tri[i]))
        result.append(np.roll(tri, -first, axis=0).reshape(9))
    if not result:
        return np.empty((0, 9), dtype=float)
    values = np.asarray(result)
    return values[np.lexsort(tuple(values[:, i] for i in reversed(range(9))))]


def main():
    assert os.environ.get('PBS_JOBID'), 'PBS required'
    p = argparse.ArgumentParser()
    p.add_argument('--environment', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.environment == 'cari4d_grail':
        sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
    sys.path.insert(0, str(KIT))
    import trimesh
    import fast_simplification
    from v2dlb.mesh_budget import budget_mesh
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    for ep in range(30):
        path = BASE / f'episode_{ep:06d}_object.ply'
        v, f = budget_mesh(path, 4096, 4096)
        assert v.dtype == np.float64 and f.dtype == np.int64
        tri = v[f]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        active = tri[area > 0]
        np.savez_compressed(a.out / f'episode_{ep:06d}.npz', vertices=v, faces=f)
        records.append(dict(episode=ep, vertices_sha256=hash_array(v), faces_sha256=hash_array(f),
                            active_triangles=int(len(active)), oriented_surface_sha256=hash_array(canonical_oriented(active)),
                            reversed_oriented_surface_sha256=hash_array(canonical_oriented(active[:, [0, 2, 1]]))))
    j = dict(status='PASS_BUDGET_ARRAYS_PRODUCED', environment=a.environment, cases=30, records=records,
             versions=dict(numpy=np.__version__, trimesh=trimesh.__version__,
                           fast_simplification=getattr(fast_simplification, '__version__', 'unknown')),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             model_inference=False, GT_loaded=False, PBS_JOBID=os.environ['PBS_JOBID'])
    (a.out / 'COMPLETE.json').write_text(json.dumps(j, indent=2) + '\n')
    print(json.dumps({k: v for k, v in j.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
