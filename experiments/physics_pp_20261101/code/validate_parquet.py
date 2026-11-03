#!/usr/bin/env python3
"""Round-trip check of a post-processed Parquet against its per-episode outputs and the source file.

Every episode is parsed with the scorer's own helpers (row grids, packing tails, orthonormal rotations,
positive scale, integer faces). Translations, poses, scale (and rotation, when the episode output carries it)
must equal the solver outputs; rotations otherwise, mesh, identity and scales must equal the source file. Own ACC-O / ACC-H are recomputed from the file.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--source', required=True)
    ap.add_argument('--episodes-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res, acc_o, acc_h = {}, [], []
    for e in range(30):
        new, old = pp.load_episode(a.parquet, e), pp.load_episode(a.source, e)
        d = np.load(Path(a.episodes_dir) / f'episode_{e:06d}.npz')
        assert np.array_equal(new['frames'], d['frames'])
        assert np.allclose(new['t'], d['object_translation'], atol=0, rtol=0), e
        assert np.allclose(new['pose'], d['mhr_pose'], atol=0, rtol=0), e
        if 'object_scale' in d.files:
            assert new['s'] == float(d['object_scale']), e
        if 'object_rotation' in d.files:                                 # rot_smooth.py output (SVD-reprojected on read)
            assert np.allclose(new['R'], d['object_rotation'], atol=1e-9, rtol=0), e
        for k in (('verts', 'faces', 'scales', 'shape') if 'object_rotation' in d.files else ('R', 'verts', 'faces', 'scales', 'shape')):
            assert np.array_equal(new[k], old[k]), (e, k)
        _, j = pp.decode(new['pose'], new['scales'], new['shape'])
        acc_o.append(pp.own_acc(new['t'], new['frames'])); acc_h.append(pp.own_acc(j[:, pp.BODY_J], new['frames']))
        res[e] = {'own_acc_o': acc_o[-1], 'own_acc_h': acc_h[-1], 'scale': new['s']}
    out = {'status': 'PASS_ROUND_TRIP', 'parquet_sha256': pp.sha(a.parquet), 'source_sha256': pp.sha(a.source),
           'own_acc_o_mean': float(np.mean(acc_o)), 'own_acc_h_mean': float(np.mean(acc_h)), 'episodes': res,
           'validator_sha256': pp.sha(__file__)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'episodes'}))


if __name__ == '__main__':
    main()
