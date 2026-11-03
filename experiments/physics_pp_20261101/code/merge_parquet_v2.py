#!/usr/bin/env python3
"""Write a submission Parquet with post-processed object translations and MHR poses.

Starts from the input Parquet and replaces only the `object_translation` and `mhr_pose` rows with the
per-episode outputs of physics_pp_v2.py, human_smooth.py or hand_clear.py; every other row and the row order are kept. Verifies that all
unchanged rows are identical, all values are finite, and every episode was replaced. Also summarizes
the per-episode reports.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

KIT = Path('/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit')
_spec = importlib.util.spec_from_file_location('v2d_scorer', KIT / 'metric_code/track_1/PEN.py')
sc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sc)
ARRAY_NAMES = list(sc._ARRAYS)
A_T, A_POSE, A_S = (ARRAY_NAMES.index(n) for n in ('object_translation', 'mhr_pose', 'object_scale'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episodes-dir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--commit-url', default=None)
    a = ap.parse_args()
    src = pd.read_parquet(a.parquet)
    df = src.copy()
    p = df['row_id'].str.split('_', expand=True)
    ep, fr, ar, pt = (p[k].astype(int).to_numpy() for k in (1, 2, 3, 4))
    xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float64).copy()
    reports, replaced = [], np.zeros(len(df), bool)
    for e in sorted(np.unique(ep)):
        d = np.load(Path(a.episodes_dir) / f'episode_{e:06d}.npz')
        reports.append(json.loads((Path(a.episodes_dir) / f'episode_{e:06d}.json').read_text()))
        fpos = {int(f): i for i, f in enumerate(d['frames'])}
        rows = np.flatnonzero((ep == e) & (ar == A_T))
        assert len(rows) == len(fpos), (e, len(rows), len(fpos))
        xyz[rows] = d['object_translation'][[fpos[int(f)] for f in fr[rows]]]
        replaced[rows] = True
        rows = np.flatnonzero((ep == e) & (ar == A_POSE))
        packed = np.zeros((len(d['frames']), 46 * 3)); packed[:, :136] = d['mhr_pose']
        packed = packed.reshape(len(d['frames']), 46, 3)
        xyz[rows] = packed[[fpos[int(f)] for f in fr[rows]], pt[rows]]
        replaced[rows] = True
        if 'object_scale' in d.files:
            rows = np.flatnonzero((ep == e) & (ar == A_S))
            assert len(rows) == 1
            xyz[rows] = [float(d['object_scale']), 0.0, 0.0]
            replaced[rows] = True
    assert np.isfinite(xyz).all()
    df[['x', 'y', 'z']] = xyz
    if a.commit_url:
        df['code_commit_url'] = a.commit_url
    same = (src[['x', 'y', 'z']].to_numpy() == xyz).all(1)
    assert same[~replaced].all(), 'a row outside the replaced arrays changed'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    keys = ['own_acc_o_before', 'own_acc_o_after', 'own_acc_h_before', 'own_acc_h_after', 'own_acc_o',
            'translation_change_cm_mean', 'human_mean_vertex_change_cm', 'fit_residual_mm_mean',
            'joint_change_cm_mean', 'scored_joint_max_change_m']
    summary = {k: float(np.mean([r[k] for r in reports])) for k in keys if all(k in r for r in reports)}
    for k in ('translation_change_cm_max', 'fit_residual_mm_max', 'joint_change_cm_max', 'human_max_vertex_change_cm'):
        if all(k in r for r in reports):
            summary[k] = float(max(r[k] for r in reports))
    summary['pen_proxy_before_cm'] = float(np.mean([r['pen_before'][0] for r in reports]))
    summary['pen_proxy_after_cm'] = float(np.mean([r['pen_after'][0] for r in reports]))
    summary['pen_frames_after'] = int(sum(r['pen_after'][2] for r in reports))
    summary['pen_max_depth_after_cm'] = float(max(r['pen_after'][1] for r in reports))
    summary['human_first_frames_exact'] = all(r.get('human_first_frame_exact', True) for r in reports)
    summary.update({'episodes': len(reports), 'rows': int(len(df)), 'rows_changed': int((~same).sum()),
                    'input_sha256': sha(a.parquet), 'output_sha256': sha(out), 'source_sha256': sha(__file__),
                    'commit_url': str(df['code_commit_url'].iloc[0])})
    (out.with_suffix('.summary.json')).write_text(json.dumps({'summary': summary, 'episodes': reports}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
