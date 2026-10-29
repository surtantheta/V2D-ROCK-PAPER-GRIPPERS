#!/usr/bin/env python3
"""Pack/check two existing Track 1 candidates, then replace ONLY their commit URL.

No inference, GitHub writes, Kaggle authentication or submissions occur here.
Run with /home/tanuj/.conda/envs/v2d/bin/python.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT = Path('/home/tanuj/V2D_challenge')
HERE = Path(__file__).resolve().parent
KIT = PROJECT / 'data/assets/submission_kit/v2d_submission_kit'
SAMPLE = KIT / 'data/track_1_sample_submission.parquet'
CANDIDATES = {
    'main_v21': PROJECT / 'work/submission_v2',
    'optimization_shared_scale_contact': PROJECT / 'work/branchfit/shared_scale_contact',
}
PLACEHOLDER = 'https://github.com/placeholder/placeholder/commit/' + '0' * 40
sys.path.insert(0, str(KIT))
from v2dlb import mhr_submission as ms
from v2dlb.mesh_budget import budget_mesh
from v2dlb.mesh_common import _mesh_faces


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(frame):
    h = hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode())
    h.update(frame[['x', 'y', 'z']].to_numpy('float64').tobytes())
    return h.hexdigest()


def uploader():
    spec = importlib.util.spec_from_file_location('official_v2d_submit', KIT / 'submit/v2d_submit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(path, source):
    """Official uploader + scorer array reader + exact source roundtrip, all frames."""
    config = json.loads((KIT / 'config/tracks.json').read_text())
    frame = uploader().validate_submission(path, 'track_1', config)
    expected = pd.read_parquet(SAMPLE, columns=['row_id'])
    assert frame.row_id.equals(expected.row_id)
    assert frame.row_id.is_unique and not frame.isna().any().any()
    parts = frame.row_id.str.split('_', expand=True)
    keys = frame.copy()
    keys['episode_index'] = parts[1].astype(int)
    keys['frame_index'] = parts[2].astype(int)
    keys['array_name'] = parts[3].astype(int).map(dict(enumerate(ms.array_order())))
    keys['point_index'] = parts[4].astype(int)
    reports, guards = [], {}
    for ep, block in keys.groupby('episode_index', sort=True):
        npz = source / f'episode_{ep:06d}.npz'
        mesh = source / f'episode_{ep:06d}_object.ply'
        guards[str(npz)] = sha(npz)
        guards[str(mesh)] = sha(mesh)
        a = ms._to_arrays(block, ['x', 'y', 'z'])
        frames = np.sort(block.loc[block.frame_index != 999999, 'frame_index'].unique())
        with np.load(npz, allow_pickle=False) as d:
            assert set(d.files) == {'pose', 'scales', 'shape', 'object_rotation', 'object_translation', 'object_scale'}
            assert all(np.isfinite(d[k]).all() for k in d.files)
            assert d['pose'].shape[1:] == (136,)
            assert d['scales'].shape == (68,) and d['shape'].shape == (45,)
            assert d['object_rotation'].shape == (len(d['pose']), 3, 3)
            assert d['object_translation'].shape == (len(d['pose']), 3)
            assert frames[-1] < len(d['pose'])
            for packed, original, n in [('mhr_pose', 'pose', 136), ('mhr_scales', 'scales', 68), ('mhr_shape', 'shape', 45)]:
                recovered = ms._unpack(a[packed], n, packed)
                target = d[original][frames] if original == 'pose' else d[original]
                assert np.array_equal(recovered, target), (ep, original)
            ms._rotations(a['object_rotation'])
            assert np.array_equal(a['object_rotation'], d['object_rotation'][frames])
            assert np.array_equal(a['object_translation'][:, 0], d['object_translation'][frames])
            scale = ms._object_scale(a['object_scale'])
            assert scale == float(d['object_scale'])
            vertices = a['object_mesh_vertices']
            faces = _mesh_faces(a['object_mesh_faces'], len(vertices))
            v, f = budget_mesh(mesh, len(faces), len(vertices))
            assert np.array_equal(vertices, v) and np.array_equal(faces, f)
            min_z = float('inf')
            for start in range(0, len(frames), 64):
                ix = frames[start:start + 64]
                z = scale * np.einsum('tj,vj->tv', d['object_rotation'][ix, 2], vertices)
                z += d['object_translation'][ix, 2, None]
                min_z = min(min_z, float(z.min()))
            assert min_z > 0, (ep, min_z)
            reports.append({'episode': int(ep), 'scored_frames': len(frames), 'object_scale': scale,
                            'mesh_vertices': len(vertices), 'mesh_faces': len(faces),
                            'min_scored_mesh_z_m': min_z, 'source_roundtrip_exact': True})
    assert len(reports) == 30 and len(frame) == 740780
    for p, h in guards.items():
        assert sha(p) == h, ('source changed during validation', p)
    return {'file': str(path), 'file_sha256': sha(path), 'bytes': path.stat().st_size,
            'rows': len(frame), 'episodes': 30, 'scored_frames': sum(r['scored_frames'] for r in reports),
            'content_sha256_excluding_commit': digest(frame),
            'code_commit_url': frame.code_commit_url.iloc[0],
            'official_uploader_validation': 'PASS', 'scorer_array_reader_roundtrip': 'PASS',
            'episode_checks': reports, 'source_sha256': guards}


def stage():
    pending = HERE / 'pending_commit'
    pending.mkdir(parents=True, exist_ok=True)
    reports = {}
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONNOUSERSITE='1')
    for name, source in CANDIDATES.items():
        path = pending / f'track1_{name}.parquet'
        log = HERE / f'pack_{name}.log'
        command = [sys.executable, str(KIT / 'eval_reconstruction.py'), '--episodes', str(source),
                   '--sample', str(SAMPLE), '--commit', PLACEHOLDER, '--out', str(path)]
        with log.open('w') as f:
            subprocess.run(command, check=True, cwd=KIT, env=env, stdout=f, stderr=subprocess.STDOUT)
        reports[name] = validate(path, source)
        print(name, 'PASS', reports[name]['rows'], 'rows', flush=True)
    # Both candidates use exactly the same submitted human, mesh, and rotations.
    a = pd.read_parquet(reports['main_v21']['file'])
    b = pd.read_parquet(reports['optimization_shared_scale_contact']['file'])
    array_ids = a.row_id.str.split('_', expand=True)[3].astype(int)
    unchanged = array_ids.isin([0, 1, 2, 3, 6, 7])
    assert np.array_equal(a.loc[unchanged, ['x', 'y', 'z']], b.loc[unchanged, ['x', 'y', 'z']])
    manifest = {'status': 'PAYLOADS_VALIDATED_AWAITING_PUBLISHED_COMMIT',
                'upload_ready': False, 'blocked_on': ['real GitHub commit URL; organizer repository access'],
                'same_human_mesh_and_rotation': True, 'sample_sha256': sha(SAMPLE),
                'candidates': reports, 'note': 'No submission or GitHub writes performed. Neither candidate has official scores.'}
    (HERE / 'PREPARATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


def finalize(url):
    match = re.fullmatch(r'https://github\.com/([\w.-]+)/([\w.-]+)/commit/([0-9a-fA-F]{40})/?', url)
    if not match or any('placeholder' in x.lower() for x in match.groups()[:2]) or match[3] == '0' * 40:
        raise SystemExit('Supply a real GitHub URL: https://github.com/OWNER/REPO/commit/FULL_40_HEX_SHA')
    snapshot = HERE / 'source_repository'
    expected = subprocess.check_output(['git', '-C', str(snapshot), 'rev-parse', 'HEAD'], text=True).strip()
    if match[3].lower() != expected.lower():
        raise SystemExit(f'Publish the prepared source commit {expected}; supplied URL names a different commit.')
    manifest = json.loads((HERE / 'PREPARATION_MANIFEST.json').read_text())
    ready = HERE / 'ready'
    temporary = HERE / 'finalizing'
    if ready.exists():
        raise SystemExit('ready/ already contains files; preserve the previous release before finalizing again.')
    if temporary.exists():
        raise SystemExit('finalizing/ already exists from an interrupted attempt; review it before retrying.')
    temporary.mkdir()
    for name, report in manifest['candidates'].items():
        source = Path(report['file'])
        assert sha(source) == report['file_sha256'], 'Pending file changed'
        for p, h in report['source_sha256'].items():
            assert sha(p) == h, ('Input changed', p)
        frame = pd.read_parquet(source)
        frame['code_commit_url'] = url.rstrip('/')
        path = temporary / source.name
        frame.to_parquet(path, index=False)
        updated = validate(path, CANDIDATES[name])
        assert updated['content_sha256_excluding_commit'] == report['content_sha256_excluding_commit']
        updated['file'] = str(ready / source.name)
        manifest['candidates'][name] = updated
    manifest.update(status='LOCAL_VALIDATION_PASS_WITH_REAL_COMMIT_URL', upload_ready=True,
                    blocked_on=[], organizer_commit_access='USER_MUST_VERIFY; private repositories require Nvidia-V2D-Challenge access',
                    remote_commit_existence='NOT_CHECKED_BY_THIS_OFFLINE_COMMAND',
                    next_step='Run official uploader --dry-run, then upload the SAME candidate file to all five Track 1 metrics.')
    (temporary / 'FINAL_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.rename(ready)
    print('Both finalized files validated in', ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--commit-url')
    args = parser.parse_args()
    if args.stage == bool(args.commit_url):
        parser.error('Choose --stage OR --commit-url URL')
    stage() if args.stage else finalize(args.commit_url)


if __name__ == '__main__':
    main()
