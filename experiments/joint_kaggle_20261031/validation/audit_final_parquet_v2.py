#!/usr/bin/env python3
"""Independent every-row final joint Parquet audit, without GT or body decode.

Requires the separate all16563-frame/all5840-point independent producer audits.
Every packed mesh must be exactly the mesh used by those physical audits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge')
OLD = Path('/home/tanuj/V2D_challenge')
LARGE = Path('/rhome/tanuj/v2d_joint_kaggle_20261031')
BASELINE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
BASE_PARQUET = OLD / 'experiments/pi3x_kaggle_20261029/ready/track1_pi3x_geometry_pose_old_masks.parquet'
BASE_PARQUET_SHA = '93ce87139874062838fd21bebef215273fa6d1c92b10d02cfc0857d587d33b31'
KIT = PROJECT / 'data/assets/submission_kit/v2d_submission_kit'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, ('source/artifact changed', path)


def pad(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    padded = np.zeros(3 * ((len(values) + 2) // 3))
    padded[:len(values)] = values
    return padded.reshape(-1, 3)


def run(args):
    assert os.environ.get('PBS_JOBID'), 'Final artifact audit requires PBS'
    assert not args.out.exists()
    import pandas as pd
    import trimesh
    sys.path.insert(0, str(KIT))
    from v2dlb import mhr_submission as ms
    from v2dlb.mesh_budget import budget_mesh
    producer_path = LARGE / 'COMPLETE.json'
    producer = read(producer_path)
    assert producer['status'] == 'FULL30_GENUINE_JOINT_SUBMISSION_COMPLETE'
    assert producer['cases'] == 30 and producer['frames'] == 16563
    assert producer['scored_frames'] == 9877 and producer['all_5840_hand_points_all_frames']
    assert producer['human_bitexact'] and producer['mesh_bitexact'] and producer['rotation_scale_bitexact']
    assert not producer['GT_used_for_inference'] and not producer['GT_read']
    producer_audit_path = ROOT / 'validation/PRODUCER_OUTPUT_REVIEW_V2.json'
    producer_audit = read(producer_audit_path)
    assert producer_audit['status'] == 'PASS_PRODUCER_ALL30'
    assert producer_audit['cases'] == 30 and producer_audit['frames'] == 16563
    assert producer_audit['all_frames_all_5840_hand_points_independently_replayed']
    assert producer_audit['producer_completion_sha256'] == sha(producer_path)
    assert producer_audit['input_manifest_sha256'] == producer['input_manifest_sha256']
    audit_by_ep = {r['episode']: r for r in producer_audit['per_episode_audits']}
    assert sorted(audit_by_ep) == list(range(30))
    pins = dict(producer_audit['source_and_output_sha256'])
    pins.update(producer['output_sha256'])
    pins.update(producer['source_sha256'])
    pins.update(producer['auxiliary_output_sha256'])
    publication_path = ROOT / 'SOURCE_PUBLICATION_RECEIPT_V2.json'
    publication = read(publication_path)
    commit_path = ROOT / 'SOURCE_COMMIT_URL_V2.txt'
    commit = commit_path.read_text().strip()
    match = re.fullmatch(r'https://github.com/surtantheta/V2D-ROCK-PAPER-GRIPPERS/commit/([0-9a-f]{40})', commit)
    assert match and match.group(1) == publication['commit_sha']
    assert publication['published'] and publication['public_repo_verified']
    assert publication['commit_url'] == commit
    assert publication['main_ref_unchanged'] and publication['README_diff'] is False
    assert publication['remote_file_hashes_verified'] == len(publication['source_sha256']) > 0
    pins.update(publication['source_sha256'])
    pack_path = ROOT / 'ready_v2/PACK_VALIDATION.json'
    pack = read(pack_path)
    assert pack['status'] == 'PACKED_AND_FULL_ROUNDTRIP_PASS'
    assert pack['rows'] == 740780 and pack['episodes'] == 30 and pack['scored_frames'] == 9877
    assert pack['file_sha256'] == sha(args.parquet)
    assert pack['source_commit'] == commit
    if 'code_commit_url' in pack:
        assert pack['code_commit_url'] == commit
    assert sha(BASE_PARQUET) == BASE_PARQUET_SHA
    metadata_path = OLD / 'experiments/pi3x_kaggle_20261029/CHALLENGE_EPISODES.json'
    metadata = read(metadata_path)
    sample_path = KIT / 'data/track_1_sample_submission.parquet'
    for path in [producer_path, producer_audit_path, publication_path, commit_path,
                 pack_path, args.parquet, BASE_PARQUET, metadata_path, sample_path, Path(__file__)]:
        pins[str(path)] = sha(path)
    verify(pins)
    sample = pd.read_parquet(sample_path, columns=['row_id'])
    frame = pd.read_parquet(args.parquet)
    baseline_frame = pd.read_parquet(BASE_PARQUET)
    assert list(frame.columns) == ['row_id', 'x', 'y', 'z', 'code_commit_url']
    assert len(frame) == len(sample) == 740780 and frame.row_id.is_unique
    assert np.array_equal(frame.row_id.to_numpy(), sample.row_id.to_numpy())
    assert frame.code_commit_url.nunique() == 1 and frame.code_commit_url.iloc[0] == commit
    assert frame.row_id.equals(baseline_frame.row_id)
    values = frame[['x', 'y', 'z']].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    split = frame.row_id.str.split('_', expand=True)
    assert split.shape[1] == 5 and (split[0] == 't1').all()
    keys = pd.DataFrame({name: split[i].astype(np.int64) for i, name in [(1, 'episode'), (2, 'frame'), (3, 'array'), (4, 'point')]})
    assert sorted(keys.episode.unique()) == list(range(30))
    order = ms.array_order()
    sentinel = ms.EPISODE_FRAME
    assert sentinel == 999999 and order.index('object_translation') == 4
    unchanged = keys.array.to_numpy() != 4
    assert np.array_equal(values[unchanged], baseline_frame[['x', 'y', 'z']].to_numpy()[unchanged])
    sub = Path(producer['submission'])
    expected_files = {f'episode_{ep:06d}{suffix}' for ep in range(30) for suffix in ['.npz', '_object.ply']}
    assert {p.name for p in sub.iterdir()} == expected_files
    count, full_frames, scored_frames, changed = 0, 0, 0, 0
    minimum_z, maximum = float('inf'), 0.
    records = []
    for ep, group in keys.groupby('episode'):
        stem = f'episode_{ep:06d}'
        data = dict(np.load(sub / (stem + '.npz'), allow_pickle=False))
        baseline = dict(np.load(BASELINE / (stem + '.npz'), allow_pickle=False))
        n = len(data['pose'])
        assert n == metadata[str(ep)]['frames'] == audit_by_ep[ep]['frames']
        assert metadata[str(ep)]['frame_ids'] == list(range(n))
        assert np.allclose(metadata[str(ep)]['times'], np.arange(n) / 30, atol=1e-10, rtol=0)
        assert set(data) == set(baseline)
        for name in data:
            assert np.isfinite(data[name]).all()
            if name != 'object_translation':
                assert data[name].shape == baseline[name].shape and data[name].dtype == baseline[name].dtype
                assert data[name].tobytes() == baseline[name].tobytes()
        assert data['pose'].shape == (n, 136) and data['scales'].shape == (68,) and data['shape'].shape == (45,)
        rotation, translation, scale = data['object_rotation'], data['object_translation'], np.asarray(data['object_scale'])
        assert rotation.shape == (n, 3, 3) and translation.shape == (n, 3) and scale.shape == () and float(scale) > 0
        assert np.max(abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3))) < 1e-6
        assert np.max(abs(np.linalg.det(rotation) - 1)) < 1e-6
        displacement = np.linalg.norm(translation - baseline['object_translation'], axis=1)
        maximum = max(maximum, float(displacement.max()))
        changed += int((displacement > 1e-6).sum())
        assert displacement.max() <= .020000000001
        mesh_path = sub / (stem + '_object.ply')
        assert sha(mesh_path) == sha(BASELINE / (stem + '_object.ply'))
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
        assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
        for start in range(0, n, 64):
            z = float(scale) * (rotation[start:start + 64, 2, :] @ vertices.T) + translation[start:start + 64, 2, None]
            minimum_z = min(minimum_z, float(z.min()))
            assert z.min() >= .02
        scored = sorted(group.loc[group.frame != sentinel, 'frame'].unique())
        assert scored and scored[-1] < n
        static = group[group.frame == sentinel]
        vertex_count = int((static.array == order.index('object_mesh_vertices')).sum())
        face_count = int((static.array == order.index('object_mesh_faces')).sum())
        assert vertex_count == face_count == 4096
        mv, mf = budget_mesh(mesh_path, face_count, vertex_count)
        assert mv.dtype == np.float64 and mf.dtype == np.int64
        # This closes the gap between optimizer/PEN meshes and submission meshes.
        assert hashlib.sha256(np.ascontiguousarray(mv).tobytes()).hexdigest() == audit_by_ep[ep]['budgeted_mesh_vertices_array_sha256'], ('physical/packed budget vertices differ', ep)
        assert hashlib.sha256(np.ascontiguousarray(mf).tobytes()).hexdigest() == audit_by_ep[ep]['budgeted_mesh_faces_array_sha256'], ('physical/packed budget faces differ', ep)
        per_episode = dict(mhr_scales=pad(data['scales']), mhr_shape=pad(data['shape']),
                           object_scale=np.array([[float(scale), 0., 0.]]),
                           object_mesh_vertices=mv, object_mesh_faces=mf.astype(float))
        for (index, array), rows in group.groupby(['frame', 'array']):
            field = order[int(array)]
            points = rows.point.to_numpy()
            if index == sentinel:
                expected = per_episode[field]
            elif field == 'mhr_pose':
                expected = pad(data['pose'][index])
            elif field == 'object_rotation':
                expected = rotation[index]
            elif field == 'object_translation':
                expected = translation[index, None]
            else:
                raise AssertionError(('unexpected per-frame array', field))
            assert np.array_equal(values[rows.index.to_numpy()], expected[points]), ('Parquet/source mismatch', ep, index, field)
            count += len(rows)
        full_frames += n
        scored_frames += len(scored)
        records.append(dict(episode=int(ep), full_frames=n, scored_frames=len(scored), first_scored_frame=int(scored[0]),
                            last_scored_frame=int(scored[-1]), maximum_correction_m=float(displacement.max()),
                            physical_and_packed_budgeted_mesh_exact=True))
    assert count == 740780 and full_frames == 16563 and scored_frames == 9877 and changed > 0
    verify(pins)
    receipt = dict(status='PASS_UPLOAD_READY', cases=30, full_frames=16563, frames=16563, scored_frames=9877,
                   rows=740780, verified_Parquet_rows=count, submission_files=60,
                   exact_sample_row_ID_order=True, all_nontranslation_values_equal_prior_uploaded_Pi3X=True,
                   every_packed_value_equals_producer_or_official_budget_mesh=True,
                   physical_and_packed_budgeted_mesh_exact_all30=True,
                   MAIN_human_mesh_R_scale_bytes_preserved=True, all_full_frame_rotations_proper=True,
                   one_positive_scale_per_episode=True, maximum_correction_m=maximum,
                   changed_full_frames_over_1um=changed, min_all_frame_all_original_vertex_Z_m=minimum_z,
                   all16563_frames_all5840_hand_points_previously_independently_replayed=True,
                   independent_producer_review_sha256=sha(producer_audit_path),
                   producer_completion_sha256=sha(producer_path), input_manifest_sha256=producer['input_manifest_sha256'],
                   code_commit_url=commit, exact_commit=commit, file_sha256=sha(args.parquet),
                   final_parquet=str(args.parquet), final_parquet_sha256=sha(args.parquet),
                   final_parquet_bytes=args.parquet.stat().st_size, GT_used_for_inference=False, GT_loaded=False,
                   exact_hosted_metric_or_SOTA_claim=False, Kaggle_upload_performed=False,
                   source_publication_hashes_verified=len(publication['source_sha256']),
                   remote_verification_basis='Separate root publication receipt verifies actual remote bytes; this audit verifies every local published source hash.',
                   source_sha256=publication['source_sha256'],
                   output_sha256={**producer['output_sha256'], str(args.parquet): sha(args.parquet), str(pack_path): sha(pack_path)},
                   sha256={str(Path(__file__)): sha(__file__), str(producer_audit_path): sha(producer_audit_path),
                           str(publication_path): sha(publication_path), str(pack_path): sha(pack_path)},
                   records=records, CPU_audit_job=os.environ['PBS_JOBID'], audited_utc=datetime.now(timezone.utc).isoformat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['records', 'source_sha256', 'output_sha256', 'sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, default=ROOT / 'ready_v2/track1_pi3x_joint_physics.parquet')
    parser.add_argument('--out', type=Path, default=ROOT / 'validation/FINAL_UPLOAD_AUDIT_V2.json')
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        failed = args.out.with_suffix('.failed.json')
        if not failed.exists():
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(json.dumps(dict(status='NO_GO_FINAL_ARTIFACT_AUDIT_FAILED', error=repr(error),
                traceback=traceback.format_exc(), CPU_job=os.environ.get('PBS_JOBID')), indent=2) + '\n')
        raise
