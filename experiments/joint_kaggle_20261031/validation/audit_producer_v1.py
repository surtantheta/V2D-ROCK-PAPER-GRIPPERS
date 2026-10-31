#!/usr/bin/env python3
"""Aggregate independently replayed all-frame case audits and final producer."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LARGE = Path('/rhome/tanuj/v2d_joint_kaggle_20261031')
BASE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert os.environ.get('PBS_JOBID'), 'Production audit requires PBS'
    target = ROOT / 'validation/PRODUCER_OUTPUT_REVIEW.json'
    assert not target.exists()
    path = LARGE / 'COMPLETE.json'
    complete = read(path)
    manifest_path = ROOT / 'FIT_INPUTS_V1.json'
    manifest = read(manifest_path)
    assert complete['status'] == 'FULL30_GENUINE_JOINT_SUBMISSION_COMPLETE'
    assert complete['cases'] == 30 and complete['frames'] == 16563 and complete['scored_frames'] == 9877
    assert complete['human_bitexact'] and complete['mesh_bitexact'] and complete['rotation_scale_bitexact']
    assert not complete['GT_used_for_inference'] and not complete['GT_read']
    assert complete['input_manifest_sha256'] == sha(manifest_path)
    assert [r['episode'] for r in complete['records']] == list(range(30))
    guards = dict(complete['source_sha256'])
    guards.update(complete['output_sha256'])
    guards.update(complete['auxiliary_output_sha256'])
    guards[str(path)] = sha(path)
    guards[str(manifest_path)] = sha(manifest_path)
    guards[str(Path(__file__))] = sha(__file__)
    rows, frames, changed, maximum = [], 0, 0, 0.
    for ep in range(30):
        folder = LARGE / 'episodes' / f'episode_{ep:06d}'
        case_path = folder / 'COMPLETE.json'
        review_path = ROOT / 'validation/SMOKE_OUTPUT_REVIEW.json' if ep == 0 else folder / 'INDEPENDENT_REVIEW.json'
        review = read(review_path)
        assert review['status'] == ('PASS_CHALLENGE_SMOKE' if ep == 0 else 'PASS_CHALLENGE_JOINT_EPISODE')
        assert review['episode'] == ep and review['all_frames_all_5840_hand_points_checked']
        assert review['producer_completion_sha256'] == sha(case_path)
        assert review['input_manifest_sha256'] == sha(manifest_path)
        assert review['human_mesh_rotation_scale_bytes_exact'] and not review['GT_loaded'] and not review['GT_used_for_inference']
        for p, h in review['source_and_output_sha256'].items():
            assert p not in guards or guards[p] == h, ('conflicting audit hash', p)
            guards[p] = h
        guards[str(review_path)] = sha(review_path)
        source = dict(np.load(BASE / f'episode_{ep:06d}.npz', allow_pickle=False))
        candidate = dict(np.load(LARGE / 'submission' / f'episode_{ep:06d}.npz', allow_pickle=False))
        assert len(source['pose']) == review['frames'] == manifest['cases'][ep]['total_frames']
        for key in source:
            if key != 'object_translation':
                assert source[key].shape == candidate[key].shape and source[key].dtype == candidate[key].dtype
                assert source[key].tobytes() == candidate[key].tobytes()
        displacement = np.linalg.norm(candidate['object_translation'] - source['object_translation'], axis=1)
        maximum = max(maximum, float(displacement.max()))
        changed += int((displacement > 1e-6).sum())
        frames += review['frames']
        rows.append(dict(episode=ep, frames=review['frames'], audit=str(review_path), audit_sha256=sha(review_path),
                         producer_completion_sha256=sha(case_path), maximum_correction_m=review['maximum_correction_m'],
                         initial_merit=review['initial_merit'], final_merit=review['final_merit'],
                         budgeted_mesh_vertices_array_sha256=review['budgeted_mesh_vertices_array_sha256'],
                         budgeted_mesh_faces_array_sha256=review['budgeted_mesh_faces_array_sha256']))
    assert frames == 16563 and maximum <= .020000000001 and changed > 0
    expected = {f'episode_{ep:06d}{suffix}' for ep in range(30) for suffix in ['.npz', '_object.ply']}
    assert {p.name for p in (LARGE / 'submission').iterdir()} == expected
    for p, h in guards.items():
        assert sha(p) == h, ('source/artifact changed', p)
    receipt = dict(status='PASS_PRODUCER_ALL30', cases=30, frames=16563, full_frames=16563, scored_frames=9877,
                   producer_completion=str(path), producer_completion_sha256=sha(path),
                   input_manifest_sha256=sha(manifest_path), submission=str(LARGE / 'submission'),
                   all_frames_all_5840_hand_points_independently_replayed=True,
                   human_mesh_rotation_scale_bytes_exact=True, maximum_correction_m=maximum,
                   frames_changed_more_than_1um=changed, per_episode_audits=rows,
                   budgeted_array_hash_definition='SHA256 of C-contiguous raw returned-array bytes; helper vertices float64 and faces int64',
                   GT_loaded=False, GT_used_for_inference=False, exact_hosted_metrics=False,
                   source_and_output_sha256=guards, output_sha256=complete['output_sha256'],
                   CPU_job=os.environ['PBS_JOBID'], audited_utc=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['per_episode_audits', 'source_and_output_sha256', 'output_sha256']}), flush=True)


if __name__ == '__main__':
    main()
