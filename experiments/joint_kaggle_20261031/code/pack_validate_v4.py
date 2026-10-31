#!/usr/bin/env python3
"""Pack the full30 translation-only joint result and verify every scored value."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge')
OLD_PROJECT = Path('/home/tanuj/V2D_challenge')
LARGE = Path('/rhome/tanuj/v2d_joint_kaggle_20261031')
BASELINE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
BASELINE_PARQUET = OLD_PROJECT / 'experiments/pi3x_kaggle_20261029/ready/track1_pi3x_geometry_pose_old_masks.parquet'
BASELINE_PARQUET_SHA = '93ce87139874062838fd21bebef215273fa6d1c92b10d02cfc0857d587d33b31'
KIT = PROJECT / 'data/assets/submission_kit/v2d_submission_kit'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('PBS_JOBID'), 'Packing and full roundtrip use PBS'
    completion = LARGE / 'COMPLETE.json'
    done = json.loads(completion.read_text())
    assert done['cases'] == 30 and done['frames'] == 16563
    assert done['GT_used_for_inference'] is False and done['GT_read'] is False
    for path, digest in done['output_sha256'].items():
        assert sha(path) == digest, ('producer changed', path)
    producer_audit_path = ROOT / 'validation/PRODUCER_OUTPUT_REVIEW_V2.json'
    producer_audit = json.loads(producer_audit_path.read_text())
    assert producer_audit['status'] == 'PASS_PRODUCER_ALL30'
    assert producer_audit['cases'] == 30 and producer_audit['full_frames'] == 16563
    assert producer_audit['producer_completion_sha256'] == sha(completion)
    source = Path(done['submission'])
    assert source.resolve().is_relative_to(LARGE)
    assert len(list(source.glob('episode_*.npz'))) == 30
    assert len(list(source.glob('episode_*_object.ply'))) == 30
    commit = (ROOT / 'SOURCE_COMMIT_URL_V2.txt').read_text().strip()
    match = re.fullmatch(r'https://github.com/surtantheta/V2D-ROCK-PAPER-GRIPPERS/commit/([0-9a-f]{40})', commit)
    assert match and match[1] != '0' * 40, 'Actual published commit required'
    publication = json.loads((ROOT / 'SOURCE_PUBLICATION_RECEIPT_V2.json').read_text())
    assert publication['published'] and publication['public_repo_verified']
    assert publication['commit_sha'] == match[1]
    for path, digest in publication['source_sha256'].items():
        assert sha(path) == digest, ('published source changed', path)
    metadata = json.loads((OLD_PROJECT / 'experiments/pi3x_kaggle_20261029/CHALLENGE_EPISODES.json').read_text())
    changes = []
    total = 0
    correction_squared_sum = 0.0
    for ep in range(30):
        stem = f'episode_{ep:06d}'
        with np.load(source / (stem + '.npz'), allow_pickle=False) as new, np.load(BASELINE / (stem + '.npz'), allow_pickle=False) as old:
            assert set(new.files) == set(old.files) == {'pose', 'scales', 'shape', 'object_rotation', 'object_translation', 'object_scale'}
            assert all(np.isfinite(new[k]).all() for k in new.files)
            n = metadata[str(ep)]['frames']
            assert len(new['pose']) == n and new['object_translation'].shape == (n, 3)
            total += n
            for key in ['pose', 'scales', 'shape', 'object_rotation', 'object_scale']:
                assert new[key].dtype == old[key].dtype and new[key].tobytes() == old[key].tobytes(), ('nontranslation field changed', ep, key)
            delta = np.linalg.norm(new['object_translation'].astype(float) - old['object_translation'], axis=1)
            assert np.max(delta) <= .02000001, ('joint correction exceeds20mm', ep, np.max(delta))
            correction_squared_sum += float(np.sum(delta * delta))
            changes.append(dict(episode=ep, frames=n, changed_frames=int((delta > 1e-10).sum()),
                                changed_frames_over_0p1mm=int((delta > 1e-4).sum()),
                                rms_translation_correction_m=float(np.sqrt(np.mean(delta * delta))),
                                maximum_translation_correction_m=float(delta.max())))
        assert sha(source / (stem + '_object.ply')) == sha(BASELINE / (stem + '_object.ply'))
    assert total == 16563
    assert sum(x['changed_frames_over_0p1mm'] for x in changes) > 0, 'No frame changed by more than0.1mm; review before spending another round'
    ready = ROOT / 'ready_v2'
    ready.mkdir(exist_ok=True)
    artifact = ready / 'track1_pi3x_joint_physics.parquet'
    assert not artifact.exists(), 'Preserve existing artifact'
    subprocess.run([sys.executable, '-B', '-u', str(KIT / 'eval_reconstruction.py'),
                    '--episodes', str(source), '--sample', str(KIT / 'data/track_1_sample_submission.parquet'),
                    '--commit', commit, '--out', str(artifact)], cwd=KIT, check=True)
    original = OLD_PROJECT / 'submissions/upload_prep_20261026/prepare_uploads.py'
    spec = importlib.util.spec_from_file_location('original_full_roundtrip', original)
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    result = validator.validate(artifact, source)
    assert (result['rows'], result['episodes'], result['scored_frames']) == (740780, 30, 9877)
    # A second, direct comparison establishes that every nontranslation packed
    # value stays exactly identical to the actually submitted Pi3X artifact.
    assert sha(BASELINE_PARQUET) == BASELINE_PARQUET_SHA
    import pandas as pd
    baseline = pd.read_parquet(BASELINE_PARQUET)
    candidate = pd.read_parquet(artifact)
    assert candidate.row_id.equals(baseline.row_id)
    baseline_content_sha256 = validator.digest(baseline)
    assert result['content_sha256_excluding_commit'] != baseline_content_sha256, 'Candidate content is identical to previous Pi3X'
    array_id = candidate.row_id.str.split('_', expand=True)[3].astype(int)
    unchanged = array_id != 4
    assert np.array_equal(candidate.loc[unchanged, ['x', 'y', 'z']].to_numpy(),
                          baseline.loc[unchanged, ['x', 'y', 'z']].to_numpy())
    # Match the exact budgeted surface on which every full-frame collision
    # objective was fitted and independently checked in the producer audit.
    parts = candidate.row_id.str.split('_', expand=True)
    ep_ids, frame_ids = parts[1].astype(int), parts[2].astype(int)
    mesh_budget_audits = []
    for ep in range(30):
        review_path = (ROOT / 'validation/SMOKE_OUTPUT_REVIEW_V2.json' if ep == 12 else
                       LARGE / f'episodes_v2/episode_{ep:06d}/INDEPENDENT_REVIEW.json')
        review = json.loads(review_path.read_text())
        assert str(review['status']).startswith('PASS') and review['episode'] == ep
        static = (ep_ids == ep) & (frame_ids == 999999)
        vertices = candidate.loc[static & (array_id == 6), ['x', 'y', 'z']].to_numpy(dtype=np.float64)
        packed_faces = candidate.loc[static & (array_id == 7), ['x', 'y', 'z']].to_numpy(dtype=np.float64)
        faces = validator._mesh_faces(packed_faces[None], len(vertices))
        vhash = hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest()
        fhash = hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()
        assert vhash == review['budgeted_mesh_vertices_array_sha256'], ('fit/upload budget vertices differ', ep)
        assert fhash == review['budgeted_mesh_faces_array_sha256'], ('fit/upload budget faces differ', ep)
        mesh_budget_audits.append(dict(episode=ep, vertices_sha256=vhash, faces_sha256=fhash,
                                      review_sha256=sha(review_path)))
    result.update(status='PACKED_AND_FULL_ROUNDTRIP_PASS', source_commit=commit,
                  candidate='Pi3X existing masks plus genuine joint translation optimization with uniform oracle-matched closest-point repair',
                  changed=changes, full_frames=total, translation_only=True,
                  full_sequence_correction_RMS_m=float(np.sqrt(correction_squared_sum / total)),
                  frames_changed_over_0p1mm=sum(x['changed_frames_over_0p1mm'] for x in changes),
                  baseline_content_sha256_excluding_commit=baseline_content_sha256,
                  content_differs_from_submitted_Pi3X=True,
                  human_MAIN_bitexact=True, human_R_scale_mesh_Pi3X_byteexact=True,
                  packed_nontranslation_values_Pi3X_exact=True,
                  mesh_budget_fit_upload_exact=True, mesh_budget_audits=mesh_budget_audits,
                  producer_independent_audit_sha256=sha(producer_audit_path),
                  producer_completion_path=str(completion), producer_completion_sha256=sha(completion),
                  publication_receipt_sha256=sha(ROOT / 'SOURCE_PUBLICATION_RECEIPT_V2.json'),
                  baseline_submitted_parquet_sha256=BASELINE_PARQUET_SHA,
                  CPU_job=os.environ['PBS_JOBID'], Kaggle_upload_performed=False)
    (ready / 'PACK_VALIDATION.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FULL30_JOINT_PARQUET_READY', str(artifact), result['file_sha256'], flush=True)


if __name__ == '__main__':
    main()
