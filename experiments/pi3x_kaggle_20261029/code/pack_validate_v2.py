"""Pack all 30 challenge clips and verify the official scorer roundtrip in PBS."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path('/home/tanuj/V2D_challenge')
K = R/'experiments/pi3x_kaggle_20261029'
L = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029')
KIT = R/'data/assets/submission_kit/v2d_submission_kit'


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert os.environ.get('PBS_JOBID'), 'Packing and full validation require PBS'
    completion = L/'cpu_v2/COMPLETE.json'
    done = json.loads(completion.read_text())
    assert done['cases'] == 30 and done['frames'] == 16563
    assert done['human_bitexact'] and done['mesh_byteexact']
    assert done['GT_used_for_inference'] is False and done['GT_read'] is False
    for path, digest in done['output_sha256'].items():
        assert sha(path) == digest, ('producer output changed', path)
    source = Path(done['submission'])
    commit = (K/'SOURCE_COMMIT_URL.txt').read_text().strip()
    match = re.fullmatch(r'https://github.com/surtantheta/V2D-ROCK-PAPER-GRIPPERS/commit/([0-9a-f]{40})', commit)
    assert match and match[1] != '0'*40, 'An actual published source commit is required'
    publication = json.loads((K/'SOURCE_PUBLICATION_RECEIPT.json').read_text())
    assert publication['commit_sha'] == match[1] and publication['published']
    metadata = json.loads((K/'CHALLENGE_EPISODES.json').read_text())
    assert len(list(source.glob('episode_*.npz'))) == 30
    assert len(list(source.glob('episode_*_object.ply'))) == 30
    for path, digest in publication['source_sha256'].items():
        assert sha(path) == digest, ('Published source changed', path)
    changed = []
    for ep in range(30):
        stem = f'episode_{ep:06d}'
        with np.load(source/f'{stem}.npz') as new, np.load(R/f'work/submission_v2/{stem}.npz') as old:
            assert set(new.files) == {'pose', 'scales', 'shape', 'object_rotation', 'object_translation', 'object_scale'}
            assert all(np.isfinite(new[key]).all() for key in new.files)
            assert len(new['pose']) == metadata[str(ep)]['frames']
            rotation = new['object_rotation']
            assert rotation.shape == (len(new['pose']), 3, 3)
            assert np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-4, rtol=0)
            assert np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
            assert new['object_translation'].shape == (len(new['pose']), 3)
            assert np.asarray(new['object_scale']).size == 1 and float(new['object_scale']) > 0
            for key in ['pose', 'scales', 'shape']:
                assert new[key].dtype == old[key].dtype and np.array_equal(new[key], old[key])
            changed.append(dict(episode=ep,
                translation_changed_frames=int(np.any(new['object_translation'] != old['object_translation'], axis=1).sum()),
                rotation_changed_frames=int(np.any(new['object_rotation'] != old['object_rotation'], axis=(1, 2)).sum()),
                scale_changed=bool(float(new['object_scale']) != float(old['object_scale']))))
        assert sha(source/f'{stem}_object.ply') == sha(R/f'work/submission_v2/{stem}_object.ply')
    assert any(c['translation_changed_frames'] or c['rotation_changed_frames'] or c['scale_changed'] for c in changed), 'Candidate is identical to MAIN'
    ready = K/'ready'
    ready.mkdir(exist_ok=True)
    parquet = ready/'track1_pi3x_geometry_pose_old_masks.parquet'
    assert not parquet.exists(), 'Preserve previous artifacts'
    subprocess.run([sys.executable, '-I', '-B', '-u', str(KIT/'eval_reconstruction.py'),
                    '--episodes', str(source), '--sample', str(KIT/'data/track_1_sample_submission.parquet'),
                    '--commit', commit, '--out', str(parquet)], check=True, cwd=KIT)
    original = R/'submissions/upload_prep_20261026/prepare_uploads.py'
    spec = importlib.util.spec_from_file_location('original_upload_validation', original)
    validation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validation
    spec.loader.exec_module(validation)
    result = validation.validate(parquet, source)
    assert result['rows'] == 740780 and result['episodes'] == 30 and result['scored_frames'] == 9877
    result.update(status='PACKED_AND_FULL_ROUNDTRIP_PASS', source_commit=commit,
        candidate='Pi3X geometry + original local pose/scale optimizer + existing MAIN masks',
        changed=changed, human_MAIN_bitexact=True, mesh_MAIN_byteexact=True,
        producer_completion_sha256=sha(completion), CPU_job=os.environ['PBS_JOBID'],
        Kaggle_upload_performed=False)
    (ready/'PACK_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FULL30_PARQUET_READY', str(parquet), result['file_sha256'], flush=True)


if __name__ == '__main__':
    main()
