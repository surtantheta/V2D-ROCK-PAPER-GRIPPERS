#!/usr/bin/env python3
"""Reject frozen wrong-instance masks BEFORE FP reacquisition, no imputation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'
SEQ = 'Date01_Sub01_stool_move'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from model_source_guard import verify_model_group
    provenance = verify_model_group('FoundationPose')
    runtime = ROOT/'labeled_association_tracking_v1'
    inputs = ROOT/'labeled_predictions_v2/RUN_INPUTS.json'
    item = next(x for x in json.loads(inputs.read_text())['items'] if x['name'] == SEQ)
    assets = ROOT/'labeled_predictions_v2/BUDGETED.json'
    asset = next(x for x in json.loads(assets.read_text())['records'] if x['sequence'] == SEQ)
    assert sha(asset['mesh_path']) == asset['mesh_sha256']
    masks = runtime/'masks'/SEQ; old_masks = Path(item['masks'])/'MASKS.json'
    before = json.loads(old_masks.read_text()); after = json.loads((masks/'MASKS.json').read_text())
    audit_path = ROOT/'labeled_predictions_v2/MASK_CONTINUITY_AUDIT.json'
    audit = next(x for x in json.loads(audit_path.read_text())['records'] if x['sequence'] == SEQ)
    expected = json.loads(json.dumps(before))
    for i in audit['rejected_local_indices']: expected['rows'][i]['selected'] = False
    assert after == expected and len(audit['rejected_local_indices']) == 11
    env = os.environ.copy()
    if torch.cuda.get_device_capability() == (7, 5): env['PYTHONPATH'] = str(ROOT/'deps/sm75')
    track = runtime/'tracks'/SEQ
    subprocess.run([PY, str(ROOT/'code/labeled_track_fp_snapshot_v1.py'), '--frames', item['frames'],
        '--masks', str(masks), '--human', str(ROOT/'labeled_predictions_v1/depth'/SEQ),
        '--object', asset['directory'], '--out', str(track)], cwd='/home/tanuj/projects/CARI4D', env=env, check=True)
    original = ROOT/'labeled_predictions_v2/human_anchored_comparison/foundationpose_PREDICTIONS.json'
    data = json.loads(original.read_text()); poses = np.load(track/'poses.npz')['T_obj2cam']
    assert poses.shape == (96, 4, 4) and np.isfinite(poses).all()
    row = next(x for x in data['sequences'] if x['sequence'] == SEQ)
    hp = ROOT/'labeled_predictions_v1/human_scale/RESULT.json'
    alpha = next(x['alpha'] for x in json.loads(hp.read_text())['records'] if x['sequence'] == SEQ)
    assert row['object_scale'] == alpha and row['mesh_sha256'] == asset['mesh_sha256']
    for i, frame in enumerate(row['frames']):
        assert frame['video_frame_index'] == int(Path(item['source_paths'][i]).stem)
        frame['R_obj2cam'] = poses[i, :3, :3].tolist()
        frame['t_obj2cam_m'] = (alpha * poses[i, :3, 3]).tolist()
    row['source']['association_tracking'] = dict(mask_manifest_sha256=sha(masks/'MASKS.json'),
        poses_path=str(track/'poses.npz'), poses_sha256=sha(track/'poses.npz'),
        identity_audit_sha256=sha(audit_path), alpha=alpha, pose_imputation=False,
        intervention='only disable mask-guided reacquisition for11 frozen wrong-instance observations')
    row['source']['poses_path'] = str(track/'poses.npz')
    row['source']['poses_sha256'] = sha(track/'poses.npz')
    row['source']['optimizer'] = 'foundationpose_with_frozen_association_mask_gate'
    row['source']['human_anchor']['same_alpha_reapplied_to_new_raw_tracking_translation'] = True
    assert len(data['sequences']) == 4 and sum(len(s['frames']) for s in data['sequences']) == 384
    out = runtime/'comparison'; out.mkdir(exist_ok=False)
    p = out/'foundationpose_association_PREDICTIONS.json'; p.write_text(json.dumps(data, indent=2, allow_nan=False))
    (runtime/'TRACKING_COMPLETE.json').write_text(json.dumps(dict(model=provenance,
        prediction=str(p), prediction_sha256=sha(p), pose_imputation=False,
        reused_three_sequences_exact=True, full_coverage=384, inference_GT_inputs=False,
        source_sha256={str(p): sha(p) for p in [Path(__file__), inputs, assets, original, hp, old_masks, masks/'MASKS.json', audit_path]}), indent=2))


if __name__ == '__main__': main()
