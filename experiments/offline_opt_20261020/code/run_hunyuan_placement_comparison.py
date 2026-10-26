#!/usr/bin/env python3
"""Paired generated meshes with identical RGB/depth/masks/keyframe tracking.

Before any GT scoring, select one generator per clip using non-keyframe mask
IoU. Hunyuan inherits predicted SAM3D scale/orientation; no native CAD is input.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from labeled_object_scorer import POSE_CONVENTION

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
CARI = Path('/home/tanuj/projects/CARI4D')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from model_source_guard import verify_model_group
    model = verify_model_group('FoundationPose')
    runtime = ROOT/'labeled_hunyuan_placement_v1'
    source = runtime/'RUNTIME.json'; data = json.loads(source.read_text())
    for p, h in data['source_sha256'].items(): assert sha(p) == h, p
    assert not data['GT_inputs_provided'] and not data['scorer_transform_used']
    inputs = ROOT/'labeled_predictions_v2/RUN_INPUTS.json'
    items = {x['name']: x for x in json.loads(inputs.read_text())['items'] if x['dataset'] == 'BEHAVE'}
    audit_path = ROOT/'labeled_predictions_v2/MASK_CONTINUITY_AUDIT.json'
    audit = {x['sequence']: x for x in json.loads(audit_path.read_text())['records']}
    hp = ROOT/'labeled_predictions_v1/human_scale/RESULT.json'
    alphas = {x['sequence']: x['alpha'] for x in json.loads(hp.read_text())['records']}
    env = os.environ.copy()
    if torch.cuda.get_device_capability() == (7, 5): env['PYTHONPATH'] = str(ROOT/'deps/sm75')
    out = runtime/'comparison'; out.mkdir(exist_ok=False)
    arms = {}; measurements = {}; sources = {str(p): sha(p) for p in [Path(__file__), source, inputs, audit_path, hp]}
    for name, budget_name in [('hunyuan', 'BUDGETED.json'), ('sam3d_matched', 'SAM3D_MATCHED_BUDGETED.json')]:
        records = []; measurements[name] = {}
        budget = runtime/budget_name; sources[str(budget)] = sha(budget)
        assets = json.loads(budget.read_text())['records']; assert len(assets) == 4
        for asset in assets:
            seq = asset['sequence']; item = items[seq]; mesh = Path(asset['mesh_path'])
            assert sha(mesh) == asset['mesh_sha256']; directory = Path(asset['directory'])
            kf_path = directory/'keyframe.json'; kf = json.loads(kf_path.read_text())
            track = runtime/'tracks'/name/seq
            subprocess.run([PY, str(ROOT/'code/labeled_track_fp_snapshot_v1.py'), '--frames', item['frames'],
                '--masks', item['masks'], '--human', str(ROOT/'labeled_predictions_v1/depth'/seq),
                '--object', str(directory), '--out', str(track)], cwd=CARI, env=env, check=True)
            poses_path = track/'poses.npz'; result = np.load(poses_path); poses = result['T_obj2cam']; iou = result['iou']
            assert poses.shape == (96, 4, 4) and np.isfinite(poses).all()
            mask_path = Path(item['masks'])/'MASKS.json'; rows = json.loads(mask_path.read_text())['rows']
            valid = np.array([r['selected'] and r['score'] >= .8 for r in rows])
            valid &= ~np.array(audit[seq]['flags'], bool); valid[kf['keyframe']] = False
            assert valid.sum() >= 8 and np.isfinite(iou[valid]).all()
            measurements[name][seq] = dict(mean_IoU=float(iou[valid].mean()), count=int(valid.sum()), local_indices=np.flatnonzero(valid).tolist())
            for p in [mesh, kf_path, poses_path, mask_path]: sources[str(p)] = sha(p)
            records.append(dict(sequence=seq, camera=1, mesh_path=str(mesh), mesh_sha256=sha(mesh),
                mesh_units='metres', object_scale=1., pose_convention=POSE_CONVENTION,
                origin_convention='same-input generated SAM3D local origin; predicted Sim3 baked into Hunyuan vertices',
                origin_correspondence_to_gt='unverified',
                source=dict(generator=name, predicted_placement_receipt_sha256=sha(source),
                    poses_sha256=sha(poses_path), scale_pose_independent_Hunyuan_estimates=False,
                    GT_used_by_inference=False, keyframe=kf['keyframe']),
                frames=[dict(video_frame_index=int(Path(p).stem), R_obj2cam=poses[i, :3, :3].tolist(),
                             t_obj2cam_m=poses[i, :3, 3].tolist()) for i, p in enumerate(item['source_paths'])]))
        arms[name] = records
    choice = {}; arms['image_selected'] = []
    for seq in items:
        a, b = measurements['hunyuan'][seq], measurements['sam3d_matched'][seq]
        assert a['local_indices'] == b['local_indices']
        winner = 'hunyuan' if a['mean_IoU'] > b['mean_IoU'] else 'sam3d_matched'
        choice[seq] = winner
        arms['image_selected'].append(next(x for x in arms[winner] if x['sequence'] == seq))
    (out/'IMAGE_SELECTION_BEFORE_GT.json').write_text(json.dumps(dict(choice=choice, measurements=measurements,
        rule='highest mean silhouette IoU on identical non-keyframe predicted masks >=.8, excluding frozen image-only association flags',
        GT_used=False, full_clip_image_evidence=True), indent=2))
    predictions = []
    for arm, records in arms.items():
        for anchored in [False, True]:
            rows = json.loads(json.dumps(records))
            for seq in rows:
                alpha = alphas[seq['sequence']] if anchored else 1.
                assert alpha is not None and alpha > 0
                seq['object_scale'] *= alpha
                for frame in seq['frames']: frame['t_obj2cam_m'] = [alpha*x for x in frame['t_obj2cam_m']]
                seq['source']['predicted_human_alpha'] = alpha
            p = out/f'{arm}_{"anchored" if anchored else "raw"}_PREDICTIONS.json'
            p.write_text(json.dumps(dict(schema='opencv-metric-object-predictions-v1', sequences=rows), indent=2))
            predictions.append(p)
    # Label data enters only these separate scorer subprocesses, after selection.
    refs = ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'
    requests = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    for p in predictions:
        subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(p),
            '--requests', str(requests), '--references', str(refs),
            '--out', str(p.with_name(p.name.replace('_PREDICTIONS', '_SCORES')))], check=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(source_sha256=sources, model=model,
        full_coverage_frames_per_arm=384, all_arms_retained=True, exact_hosted_metrics=False,
        Hunyuan_uses_predicted_SAM_placement=True, GT_used_by_inference=False), indent=2))


if __name__ == '__main__': main()
