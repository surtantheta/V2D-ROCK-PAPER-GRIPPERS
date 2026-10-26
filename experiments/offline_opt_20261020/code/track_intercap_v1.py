#!/usr/bin/env python3
"""Fresh FP trajectories for all eight frozen InterCap development takes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from labeled_object_scorer import POSE_CONVENTION
from model_source_guard import verify_model_group

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    model = verify_model_group('FoundationPose'); runtime = ROOT/'labeled_intercap_v1'
    manifest = runtime/'FP_RUNTIME.json'; data = json.loads(manifest.read_text())
    assert data['complete_take_coverage'] and len(data['records']) == 8
    assert sum(len(x['source_paths']) for x in data['records']) == 246
    assert not data['GT_inputs_provided'] and not data['reserve_takes_included']
    env = os.environ.copy()
    if torch.cuda.get_device_capability() == (7, 5): env['PYTHONPATH'] = str(ROOT/'deps/sm75')
    records = []; sources = {str(p): sha(p) for p in [Path(__file__), manifest, ROOT/'code/labeled_track_fp_snapshot_v1.py']}
    for item in data['records']:
        assert item['sequence'] == item['take']; directory = Path(item['directory']); mesh = Path(item['mesh_path'])
        assert sha(mesh) == item['mesh_sha256']; track = runtime/'tracks'/item['name']
        subprocess.run([PY, str(ROOT/'code/labeled_track_fp_snapshot_v1.py'), '--frames', item['frames'],
            '--masks', item['masks'], '--human', item['depth'], '--object', str(directory), '--out', str(track)],
            cwd='/home/tanuj/projects/CARI4D', env=env, check=True)
        poses_path = track/'poses.npz'; poses = np.load(poses_path, allow_pickle=False)['T_obj2cam']
        n = len(item['native_frame_numbers']); assert poses.shape == (n, 4, 4) and np.isfinite(poses).all()
        assert item['native_frame_numbers'] == [int(Path(p).stem) for p in item['source_paths']]
        assert np.all(np.diff(item['native_frame_numbers']) == 5)
        assert np.allclose(poses[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-5)
        assert np.all(np.linalg.det(poses[:, :3, :3]) > .99999) and (poses[:, 2, 3] > 0).all()
        kf_path = directory/'keyframe.json'; kf = json.loads(kf_path.read_text())
        for p in [poses_path, mesh, kf_path, Path(item['depth'])/'RUN.json', Path(item['masks'])/'MASKS.json']:
            sources[str(p)] = sha(p)
        records.append(dict(sequence=item['take'], camera=1, mesh_path=str(mesh), mesh_sha256=sha(mesh),
            mesh_units='metres', object_scale=1., pose_convention=POSE_CONVENTION,
            origin_convention=kf['origin_convention'], origin_correspondence_to_gt='unverified',
            source=dict(generator='fresh SAM3D; RGB-predicted MoGe K/depth; actual kit4096 budget',
                poses_path=str(poses_path), poses_sha256=sha(poses_path), runtime_name=item['name'],
                keyframe_local_index=kf['keyframe'], GT_inputs_provided=False, native_CAD_input=False,
                reserve_take=False, nominal_source_fps=30., source_frame_cadence=5,
                K_depth_scale='uncalibrated RGB prediction; no native intrinsics or human scale yet'),
            frames=[dict(video_frame_index=int(frame), R_obj2cam=poses[i, :3, :3].tolist(),
                         t_obj2cam_m=poses[i, :3, 3].tolist()) for i, frame in enumerate(item['native_frame_numbers'])]))
        print('tracked', item['take'], n, 'native frames', flush=True)
    expected = {x['take']: x['native_frame_numbers'] for x in data['records']}
    actual = {x['sequence']: [f['video_frame_index'] for f in x['frames']] for x in records}
    assert actual == expected
    out = runtime/'baseline'; out.mkdir(exist_ok=False); pred = out/'foundationpose_raw_PREDICTIONS.json'
    pred.write_text(json.dumps(dict(schema='opencv-metric-object-predictions-v1', sequences=records), indent=2, allow_nan=False))
    (runtime/'TRACKING_COMPLETE.json').write_text(json.dumps(dict(prediction=str(pred), prediction_sha256=sha(pred),
        source_sha256=sources, model=model, frames=246, takes=8, exact_requested_coverage=True,
        GT_inputs_provided=False, reserve_takes_included=False,
        cadence='five native source frames per retained image;30fps nominal, not measured hardware timestamps'), indent=2))


if __name__ == '__main__': main()
