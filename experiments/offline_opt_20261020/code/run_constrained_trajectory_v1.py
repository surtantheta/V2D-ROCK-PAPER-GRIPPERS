#!/usr/bin/env python3
"""RGB-derived observations and frozen five-arm trajectory experiment; no GT."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh

import constrained_trajectory_v1 as core

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample_surface(mesh, count, seed):
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    triangle = v[f]; area = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]), axis=1)/2
    rng = np.random.default_rng(seed); ids = rng.choice(len(f), count, p=area/area.sum())
    a, b = rng.random((2, count)); s = np.sqrt(a)
    bary = np.c_[1-s, s*(1-b), s*b]
    return np.einsum('nk,nkd->nd', bary, triangle[ids])


def prepare(item, prediction, depthdir, maskpath, contact_record, timestamps, sources):
    cfg = core.PROTOCOL; meshpath = Path(prediction['mesh_path']); assert sha(meshpath) == prediction['mesh_sha256']
    mesh = trimesh.load(meshpath, process=False); assert np.asarray(mesh.vertices).shape == (4096, 3)
    surface = sample_surface(mesh, cfg['surface_samples'], cfg['surface_seed'])*prediction['object_scale']
    rows = json.loads(maskpath.read_text())['rows']; runpath = depthdir/'RUN.json'; run = json.loads(runpath.read_text())
    width, height = run['model_wh']; source_w, source_h = run['source_wh']
    k = np.diag([width/source_w, height/source_h, 1.]) @ np.asarray(run['K_fixed_source'])
    rotations = np.array([f['R_obj2cam'] for f in prediction['frames']]); translations = np.array([f['t_obj2cam_m'] for f in prediction['frames']])
    assert len(rows) == len(translations) == len(item['source_paths']) == len(timestamps)
    assert np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    raw_path = prediction['source'].get('association_tracking', {}).get('poses_path')
    poses_path = Path(raw_path) if raw_path else ROOT/'labeled_predictions_v2/tracks'/item['name']/'poses.npz'
    track = np.load(poses_path, allow_pickle=False)
    quality = np.nan_to_num(track['iou'], nan=.1).clip(.1, 1.)
    quality = np.where([r['selected'] and r['score'] >= cfg['mask_confidence'] for r in rows], quality, .1)
    # Depth is in the same uncalibrated gauge as the generated mesh/FP pose.
    # object_scale here is the already-applied constant predicted-human alpha.
    alpha = float(prediction['source']['human_anchor']['alpha'])
    assert np.isclose(alpha, prediction['object_scale'])
    assert np.allclose(track['T_obj2cam'][:, :3, :3], rotations, rtol=1e-12, atol=1e-12)
    assert np.allclose(alpha*track['T_obj2cam'][:, :3, 3], translations, rtol=1e-12, atol=1e-12)
    tree = cKDTree(surface); clouds = []; confidence = []; contacts = {}; audit = []
    contact_frames = {x['local_frame_index']: x for x in contact_record['frames']}
    for p in [meshpath, maskpath, runpath, poses_path]: sources[str(p)] = sha(p)
    for i, row in enumerate(rows):
        mp = Path(row['mask']); pp = Path(row['person_mask']); dp = depthdir/f'{i:06d}.npz'
        assert sha(mp) == row['mask_sha256'] and sha(pp) == row['person_mask_sha256']
        for p in [mp, pp, dp]: sources[str(p)] = sha(p)
        obj = np.asarray(Image.open(mp).resize((width, height), Image.Resampling.NEAREST)) > 127
        person = np.asarray(Image.open(pp).resize((width, height), Image.Resampling.NEAREST)) > 127
        depth = np.asarray(np.load(dp)['depth'], float)*alpha
        assert depth.shape == (height, width)
        admissible = ndimage.binary_erosion(obj, iterations=1) & ~ndimage.binary_dilation(person, iterations=1)
        admissible &= np.isfinite(depth) & (depth > 0)
        enabled = row['selected'] and row['score'] >= cfg['mask_confidence']
        if enabled and admissible.any():
            z = depth[admissible]; median = np.median(z); mad = np.median(abs(z-median))*1.4826
            admissible &= abs(depth-median) <= max(.05, 3*mad)
        else: admissible[:] = False
        y, x = np.nonzero(admissible); eligible_pixels = len(x)
        if len(x) >= cfg['minimum_depth_points']:
            rng = np.random.default_rng(1000+i)
            keep = rng.choice(len(x), min(len(x), cfg['max_depth_points']), replace=False); x, y = x[keep], y[keep]
            z = depth[y, x]
            points = np.c_[((x+.5-k[0, 2])/k[0, 0])*z, ((y+.5-k[1, 2])/k[1, 1])*z, z]
        else: points = np.empty((0, 3))
        clouds.append(points); confidence.append(float(row['score']) if enabled else 0.)
        hand_audit = []
        if enabled and i in contact_frames and obj.any():
            cf = contact_frames[i]; tp = Path(cf['tips_path']); assert sha(tp) == cf['tips_sha256']; sources[str(tp)] = sha(tp)
            tips = np.load(tp, allow_pickle=False); positions = tips['fingertips_cam_m']; names = tips['tip_names'].tolist()
            assert positions.shape == (10, 3) and np.isfinite(positions).all()
            assert np.allclose(tips['K'], run['K_fixed_source'])
            assert cf['video_frame_index'] == prediction['frames'][i]['video_frame_index']
            # Predicted human points are already metric camera coordinates: no alpha.
            uv = positions[:, :2]/positions[:, 2:] @ np.diag([k[0, 0], k[1, 1]]) + k[:2, 2]
            distance_2d = ndimage.distance_transform_edt(~obj)
            for side in ['l', 'r']:
                kept = []
                for j, name in enumerate(names):
                    if not name.startswith(side) or positions[j, 2] <= 0: continue
                    u, v = np.floor(uv[j]).astype(int)
                    if not (0 <= u < width and 0 <= v < height): continue
                    distance = float(tree.query((positions[j]-translations[i]) @ rotations[i], workers=1)[0])
                    if distance_2d[v, u] <= cfg['contact_mask_distance_model_px'] and distance <= cfg['initial_contact_surface_gate_m']:
                        kept.append(j)
                if kept:
                    contacts.setdefault(i, []).append(positions[kept].copy())
                    hand_audit.append(dict(side=side, tips=[names[j] for j in kept], points_cam_m=positions[kept].tolist()))
        audit.append(dict(local_frame_index=i, video_frame_index=prediction['frames'][i]['video_frame_index'],
            selected=bool(enabled), depth_pixels_after_mask_and_robust_z=eligible_pixels,
            sampled_depth_points=len(points), active_hands=hand_audit))
    return dict(surface=surface, rotations=rotations, baseline=translations, clouds=clouds,
        quality=quality, depth_confidence=np.array(confidence), times=np.asarray(timestamps),
        cadence=float(np.median(np.diff(timestamps))), contacts=contacts,
        frame_ids=[f['video_frame_index'] for f in prediction['frames']], frame_step=1), audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=ROOT/'constrained_behave_v1')
    args = ap.parse_args(); gates = core.self_test(); args.out.mkdir(exist_ok=False)
    original = ROOT/'labeled_association_tracking_v1/comparison/foundationpose_association_PREDICTIONS.json'
    inputs = ROOT/'labeled_predictions_v2/RUN_INPUTS.json'; contact_path = ROOT/'labeled_contact_inputs_v1/CONTACT_INPUTS.json'
    request = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    baseline = json.loads(original.read_text()); items = {x['name']: x for x in json.loads(inputs.read_text())['items'] if x['dataset'] == 'BEHAVE'}
    contact = {x['sequence']: x for x in json.loads(contact_path.read_text())['records']}
    requests = {x['sequence']: x for x in json.loads(request.read_text())['BEHAVE']}
    sources = {str(p): sha(p) for p in [Path(__file__), Path(core.__file__), original, inputs, contact_path, request]}
    arms = dict(foundationpose=(False, False, False), temporal=(False, True, False),
        depth=(True, False, False), depth_temporal=(True, True, False), depth_temporal_contact=(True, True, True))
    outputs = {name: json.loads(json.dumps(baseline)) for name in arms}; summaries = []
    for sequence in baseline['sequences']:
        name = sequence['sequence']; item = items[name]; request_row = requests[name]
        assert [f['video_frame_index'] for f in sequence['frames']] == request_row['video_frame_indices']
        maskpath = Path(item['masks'])/'MASKS.json'
        if name == 'Date01_Sub01_stool_move': maskpath = ROOT/'labeled_association_tracking_v1/masks'/name/'MASKS.json'
        observation, audit = prepare(item, sequence, ROOT/'labeled_predictions_v1/depth'/name,
            maskpath, contact[name], request_row['timestamps_seconds'], sources)
        (args.out/f'{name}_OBSERVATIONS.json').write_text(json.dumps(audit, indent=2))
        summary = dict(sequence=name, frames=len(sequence['frames']), depth_frames=sum(len(x)>0 for x in observation['clouds']),
            contact_frames=sorted(observation['contacts']), contact_hands=sum(len(v) for v in observation['contacts'].values()), arms={})
        for arm, switches in arms.items():
            if arm == 'foundationpose':
                t = observation['baseline']; diagnostics = dict(unchanged=True)
            else:
                t, diagnostics = core.solve(**observation, use_depth=switches[0], use_temporal=switches[1], use_contact=switches[2])
            row = next(x for x in outputs[arm]['sequences'] if x['sequence'] == name)
            if arm != 'foundationpose':
                for i, frame in enumerate(row['frames']): frame['t_obj2cam_m'] = t[i].tolist()
                poses = np.tile(np.eye(4), (len(t), 1, 1))
                poses[:, :3, :3] = observation['rotations']; poses[:, :3, 3] = t
                poses_path = args.out/f'{name}_{arm}_POSES.npz'
                np.savez_compressed(poses_path, T_obj2cam=poses)
                row['source']['constrained_translation'] = dict(protocol=core.PROTOCOL, arm=arm,
                    original_prediction_sha256=sha(original), rotations_fixed=True, object_scale_fixed=True,
                    mesh_fixed=True, alpha_depth_geometry_translation_consistent=True,
                    body_tips_already_metric=True, contact_inputs_sha256=sha(contact_path),
                    GT_inputs_provided=False, whole_trajectory_joint_quadratic=True)
                row['source']['poses_path'] = str(poses_path)
                row['source']['poses_sha256'] = sha(poses_path)
                row['source']['poses_units'] = 'metres after fixed predicted-human alpha'
                row['source']['optimizer'] = arm
            summary['arms'][arm] = diagnostics
        summaries.append(summary); print(name, 'depth frames', summary['depth_frames'], 'contact hands', summary['contact_hands'], flush=True)
    records = []
    for name, data in outputs.items():
        assert sum(len(s['frames']) for s in data['sequences']) == 384
        p = args.out/f'{name}_PREDICTIONS.json'; p.write_text(json.dumps(data, indent=2, allow_nan=False))
        records.append(dict(arm=name, prediction=str(p), prediction_sha256=sha(p)))
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(protocol=core.PROTOCOL, gates=gates,
        sequences=summaries, predictions=records, source_sha256=sources, full_frames=384,
        GT_inputs_provided=False, post_failure_engineering_development=True,
        contact_is_proximity_proxy_not_PEN=True), indent=2, allow_nan=False))


if __name__ == '__main__': main()
