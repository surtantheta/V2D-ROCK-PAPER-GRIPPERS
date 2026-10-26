#!/usr/bin/env python3
"""Independent CPU checks: actual FP, predicted-human alpha and human-only Sim3.

No native object GT is opened. Native human cache is scorer-only.
"""
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    track_receipt = json.loads((ROOT/'labeled_intercap_v1/TRACKING_COMPLETE.json').read_text())
    human_result = json.loads((ROOT/'labeled_intercap_v1/human_scale/RESULT.json').read_text())
    sim3_path = ROOT/'intercap_human_alignment_sidecars_v1/TRANSFORMS.json'
    alignment = json.loads(sim3_path.read_text())
    constrained = json.loads((ROOT/'constrained_intercap_v1/COMPLETE.json').read_text())
    hashes = {}
    for data in [track_receipt, human_result,
                 human_result['model_provenance'], alignment, constrained]:
        for path, expected in data['source_sha256'].items():
            assert path not in hashes or hashes[path] == expected
            hashes[path] = expected
    model_receipt_path = track_receipt['model']['model_source_receipt']
    assert sha(model_receipt_path) == track_receipt['model']['model_source_receipt_sha256']
    model_receipt = json.loads(Path(model_receipt_path).read_text())
    for path, data in model_receipt['groups']['FoundationPose'].items():
        hashes[path] = data['sha256']
    for path, expected in hashes.items():
        assert sha(path) == expected, path
    assert sha(track_receipt['prediction']) == track_receipt['prediction_sha256']
    runtime = json.loads((ROOT/'labeled_intercap_v1/RUN_INPUTS.json').read_text())
    items = {r['take']: r for r in runtime['items']}
    requests = {r['take']: r for r in json.loads((ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json').read_text())['InterCap']}
    raw = {r['sequence']: r for r in json.loads(Path(track_receipt['prediction']).read_text())['sequences']}
    alpha_lookup = {r['sequence']: r for r in human_result['records']}
    transforms = {r['sequence']: r for r in alignment['records']}
    assert set(items) == set(requests) == set(raw) == set(transforms) and len(raw) == 8
    assert alignment['GT_human_only_for_scoring'] and alignment['single_transform_per_clip']
    assert not alignment['native_object_GT_fields_accessed'] and not alignment['ICP_used'] and not alignment['exact_hosted_alignment']
    faces = np.load(PROJECT/'models/mhr/mhr_faces_lod1.npy')
    bary = np.array([(a, b, 1-a-b) for a in np.linspace(0, 1, 5)
        for b in np.linspace(0, 1, 5) if a+b <= 1+1e-9])
    alpha_checks = []; pose_checks = []
    for take, item in items.items():
        seq = raw[take]; declared = alpha_lookup[item['name']]
        assert seq['source']['runtime_name'] == item['name'] and seq['object_scale'] == 1.
        assert [f['video_frame_index'] for f in seq['frames']] == requests[take]['native_frame_numbers']
        poses = np.load(seq['source']['poses_path'])['T_obj2cam']
        assert poses.shape == (len(seq['frames']), 4, 4) and np.isfinite(poses).all()
        assert np.array_equal(poses[:, :3, :3], np.array([f['R_obj2cam'] for f in seq['frames']]))
        assert np.array_equal(poses[:, :3, 3], np.array([f['t_obj2cam_m'] for f in seq['frames']]))
        assert np.max(abs(poses[:, :3, :3]@poses[:, :3, :3].transpose(0, 2, 1)-np.eye(3))) < 1e-5
        assert np.min(np.linalg.det(poses[:, :3, :3])) > .99999 and np.min(poses[:, 2, 3]) > 0
        rows = json.loads((Path(item['masks'])/'MASKS.json').read_text())['rows']
        run = json.loads((Path(item['depth'])/'RUN.json').read_text())
        rw, rh = run['model_wh']; W, H = run['source_wh']
        ratios = []; accepted = 0
        for record in declared['frames']:
            i = record['frame']; bodypath = ROOT/'labeled_intercap_v1/human_scale'/item['name']/f'body_{i:06d}.npz'
            assert bodypath.exists()
            body = np.load(bodypath); vertices = body['vertices_cam'].astype(np.float64)
            assert vertices.shape == (18439, 3) and np.isfinite(vertices).all() and (vertices[:, 2] > 0).all()
            assert np.array_equal(body['K'], np.asarray(run['K_fixed_source'], np.float32))
            k = np.diag([rw/W, rh/H, 1.])@body['K']
            points = np.einsum('sk,fkd->fsd', bary, vertices[faces]).reshape(-1, 3)
            keep = np.isfinite(points).all(1)&(points[:, 2] > .001); points = points[keep]
            u = np.floor(points[:, 0]/points[:, 2]*k[0, 0]+k[0, 2]).astype(int)
            v = np.floor(points[:, 1]/points[:, 2]*k[1, 1]+k[1, 2]).astype(int)
            keep = (u >= 0)&(u < rw)&(v >= 0)&(v < rh)
            z = np.full((rh, rw), np.inf); np.minimum.at(z, (v[keep], u[keep]), points[keep, 2])
            person = cv2.imread(rows[i]['person_mask'], 0) > 127
            obj = cv2.imread(rows[i]['mask'], 0) > 127
            person = cv2.resize(person.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0
            obj = cv2.resize(obj.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
            interior = cv2.erode(np.isfinite(z).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            exclusion = cv2.dilate(obj, np.ones((5, 5), np.uint8)) > 0
            depth = np.load(Path(item['depth'])/f'{i:06d}.npz')['depth'].astype(float)
            good = interior&person&~exclusion&np.isfinite(depth)&(depth > 1e-3)
            ratio = z[good]/depth[good]; ratio = ratio[np.isfinite(ratio)&(ratio > 0)]
            assert len(ratio) == record['pixels']
            if len(ratio) >= 200:
                assert abs(np.median(ratio)-record['ratio_median']) < 1e-12
                ratios.append(ratio); accepted += 1
        recovered = float(np.median(np.concatenate(ratios)))
        assert abs(recovered-declared['alpha']) < 1e-12 and accepted == declared['accepted_frames']
        alpha_checks.append(dict(take=take, runtime_name=item['name'], alpha=recovered,
            accepted_frames=accepted, sampled_body_frames=len(declared['frames']),
            first_body_SHA256=sha(ROOT/'labeled_intercap_v1/human_scale'/item['name']/'body_000000.npz')))
        pose_checks.append(dict(take=take, frames=len(seq['frames']), exact_raw_JSON_NPZ_pose_match=True))
    # Independently reconstruct the MHR human targets and separate SVD fit, no kit fitter imported.
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    import torch
    import smplx
    torch.set_num_threads(2)
    f = smplx.SMPLH('/rhome/tanuj/CONTHO_base_data/human_models/smplh', gender='neutral', use_pca=False).faces
    mapdata = np.load(PROJECT/'third_party/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz')
    native = np.load(ROOT/'intercap_native_contract_v1/FIRST_FRAME_GT_SMPLH_VERTICES.npz')
    fits = []
    for take, item in items.items():
        record = transforms[take]; x = record['primary_transform']
        assert record['RGB_frame'] == record['GT_frame'] == record['local_frame_index'] == 0
        assert record['predicted_body_sha256'] == sha(record['predicted_body_path'])
        p = np.load(record['predicted_body_path'])['vertices_cam'].astype(np.float64)
        g = native[item['name']].astype(np.float64)
        g = (g[f[mapdata['triangle_ids']]]*mapdata['baryc_coords'][:, :, None]).sum(axis=1)
        pc = p-p.mean(0); gc = g-g.mean(0); u, singular, vh = np.linalg.svd(pc.T@gc)
        correction = np.diag([1., 1., np.linalg.det(u@vh)]); q = u@correction@vh
        scale = float(np.trace(np.diag(singular)@correction)/(pc*pc).sum())
        shift = g.mean(0)-scale*p.mean(0)@q
        error = max(abs(scale-x['scale']), np.max(abs(q.T-np.asarray(x['R']))), np.max(abs(shift-x['t'])))
        assert error < 1e-10
        fits.append(dict(take=take, scale=scale, max_frozen_transform_parameter_error=float(error),
            mean_human_fit_cm=float(100*np.linalg.norm(scale*p@q+shift-g, axis=1).mean())))
    arm_checks = []
    anchored = json.loads((ROOT/'labeled_intercap_v1/baseline/foundationpose_human_scale_PREDICTIONS.json').read_text())
    for row in constrained['predictions']:
        path = Path(row['prediction']); assert sha(path) == row['prediction_sha256']
        data = json.loads(path.read_text()); assert len(data['sequences']) == 8
        for seq in data['sequences']:
            take = seq['sequence']; old = raw[take]; a = alpha_lookup[items[take]['name']]['alpha']
            assert seq['object_scale'] == a and seq['mesh_path'] == old['mesh_path'] and seq['mesh_sha256'] == old['mesh_sha256']
            assert seq['source']['human_anchor']['alpha'] == a
            assert [f['video_frame_index'] for f in seq['frames']] == requests[take]['native_frame_numbers']
            assert np.array_equal([f['R_obj2cam'] for f in seq['frames']], [f['R_obj2cam'] for f in old['frames']])
            base = a*np.asarray([f['t_obj2cam_m'] for f in old['frames']])
            t = np.asarray([f['t_obj2cam_m'] for f in seq['frames']])
            if row['arm'] == 'foundationpose':
                assert np.array_equal(base, t) and data == anchored
            else:
                assert np.max(abs(t-base)) <= constrained['protocol']['translation_bound_m']+1e-8
        arm_checks.append(dict(arm=row['arm'], frames=sum(len(s['frames']) for s in data['sequences']),
            geometry_scale_rotations_fixed=True, native_frame_coverage_exact=True))
    output = dict(unique_source_hashes_passed=len(hashes), actual_raw_FP_pose_checks=pose_checks,
        independently_recomputed_predicted_alpha=alpha_checks,
        independent_cached_human_Sim3_fits=fits, constrained_arm_checks=arm_checks,
        source_frames=246, takes=8, native_object_GT_opened=False)
    dest = PROJECT/'validation/OPT_INTERCAP_COMPLETED_COMPONENTS_20261020.json'
    dest.write_text(json.dumps(output, indent=2, allow_nan=False)); print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
