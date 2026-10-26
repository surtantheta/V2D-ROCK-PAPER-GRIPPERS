#!/usr/bin/env python3
"""Scorer-only InterCap first-frame HUMAN Sim3; no object GT or ICP.

Certified native camera SMPLH6890 -> official barycentric MHR18439 target.
Reads predicted body local frame zero only; requires all eight cases complete.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
KIT = PROJECT/'data/assets/submission_kit/v2d_submission_kit'
MAPS = PROJECT/'third_party/MHR/tools/mhr_smpl_conversion/assets'
NATIVE = ROOT/'intercap_native_contract_v1'
MODELS = Path('/rhome/tanuj/CONTHO_base_data/human_models/smplh')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mapping(path, face_count, vertices):
    source = np.load(path, allow_pickle=False)
    ids = source['triangle_ids']; weights = source['baryc_coords']
    assert ids.shape == (vertices,) and weights.shape == (vertices, 3)
    assert ids.min() >= 0 and ids.max() < face_count
    assert np.isfinite(weights).all() and weights.min() > -1e-10
    assert np.max(abs(weights.sum(1)-1)) < 1e-12
    return ids, weights


def transfer(vertices, faces, asset):
    ids, weights = asset
    return np.einsum('vkd,vk->vd', vertices[faces[ids]], weights)


def output_transform(transform):
    scale, row_rotation, translation = transform
    return dict(scale=float(scale), R=np.asarray(row_rotation).T.tolist(),
        t=np.asarray(translation).tolist(), convention='X_GT = scale * (X_pred @ R.T) + t')


def fit_checked(predicted, target, fit, apply):
    transform = fit(predicted, target)
    result = output_transform(transform)
    rotation = np.asarray(result['R'])
    assert result['scale'] > 0 and np.isfinite(result['scale'])
    assert np.isfinite(rotation).all() and np.isfinite(result['t']).all()
    assert np.max(abs(rotation@rotation.T-np.eye(3))) < 1e-10
    assert abs(np.linalg.det(rotation)-1) < 1e-10
    residual = np.linalg.norm(apply(predicted, transform)-target, axis=1)
    return result, residual


def self_test():
    sys.path.insert(0, str(KIT))
    from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform
    rng = np.random.default_rng(2110); points = rng.normal(size=(200, 3))
    angle = .37; c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    target = 1.73*points@rotation.T+np.array([.13, -.07, .2])
    transform, residual = fit_checked(points, target, fit_similarity_transform, apply_similarity_transform)
    assert residual.max() < 1e-12
    assert abs(transform['scale']-1.73) < 1e-12
    assert np.max(abs(np.asarray(transform['R'])-rotation)) < 1e-12
    # Check pose composition with anisotropic local scale and nonidentity object R.
    obj_rotation = rotation.T; obj_scale = np.array([.3, .7, 1.2]); obj_t = [.1, -.2, 3.]
    direct = 1.73*((points*obj_scale)@obj_rotation.T+obj_t)@rotation.T+transform['t']
    composed = (points*(1.73*obj_scale))@(rotation@obj_rotation).T+1.73*(rotation@obj_t)+transform['t']
    maximum = float(np.max(abs(direct-composed)))
    assert maximum < 1e-12
    return dict(known_Sim3_recovery_max_error_m=float(residual.max()),
        anisotropic_object_pose_composition_max_error_m=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'intercap_human_alignment_sidecars_v1')
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test_only:
        print(json.dumps(tests)); return
    assert not args.out.exists(), 'output must be new'
    requests_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    runtime_path = ROOT/'labeled_intercap_v1/RUN_INPUTS.json'
    body_result_path = ROOT/'labeled_intercap_v1/human_scale/RESULT.json'
    requests = json.loads(requests_path.read_text())['InterCap']
    runtime = json.loads(runtime_path.read_text())
    assert not runtime['GT_fields_included'] and not runtime['native_CAD_input'] and not runtime['reserve_takes_included']
    items = {r['take']: r for r in runtime['items']}
    assert len(requests) == len(items) == 8 and sum(len(r['native_frame_numbers']) for r in requests) == 246
    assert {r['take'] for r in requests} == set(items)
    # Do not create any output before ALL exact first-frame predictions are present.
    for request in requests:
        assert request['native_frame_numbers'][0] == 0 and request['camera'] == 1
        path = ROOT/'labeled_intercap_v1/human_scale'/items[request['take']]['name']/'body_000000.npz'
        assert path.is_file(), f'first predicted human missing; no later-frame fallback: {path}'
    body_result = json.loads(body_result_path.read_text())
    assert not body_result['GT_inputs_provided'] and not body_result['native_object_geometry_provided']
    assert body_result['gates']['authoritative_model_head_topology']['kit_faces_exactly_equal']
    assert len(body_result['records']) == 8
    assert {r['sequence'] for r in body_result['records']} == {x['name'] for x in items.values()}
    native_human_path = NATIVE/'FIRST_FRAME_NATIVE_HUMANS.json'
    decoder_path = NATIVE/'HUMAN_DECODER_CHECKS.json'
    vertices_path = NATIVE/'FIRST_FRAME_GT_SMPLH_VERTICES.npz'
    native_humans = json.loads(native_human_path.read_text())
    assert native_humans['scorer_only'] and not native_humans['native_object_fields_included']
    native_lookup = {r['take']: r for r in native_humans['records']}
    assert set(native_lookup) == set(items)
    proof = json.loads(decoder_path.read_text())
    assert proof['GT_human_only'] and not proof['native_object_fields_used'] and proof['primary_flat_hand_mean'] is False
    positive_checks = [r for r in proof['records'] if not r['flat_hand_mean']]
    assert {r['take'] for r in positive_checks} == set(items)
    assert max(r['max_joint_error_m'] for r in positive_checks) < 1e-5
    sources = {}
    def pin(path):
        sources[str(Path(path).resolve())] = sha(path)
    for provenance in [proof, body_result, body_result['model_provenance']]:
        for path, expected in provenance['source_sha256'].items():
            assert sha(path) == expected, f'human source guard changed: {path}'
            sources[str(Path(path).resolve())] = expected
    for path in [Path(__file__), requests_path, runtime_path, body_result_path,
                 native_human_path, decoder_path, vertices_path,
                 MAPS/'smpl2mhr_mapping.npz', MAPS/'mhr2smpl_mapping.npz',
                 PROJECT/'models/mhr/mhr_faces_lod1.npy', KIT/'v2dlb/mhr_metrics.py']:
        pin(path)
    # Faces only: body geometry/pose remains the independently certified native cache.
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    import torch
    import smplx
    torch.set_num_threads(2)
    smpl_faces = np.asarray(smplx.SMPLH(str(MODELS), gender='neutral', use_pca=False,
        flat_hand_mean=False).cpu().faces, np.int64)
    pin(inspect.getsourcefile(smplx.SMPLH))
    mhr_faces = np.load(PROJECT/'models/mhr/mhr_faces_lod1.npy', allow_pickle=False)
    assert smpl_faces.max() == 6889 and mhr_faces.max() == 18438
    forward = mapping(MAPS/'smpl2mhr_mapping.npz', len(smpl_faces), 18439)
    reverse = mapping(MAPS/'mhr2smpl_mapping.npz', len(mhr_faces), 6890)
    native_vertices = np.load(vertices_path, allow_pickle=False)
    assert set(native_vertices.files) == {r['name'] for r in items.values()}
    from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform
    records = []
    for request in requests:
        take = request['take']; item = items[take]; native = native_lookup[take]
        assert native['native_frame'] == request['native_frame_numbers'][0] == item['native_frame_numbers'][0] == 0
        assert item['source_paths'][0] == request['rgb_paths'][0]
        assert int(Path(item['source_paths'][0]).stem) == 0
        human_path = ROOT/'labeled_intercap_v1/human_scale'/item['name']/'body_000000.npz'
        human = np.load(human_path, allow_pickle=False)
        predicted = human['vertices_cam'].astype(np.float64)
        assert predicted.shape == (18439, 3) and np.isfinite(predicted).all() and (predicted[:, 2] > 0).all()
        depth_run_path = ROOT/'labeled_intercap_v1/depth'/item['name']/'RUN.json'
        depth_run = json.loads(depth_run_path.read_text())
        assert np.array_equal(human['K'], np.asarray(depth_run['K_fixed_source'], np.float32))
        gt = native_vertices[item['name']].astype(np.float64)
        assert gt.shape == (6890, 3) and np.isfinite(gt).all() and (gt[:, 2] > 0).all()
        target = transfer(gt, smpl_faces, forward)
        primary, residual = fit_checked(predicted, target, fit_similarity_transform, apply_similarity_transform)
        reverse_transform, _ = fit_checked(transfer(predicted, mhr_faces, reverse), gt,
            fit_similarity_transform, apply_similarity_transform)
        offset = np.array([.13, -.07, .2]); factor = 1.73
        equivariance = float(abs(transfer(factor*gt+offset, smpl_faces, forward)-(factor*target+offset)).max())
        assert equivariance < 1e-11
        for path in [human_path, depth_run_path]:
            pin(path)
        records.append(dict(sequence=take, name=item['name'], camera=1,
            RGB_frame=0, local_frame_index=0, GT_frame=0,
            GT_annotation_id=native['annotation_id'], GT_image_id=native['image_id'],
            gender=native['smpl_param']['gender'], predicted_body_path=str(human_path),
            predicted_body_sha256=sha(human_path), native_human_vertices_path=str(vertices_path),
            native_human_vertices_key=item['name'], primary_transform=primary,
            reverse_topology_diagnostic_transform=reverse_transform,
            primary_correspondence_count=18439,
            primary_human_fit_mean_vertex_residual_cm=float(100*residual.mean()),
            primary_human_fit_p95_vertex_residual_cm=float(100*np.percentile(residual, 95)),
            mapping_affine_equivariance_max_error_m=equivariance,
            units='metres; native camera1 and predicted OpenCV camera',
            native_object_GT_fields_accessed=False))
        print(take, 'scale', primary['scale'], 'human residual cm', records[-1]['primary_human_fit_mean_vertex_residual_cm'], flush=True)
    assert len(records) == 8
    report = dict(schema='human-only-first-frame-sim3-sidecars-v1', dataset='InterCap',
        records=records, transform_convention='X_GT = scale * (X_pred @ R.T) + t',
        primary_policy='exact first source frame0; certified native SMPLH6890 to MHR18439 official barycentric map; exact kit fit',
        fit_function=str(KIT/'v2dlb/mhr_metrics.py')+':fit_similarity_transform',
        object_pose_composition='object_scale_new=s*object_scale_old; R_obj_new=R_align@R_obj_old; t_new=s*(R_align@t_old)+t_align',
        single_transform_per_clip=True, GT_human_only_for_scoring=True,
        native_object_GT_fields_accessed=False, ICP_used=False, exact_hosted_alignment=False,
        reserve_samples_included=False, surrogate_warning='InterCap native SMPLH GT mapped to MHR is a human-correspondence surrogate, not native challenge MHR GT. Never feed this scorer-only sidecar into inference.',
        self_test=tests, source_sha256=sources)
    args.out.mkdir()
    (args.out/'TRANSFORMS.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('saved', args.out/'TRANSFORMS.json', flush=True)


if __name__ == '__main__':
    main()
