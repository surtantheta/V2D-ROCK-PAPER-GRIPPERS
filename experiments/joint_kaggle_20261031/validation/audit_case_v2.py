#!/usr/bin/env python3
"""Independent full-frame challenge hands, SDF, merit and export audit.

No optimizer is imported. The numerical replay helpers come from the separately
authored and already exercised independent local audit. No GT is accessed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/tanuj/V2D_challenge')
PRIOR = ROOT.parent / 'physical_refinement_20261031'
BASELINE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
LARGE = Path('/rhome/tanuj/v2d_joint_kaggle_20261031')
KIT = OLD / 'data/assets/submission_kit/v2d_submission_kit'
INDEPENDENT = ROOT.parent / 'validation_20261031/audit_joint_full_v3.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, ('source/artifact changed', path)


def run(args):
    assert os.environ.get('PBS_JOBID'), 'Production validation requires PBS'
    ep = args.episode
    assert 0 <= ep < 30 and not args.out.exists()
    manifest_path = ROOT / 'FIT_INPUTS_V2.json'
    manifest = read(manifest_path)
    verify(manifest['source_sha256'])
    verify(manifest['case_sha256'][str(ep)])
    cfg = read(PRIOR / 'JOINT_INPUTS_V3.json')['config']
    if 'config' in manifest:
        assert manifest['config'] == cfg
    metadata = manifest['metadata'][str(ep)]
    folder = LARGE / 'episodes_v2' / f'episode_{ep:06d}'
    complete_path = folder / 'COMPLETE.json'
    complete = read(complete_path)
    assert complete['status'] == 'CHALLENGE_JOINT_EPISODE_COMPLETE'
    assert complete['episode'] == ep and complete['cases'] == 1
    assert complete['input_manifest_sha256'] == sha(manifest_path)
    assert complete['all_5840_hand_points_all_frames'] and not complete['GT_used_for_inference'] and not complete['GT_read']
    verify(complete['output_sha256'])
    verify(complete['auxiliary_output_sha256'])
    sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
    replay = load(INDEPENDENT, 'independent_local_joint_replay_helpers')
    import torch
    import numba
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(2)
    pen = load(KIT / 'metric_code/track_1/PEN.py', 'independent_challenge_joint_pen')
    assert pen._PENETRATION_KERNELS is not None
    sys.path.insert(0, str(KIT))
    from v2dlb.mesh_budget import budget_mesh
    hand_path = OLD / 'experiments/post_feedback_20261026/human_penetration_audit/run_v2_cari4d/HAND_VERTEX_IDS.npy'
    hand = np.load(hand_path, allow_pickle=False)
    assert len(hand) == 5840
    model_path = OLD / 'models/mhr/mhr_model.pt'
    model = torch.jit.load(str(model_path), map_location='cpu').eval()
    stem = f'episode_{ep:06d}'
    source = dict(np.load(BASELINE / (stem + '.npz'), allow_pickle=False))
    candidate_path = folder / 'submission' / (stem + '.npz')
    candidate = dict(np.load(candidate_path, allow_pickle=False))
    state_path = folder / (stem + '_PHYSICAL_STATE.npz')
    state = dict(np.load(state_path, allow_pickle=False))
    diagnostic_path = folder / (stem + '_JOINT.json')
    diagnostic = read(diagnostic_path)
    diag = diagnostic['diagnostics']
    assert diag['config'] == cfg and diagnostic['GT_used'] is False
    row = complete['records'][0]
    n = len(source['pose'])
    assert n == complete['frames'] == diagnostic['frames'] == row['frames'] == metadata['frames']
    assert metadata['frame_ids'] == list(range(n))
    assert np.allclose(metadata['times'], np.arange(n) / 30., atol=1e-10, rtol=0)
    ids = np.arange(n, dtype=np.int64)
    assert np.array_equal(state['native_frame_ids'], ids) and np.array_equal(state['hand_vertex_ids'], hand)
    assert set(candidate) == set(source)
    for name in source:
        assert np.isfinite(candidate[name]).all()
        if name != 'object_translation':
            assert source[name].shape == candidate[name].shape and source[name].dtype == candidate[name].dtype
            assert source[name].tobytes() == candidate[name].tobytes(), ('unexpected changed field', ep, name)
    mesh_path = folder / 'submission' / (stem + '_object.ply')
    assert sha(mesh_path) == sha(BASELINE / (stem + '_object.ply'))
    humans = replay.decode(model, source, torch)
    hands = humans[:, hand]
    hand_error = float(np.max(np.abs(hands - state['hands_scene_m'])))
    assert hand_error < 1e-12
    assert np.array_equal(source['object_translation'], state['baseline_translation'])
    assert np.array_equal(candidate['object_translation'], state['candidate_translation'])
    delta = candidate['object_translation'] - source['object_translation']
    maximum = float(np.linalg.norm(delta, axis=1).max())
    assert maximum <= cfg['maximum_correction_m'] + 1e-12 and abs(maximum - diag['maximum_correction_m']) < 1e-12
    v, f = budget_mesh(BASELINE / (stem + '_object.ply'), 4096, 4096)
    rotation = candidate['object_rotation']
    scale = float(candidate['object_scale'])
    assert scale > 0 and rotation.shape == (n, 3, 3)
    assert np.max(abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3))) < 1e-6
    assert np.max(abs(np.linalg.det(rotation) - 1)) < 1e-6
    radius = cfg['clearance_margin_m'] + cfg['maximum_correction_m']
    initial = replay.signed(hands, source['object_translation'], rotation, scale, v, f, pen, radius)
    final = replay.signed(hands, candidate['object_translation'], rotation, scale, v, f, pen, radius)
    sdf_error = max(float(np.max(abs(initial - state['initial_signed_distance_scene_m']))),
                    float(np.max(abs(final - state['final_signed_distance_scene_m']))))
    assert sdf_error < 1e-9
    counts = (initial < cfg['clearance_margin_m']).sum(axis=1)
    normalizers = np.maximum(counts, len(hand) / cfg['reference_hand_sampling_density'])
    assert np.array_equal(counts, diag['baseline_active_counts'])
    assert np.array_equal(normalizers, diag['frozen_perframe_normalizers'])
    initial_cost = replay.merit(np.zeros_like(delta), source['object_translation'], ids, initial, normalizers, cfg)
    final_cost = replay.merit(delta, source['object_translation'], ids, final, normalizers, cfg)
    assert abs(initial_cost - diag['initial_merit']) < 1e-8 and abs(final_cost - diag['final_merit']) < 1e-8
    assert final_cost <= initial_cost + 1e-10
    assert diag['time']['native_stride'] == 1 and diag['time']['eligible_triples'] == n - 2
    before, after = np.maximum(-initial, 0), np.maximum(-final, 0)
    assert np.square(after).mean() <= np.square(before).mean() + cfg['mean_penetration_squared_tolerance_m2'] + 1e-14
    assert after.max() <= before.max() + cfg['maximum_point_depth_increase_m'] + 1e-10
    change = dict(newly_penetrating_frames=int((~(before > 1e-6).any(axis=1) & (after > 1e-6).any(axis=1)).sum()),
                  cleared_frames=int(((before > 1e-6).any(axis=1) & ~(after > 1e-6).any(axis=1)).sum()),
                  maximum_depth_worsened_frames=int((after.max(axis=1) > before.max(axis=1) + 1e-8).sum()),
                  strict_new_inside_points=int(((before <= 0) & (after > 0)).sum()))
    assert change == diagnostic['penetration_changes'] == row['penetration_changes']
    triangles = v[f]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    mu = np.average(triangles.mean(axis=1), axis=0, weights=areas)
    centers = scale * np.einsum('tij,j->ti', rotation, mu) + source['object_translation']
    case = manifest['cases'][ep]
    camera_path = Path(case['paths']['poses'])
    camera = np.load(camera_path, allow_pickle=False)['K_model']
    camera = camera[0] if camera.ndim == 3 else camera
    assert np.allclose(camera, case['calibration']['MAIN_camera']['S6_K'], atol=1e-6, rtol=0)
    def project(points):
        h = points @ camera.T
        return h[:, :2] / h[:, 2, None]
    projected = float(np.linalg.norm(project(centers + delta) - project(centers), axis=1).max())
    visibility = diagnostic['final_image_depth_guard']
    assert projected <= 16 + 1e-10 and abs(projected - visibility['maximum_surface_centroid_projection_change_px']) < 1e-9
    minimum_z = min(float((scale * (v @ r.T) + t)[:, 2].min()) for r, t in zip(rotation, candidate['object_translation']))
    assert minimum_z > .001 and abs(minimum_z - visibility['minimum_fullmesh_camera_z_m']) < 1e-10
    assert not visibility['reasons']
    samples = {x['frame']['local_index']: x for x in case['samples']}
    for ob in visibility['observations']:
        i = ob['frame']
        assert i in samples and samples[i]['selected'] and samples[i]['state'] not in ['absent', 'uncertain', 'ambiguous']
        position = samples[i]['position']
        record = case['calibration']['frames'][position]
        expected = bool(case['calibration']['calibration_metric_diagnostic_eligible']
                        and record.get('calibrated_body_residual_cm') is not None
                        and record['calibrated_body_residual_cm']['mean'] <= 15 and record['body']['body_pixels'] >= 64)
        assert ob['metric_calibration_reliable'] == expected
        if not expected:
            assert ob['frozen_baseline_cloud_points'] == 0 and ob['baseline_masked_depth_residual_m'] is None
    obs = visibility['observations']
    if obs:
        baseline_iou = np.mean([x['baseline_iou'] for x in obs])
        candidate_iou = np.mean([x['candidate_iou'] for x in obs])
        assert baseline_iou < .1 or candidate_iou >= baseline_iou - .02 - 1e-12
        depths = [x for x in obs if x['baseline_masked_depth_residual_m'] is not None]
        if depths:
            assert np.mean([x['candidate_masked_depth_residual_m'] - x['baseline_masked_depth_residual_m'] for x in depths]) <= .01 + 1e-12
    verify(complete['output_sha256'])
    verify(complete['auxiliary_output_sha256'])
    guards = dict(manifest['source_sha256'])
    guards.update(manifest['case_sha256'][str(ep)])
    guards.update(complete['output_sha256'])
    guards.update(complete['auxiliary_output_sha256'])
    for path in [manifest_path, complete_path, INDEPENDENT, Path(__file__), hand_path, model_path]:
        guards[str(path)] = sha(path)
    receipt = dict(status='PASS_CHALLENGE_SMOKE_V2' if args.smoke else 'PASS_CHALLENGE_JOINT_EPISODE',
                   episode=ep, cases=1, frames=n, all_frames_all_5840_hand_points_checked=True,
                   producer_completion=str(complete_path), producer_completion_sha256=sha(complete_path),
                   input_manifest_sha256=sha(manifest_path), independent_source_sha256=sha(__file__),
                   decoded_hand_max_error_m=hand_error, independent_signed_distance_max_error_m=sdf_error,
                   initial_merit=initial_cost, final_merit=final_cost, maximum_correction_m=maximum,
                   maximum_projection_change_px=projected, min_full_budgeted_mesh_camera_z_m=minimum_z,
                   budgeted_mesh_vertices_array_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
                   budgeted_mesh_faces_array_sha256=hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(),
                   human_mesh_rotation_scale_bytes_exact=True, posthoc_smoothing=False,
                   penetration_changes=change, GT_loaded=False, GT_used_for_inference=False,
                   source_and_output_sha256=guards, CPU_job=os.environ['PBS_JOBID'],
                   audited_utc=datetime.now(timezone.utc).isoformat(),
                   limitations=['5840-point proxy differs from hidden hosted512 hand roles.',
                                'Image/depth guard input joins and scalar tests checked; numerical guard was unchanged from validated local source.',
                                'Objective/physical consistency is not a GT accuracy result.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['source_and_output_sha256', 'limitations']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        assert not args.smoke or args.episode == 12
        run(args)
    except BaseException as error:
        failure = args.out.with_suffix('.failed.json')
        if not failure.exists():
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(json.dumps(dict(status='NO_GO_CHALLENGE_JOINT_AUDIT_FAILED', episode=args.episode,
                error=repr(error), traceback=traceback.format_exc(), CPU_job=os.environ.get('PBS_JOBID')), indent=2) + '\n')
        raise
