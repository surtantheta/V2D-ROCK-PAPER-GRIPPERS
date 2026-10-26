#!/usr/bin/env python3
"""Full-cohort predicted-only association/shared-scale ablations; no GT imports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

import constrained_trajectory_v1 as previous
from run_constrained_intercap_v1 import sample_surface
import centroid_scale_trajectory_v2 as core

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
ARMS = ['baseline', 'identity', 'identity_temporal', 'fixedscale_geometry',
        'shared_scale', 'shared_scale_contact']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configuration(dataset):
    if dataset == 'InterCap':
        return dict(baseline=ROOT/'labeled_intercap_v1/baseline/foundationpose_human_scale_PREDICTIONS.json',
                    inputs=ROOT/'labeled_intercap_v1/RUN_INPUTS.json',
                    contacts=ROOT/'labeled_intercap_contact_inputs_v1/CONTACT_INPUTS.json',
                    depth=ROOT/'labeled_intercap_v1/depth', count=8, frames=246, step=5)
    return dict(baseline=ROOT/'labeled_association_tracking_v1/comparison/foundationpose_association_PREDICTIONS.json',
                inputs=ROOT/'labeled_predictions_v2/RUN_INPUTS.json',
                contacts=ROOT/'labeled_contact_inputs_v1/CONTACT_INPUTS.json',
                depth=ROOT/'labeled_predictions_v1/depth', count=4, frames=384, step=1)


def observe(sequence, item, depthdir, contact, times, step, sources):
    """Keep all accepted pixels fixed; no initial predicted-surface admission gate."""
    def pin(path, expected=None):
        path = Path(path); value = sources.setdefault(str(path), sha(path))
        if expected is not None:
            assert value == expected, str(path)
        return path
    meshpath = pin(sequence['mesh_path'], sequence['mesh_sha256'])
    mesh = trimesh.load(meshpath, process=False)
    assert np.asarray(mesh.vertices).shape == (4096, 3)
    assert np.asarray(mesh.faces).shape == (4096, 3)
    alpha = float(sequence['source']['human_anchor']['alpha'])
    assert np.isclose(sequence['object_scale'], alpha)
    surface = sample_surface(mesh, 4096, 271828)*alpha
    triangle = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    area = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0],
                                   triangle[:, 2]-triangle[:, 0]), axis=1)/2
    mu = np.average(triangle.mean(1), axis=0, weights=area)*alpha
    manifest = pin(sequence['source']['identity_mask_manifest'])
    rows = json.loads(manifest.read_text())['rows']
    runpath = pin(depthdir/'RUN.json'); run = json.loads(runpath.read_text())
    w, h = run['model_wh']; sw, sh = run['source_wh']
    k = np.diag([w/sw, h/sh, 1.]) @ np.asarray(run['K_fixed_source'])
    rotations = np.array([f['R_obj2cam'] for f in sequence['frames']])
    translation = np.array([f['t_obj2cam_m'] for f in sequence['frames']])
    assert len(rows) == len(translation) == len(item['source_paths']) == len(times)
    assert np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    trackpath = pin(sequence['source']['poses_path'], sequence['source']['poses_sha256'])
    track = np.load(trackpath, allow_pickle=False)
    assert np.allclose(track['T_obj2cam'][:, :3, :3], rotations, atol=1e-10, rtol=1e-10)
    assert np.allclose(alpha*track['T_obj2cam'][:, :3, 3], translation, atol=1e-10, rtol=1e-10)
    quality = np.nan_to_num(track['iou'], nan=.1).clip(.1, 1.)
    clouds = []; images = []; confidence = []; contacts = {}; audit = []
    contact_lookup = {x['local_frame_index']: x for x in contact['frames']}
    for i, row in enumerate(rows):
        mp = pin(row['mask'], row['mask_sha256'])
        pp = pin(row['person_mask'], row['person_mask_sha256'])
        dp = pin(depthdir/f'{i:06d}.npz')
        obj = np.asarray(Image.open(mp).resize((w, h), Image.Resampling.NEAREST)) > 127
        person = np.asarray(Image.open(pp).resize((w, h), Image.Resampling.NEAREST)) > 127
        depth = np.asarray(np.load(dp)['depth'], float)*alpha
        assert depth.shape == (h, w)
        enabled = bool(row['selected'] and row['score'] >= .8)
        if not enabled:
            quality[i] = .1
        # Person overlap may hide projected model pixels, but cannot delete
        # accepted positive object-mask evidence from the reverse coverage term.
        positive = obj
        observed = ndimage.binary_erosion(obj, iterations=1) & ~ndimage.binary_dilation(person, iterations=1)
        observed &= np.isfinite(depth) & (depth > 0)
        if enabled and observed.any():
            z = depth[observed]; med = np.median(z); mad = 1.4826*np.median(abs(z-med))
            observed &= abs(depth-med) <= max(.05, 3*mad)
        else:
            observed[:] = False
        y, x = np.nonzero(observed); eligible = len(x)
        rng = np.random.default_rng(1000+i)
        if len(x) >= 32:
            pick = rng.choice(len(x), min(len(x), 256), replace=False)
            x, y = x[pick], y[pick]; z = depth[y, x]
            cloud = np.c_[((x+.5-k[0, 2])/k[0, 0])*z,
                          ((y+.5-k[1, 2])/k[1, 1])*z, z]
        else:
            cloud = np.empty((0, 3))
        clouds.append(cloud); confidence.append(float(row['score']) if enabled else 0.)
        y, x = np.nonzero(positive if enabled else np.zeros_like(obj))
        if len(x):
            pick = rng.choice(len(x), min(len(x), 128), replace=False)
            maskuv = np.c_[x[pick]+.5, y[pick]+.5]
        else:
            maskuv = np.empty((0, 2))
        images.append(dict(Kmodel=k.copy(), maskuv=maskuv,
                           maskdistancefield=ndimage.distance_transform_edt(~(obj | person)),
                           confidence=float(row['score']) if enabled else 0., enabled=enabled))
        hands = []
        if enabled and i in contact_lookup and obj.any():
            record = contact_lookup[i]; tp = pin(record['tips_path'], record['tips_sha256'])
            tips = np.load(tp, allow_pickle=False)
            positions = tips['fingertips_cam_m']; names = tips['tip_names'].tolist()
            assert positions.shape == (10, 3) and np.isfinite(positions).all()
            assert np.allclose(tips['K'], run['K_fixed_source'])
            assert record['video_frame_index'] == sequence['frames'][i]['video_frame_index']
            distance = ndimage.distance_transform_edt(~obj)
            uv = positions[:, :2]/positions[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]
            for side in ['l', 'r']:
                keep = []
                for j, name in enumerate(names):
                    if not name.startswith(side) or positions[j, 2] <= 0:
                        continue
                    u, v = np.floor(uv[j]).astype(int)
                    if 0 <= u < w and 0 <= v < h and distance[v, u] <= 4.:
                        keep.append(j)
                if keep:
                    contacts.setdefault(i, []).append(positions[keep].copy())
                    hands.append(dict(side=side, tips=[names[j] for j in keep]))
        audit.append(dict(local_frame_index=i, video_frame_index=sequence['frames'][i]['video_frame_index'],
                          selected=enabled, eligible_depth_pixels=eligible,
                          sampled_depth_points=len(cloud), positive_mask_points=len(maskuv), active_hands=hands))
    return dict(surface_metric=surface, mu_metric=mu, rotations=rotations,
                baseline_translation=translation, clouds=clouds, quality=quality,
                depth_confidence=np.asarray(confidence), times=np.asarray(times),
                image_observations=images, contacts=contacts,
                frame_ids=[f['video_frame_index'] for f in sequence['frames']], frame_step=step), audit


def fit_arm(arm, obs, times, ids, step):
    """Fallback on declared numerical failures; contract/provenance errors abort."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if arm == 'identity_temporal':
                t, diagnostics = previous.solve(surface=obs['surface_metric'], rotations=obs['rotations'],
                    baseline=obs['baseline_translation'], clouds=obs['clouds'], quality=obs['quality'],
                    depth_confidence=obs['depth_confidence'], times=obs['times'],
                    cadence=float(np.median(np.diff(times))), contacts={},
                    use_depth=False, use_temporal=True, use_contact=False,
                    frame_ids=ids, frame_step=step)
                return t, 1., diagnostics
            return core.solve(**obs, allow_scale=arm != 'fixedscale_geometry',
                use_depth=True, use_temporal=True, use_contact=arm == 'shared_scale_contact')
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        failure = exc
    except ValueError as exc:
        if not any(message in str(exc).lower() for message in [
                'residuals are not finite', 'array must not contain infs or nans',
                'data must be finite', 'input contains nan', 'input contains inf']):
            raise
        failure = exc
    return obs['baseline_translation'].copy(), 1., dict(
        numerical_failure=dict(type=type(failure).__name__, message=str(failure)),
        valid_fit_numerics=False)


def export_validated(t, ratio, diagnostics, obs, vertices_metric):
    """Validate physical geometry; an arbitrary local origin may have negative Z."""
    baseline_z = (np.einsum('nij,mj->nmi', obs['rotations'], vertices_metric)
                  + obs['baseline_translation'][:, None, :])[..., 2]
    finite = bool(np.isfinite(t).all() and np.isfinite(ratio) and ratio > 0)
    objective_finite = bool(diagnostics.get('finite_final_objective', True))
    if 'final_objective' in diagnostics:
        objective_finite &= all(np.isfinite(x) for x in diagnostics['final_objective'].values())
    candidate_z = None
    if finite:
        with np.errstate(over='ignore', invalid='ignore'):
            candidate_z = (np.einsum('nij,mj->nmi', obs['rotations'], vertices_metric*ratio)
                           + t[:, None, :])[..., 2]
    geometry_finite = bool(candidate_z is not None and np.isfinite(candidate_z).all())
    minimum = float(np.min(candidate_z)) if geometry_finite else None
    valid = bool(finite and objective_finite and diagnostics.get('valid_fit_numerics', True)
                 and geometry_finite and minimum >= .02)
    reason = None if valid else (
        'solver or nonfinite numerical failure' if not (finite and objective_finite
            and diagnostics.get('valid_fit_numerics', True) and geometry_finite)
        else 'candidate mesh behind or within 2cm of camera')
    diagnostics['export_validation'] = dict(candidate_all_mesh_vertices_min_z_m=minimum,
        identity_all_mesh_vertices_min_z_m=float(np.min(baseline_z)),
        whole_case_identity_fallback=not valid, fallback_reason=reason, all_frames_retained=True)
    if not valid:
        t = obs['baseline_translation'].copy(); ratio = 1.
    diagnostics['exported_centroids_metric'] = (
        t + ratio*np.einsum('nij,j->ni', obs['rotations'], obs['mu_metric'])).tolist()
    diagnostics['exported_relative_scale'] = float(ratio)
    def finite_diagnostics(value):
        if isinstance(value, dict):
            return {key: finite_diagnostics(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [finite_diagnostics(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        return value
    return t, ratio, finite_diagnostics(diagnostics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'identity_scale_v2/optimized')
    args = ap.parse_args(); args.out.mkdir(exist_ok=False)
    requests_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    requests = json.loads(requests_path.read_text())
    sources = {str(p): sha(p) for p in [Path(__file__), Path(core.__file__), Path(previous.__file__),
                                       ROOT/'code/run_constrained_intercap_v1.py', requests_path]}
    reports = []
    for dataset in ['BEHAVE', 'InterCap']:
        cfg = configuration(dataset)
        replay_path = ROOT/f'identity_scale_v2/replay/{dataset}_PREDICTIONS.json'
        for p in [cfg['baseline'], cfg['inputs'], cfg['contacts'], replay_path]:
            sources[str(p)] = sha(p)
        baseline = json.loads(cfg['baseline'].read_text())
        replay = json.loads(replay_path.read_text())
        items = {x['take'] if dataset == 'InterCap' else x['name']: x
                 for x in json.loads(cfg['inputs'].read_text())['items'] if x['dataset'] == dataset}
        contacts = {x['sequence']: x for x in json.loads(cfg['contacts'].read_text())['records']}
        req = {x['take'] if dataset == 'InterCap' else x['sequence']: x for x in requests[dataset]}
        names = [x['sequence'] for x in replay['sequences']]
        assert len(names) == len(set(names)) == cfg['count']
        assert set(names) == set(items) == set(contacts) == set(req)
        assert {x['sequence'] for x in baseline['sequences']} == set(names)
        outputs = {arm: copy.deepcopy(baseline if arm == 'baseline' else replay) for arm in ARMS}
        summaries = []
        folder = args.out/dataset; folder.mkdir()
        for sequence in replay['sequences']:
            name = sequence['sequence']; item = items[name]
            ids = req[name]['native_frame_numbers' if dataset == 'InterCap' else 'video_frame_indices']
            assert [f['video_frame_index'] for f in sequence['frames']] == ids
            times = np.asarray(ids)/30. if dataset == 'InterCap' else np.asarray(req[name]['timestamps_seconds'])
            obs, audit = observe(sequence, item, cfg['depth']/item['name'], contacts[name], times, cfg['step'], sources)
            mesh = trimesh.load(sequence['mesh_path'], process=False)
            vertices_metric = np.asarray(mesh.vertices)*sequence['object_scale']
            (folder/f'{item["name"]}_OBSERVATIONS.json').write_text(json.dumps(audit, indent=2))
            summary = dict(sequence=name, frames=len(ids), arms={})
            for arm in ARMS[2:]:
                t, ratio, diagnostics = fit_arm(arm, obs, times, ids, cfg['step'])
                t, ratio, diagnostics = export_validated(t, ratio, diagnostics, obs, vertices_metric)
                row = next(x for x in outputs[arm]['sequences'] if x['sequence'] == name)
                row['object_scale'] = float(sequence['object_scale']*ratio)
                for i, frame in enumerate(row['frames']):
                    frame['t_obj2cam_m'] = t[i].tolist()
                poses = np.tile(np.eye(4), (len(t), 1, 1))
                poses[:, :3, :3] = obs['rotations']; poses[:, :3, 3] = t
                path = folder/f'{item["name"]}_{arm}_POSES.npz'
                np.savez_compressed(path, T_obj2cam=poses, relative_object_scale=ratio)
                row['source']['centroid_scale_v2'] = dict(arm=arm, relative_object_scale=float(ratio),
                    protocol=core.PROTOCOL if arm != 'identity_temporal' else previous.PROTOCOL,
                    area_centroid_metric=obs['mu_metric'].tolist(),
                    export_validation=diagnostics['export_validation'],
                    human_and_observed_depth_fixed_metric=True, rotation_fixed=True,
                    original_replay_prediction_sha256=sha(replay_path), GT_inputs=False)
                row['source'].update(poses_path=str(path), poses_sha256=sha(path),
                                     poses_units='camera metres; human alpha already applied', optimizer=arm)
                summary['arms'][arm] = diagnostics
            summaries.append(summary)
            print(dataset, name, 'finished', flush=True)
        records = []
        for arm, prediction in outputs.items():
            assert sum(len(x['frames']) for x in prediction['sequences']) == cfg['frames']
            path = folder/f'{arm}_PREDICTIONS.json'
            path.write_text(json.dumps(prediction, indent=2, allow_nan=False))
            records.append(dict(arm=arm, prediction=str(path), prediction_sha256=sha(path)))
        reports.append(dict(dataset=dataset, records=records, sequences=summaries,
                            frames_per_arm=cfg['frames'], takes_per_arm=cfg['count']))
    for path, expected in sources.items():
        assert sha(path) == expected, path
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(records=reports, arms=ARMS,
        source_sha256=sources, GT_inputs=False, reserve_included=False,
        full_frames_per_arm=630, development_after_failure_analysis=True), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
