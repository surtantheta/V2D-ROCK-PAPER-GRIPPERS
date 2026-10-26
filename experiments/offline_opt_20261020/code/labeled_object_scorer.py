#!/usr/bin/env python3
"""CPU BEHAVE/InterCap development object scorer; raw camera frame, no GT alignment.

Prediction schema: opencv-metric-object-predictions-v1, sequences=[{
 sequence, camera:1, mesh_path, mesh_sha256, mesh_units:'metres',
 pose_convention:POSE_CONVENTION, object_scale:1.0, origin_convention:'...',
 origin_correspondence_to_gt:'unverified'|'verified', source:{...},
 frames:[{video_frame_index, R_obj2cam:[[...]], t_obj2cam_m:[...]}]}].
Verified origin correspondence additionally requires origin_correspondence_source.
Source is a declaration/provenance record, not automatic proof of GT-free inference.
"""
import argparse
import hashlib
import json
import mmap
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

POSE_CONVENTION = 'X_cam = object_scale * (V_local @ R_obj2cam.T) + t_obj2cam_m'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{label}: expected finite {shape}, got {result.shape}')
    return result


def validate_rotation(value):
    rotation = finite_array(value, (3, 3), 'R_obj2cam')
    if np.max(np.abs(rotation @ rotation.T - np.eye(3))) > 1e-5 or abs(np.linalg.det(rotation) - 1) > 1e-5:
        raise ValueError('R_obj2cam must be SO(3); scorer does not repair predictions')
    return rotation


def load_surface(path, offset=None):
    mesh = trimesh.load(str(path), force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('invalid mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('invalid triangle indices')
    if offset is not None:
        vertices = vertices - finite_array(offset, (3,), 'mesh offset')
    return vertices, faces


def surface_samples(vertices, faces, count=4096, seed=314159):
    """Deterministic area-weighted samples; zero-area kit padding is ignored."""
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    keep = area > 1e-14
    triangles, area = triangles[keep], area[keep]
    if not len(area) or not np.isfinite(area).all() or not np.isfinite(area.sum()):
        raise ValueError('mesh has no finite nondegenerate surface')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triangles), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    tri = triangles[chosen]
    samples = tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
    centroid = np.sum(triangles.mean(1) * area[:, None], axis=0) / area.sum()
    return samples, centroid


def surface_cd_sum_cm(predicted, target):
    """SUM of two mean Euclidean NN distances, matching the kit definition."""
    return float(100 * (cKDTree(target).query(predicted, workers=1)[0].mean()
                        + cKDTree(predicted).query(target, workers=1)[0].mean()))


def acceleration(values, times):
    dt = np.diff(times)
    if (dt <= 0).any():
        raise ValueError('timestamps must strictly increase')
    velocity = np.diff(values, axis=0) / dt[:, None]
    return 2 * np.diff(velocity, axis=0) / (times[2:] - times[:-2])[:, None]


def motion_errors(predicted, target, times, frame_indices, gt_rows, cadence=1, gt_cadence=1):
    """Only original declared-cadence source and GT triples enter ACC."""
    if len(times) < 3:
        return {'eligible_triples': 0, 'timestamp_acc_error_cm_s2': None, 'nominal_acc_error_cm_retained_sample2': None}
    eligible = ((np.diff(frame_indices)[:-1] == cadence) & (np.diff(frame_indices)[1:] == cadence)
                & (np.diff(gt_rows)[:-1] == gt_cadence) & (np.diff(gt_rows)[1:] == gt_cadence))
    if not eligible.any():
        return {'eligible_triples': 0, 'timestamp_acc_error_cm_s2': None, 'nominal_acc_error_cm_retained_sample2': None}
    actual = np.linalg.norm(acceleration(predicted, times) - acceleration(target, times), axis=1)[eligible] * 100
    nominal = np.linalg.norm(np.diff(predicted, n=2, axis=0) - np.diff(target, n=2, axis=0), axis=1)[eligible] * 100
    return {'eligible_triples': int(eligible.sum()), 'timestamp_acc_error_cm_s2': float(actual.mean()),
            'nominal_acc_error_cm_retained_sample2': float(nominal.mean()), 'per_triple_timestamp_error_cm_s2': actual.tolist(),
            'per_triple_nominal_error_cm_retained_sample2': nominal.tolist()}


def self_test():
    vertices = np.array([[0., 0, 0], [.4, 0, 0], [0, .3, 0]])
    points, _ = surface_samples(vertices, np.array([[0, 1, 2]]), 1024)
    identical = surface_cd_sum_cm(points, points)
    shift = surface_cd_sum_cm(points + [0, 0, .1], points)
    assert identical < 1e-10 and abs(shift - 20) < 1e-8, (identical, shift)
    times = np.array([0., .031, .065, .101, .14])
    target = np.zeros((5, 3)); predicted = target.copy(); predicted[:, 0] = .5 * 2 * times ** 2
    assert np.max(np.abs(acceleration(predicted, times)[:, 0] - 2)) < 1e-10
    result = motion_errors(predicted, target, times, np.arange(5), np.arange(5))
    assert abs(result['timestamp_acc_error_cm_s2'] - 200) < 1e-8
    assert motion_errors(predicted, target, times, np.arange(5) * 5, np.arange(5))['eligible_triples'] == 0
    # Changing local origin with matching t compensation leaves posed samples unchanged.
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    offset = np.array([.1, -.2, .3]); translation = np.array([.4, .5, 2.])
    assert np.max(abs((points - offset) @ rotation.T + translation + rotation @ offset
                      - (points @ rotation.T + translation))) < 1e-12
    try:
        validate_rotation(np.diag([1., 1, -1]))
    except ValueError:
        pass
    else:
        raise AssertionError('reflection accepted')
    return {'self_cd_cm': identical, 'known_10cm_plane_shift_cd_sum_cm': shift,
            'nonuniform_time_acceleration_gate': True, 'sparse_frames_acc_excluded': True,
            'recenter_compensation_gate': True, 'reflection_rejected': True}


def unique_index(records, key):
    result = {}
    for record in records:
        if record[key] in result:
            raise ValueError(f'duplicate frozen {key}')
        result[record[key]] = record
    return result


def selected_native_records(path, image_ids, annotation_ids):
    """Decode only selected ID records; never json.load native development+reserve."""
    decoder = json.JSONDecoder()
    outputs = []
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        split = data.find(b'"annotations"')
        if split < 0:
            raise ValueError('native annotations section missing')
        for start, stop, wanted in [(0, split, set(image_ids)), (split, len(data), set(annotation_ids))]:
            found = {}
            for match in re.compile(rb'"id"\s*:\s*(\d+)\s*,').finditer(data, start, stop):
                identity = int(match.group(1))
                if identity not in wanted:
                    continue
                begin = data.rfind(b'{', start, match.start())
                record, _ = decoder.raw_decode(data[begin:begin + 200000].decode('utf-8'))
                if record.get('id') != identity or identity in found:
                    raise ValueError('native ID record structure/uniqueness mismatch')
                found[identity] = record
            if set(found) != wanted:
                raise ValueError('native requested IDs missing')
            outputs.append(found)
    return outputs


def intercap_references(request, references, fps):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('InterCap requires explicit positive --intercap-source-fps; timestamps are nominal')
    if references.get('schema') != 'intercap-scorer-only-reference-v1' or references.get('reserve_samples_included') is not False:
        raise ValueError('require development-only InterCap references without reserve records')
    sources, normalized = {}, []
    inputs = unique_index(request['InterCap'], 'take')
    refs = unique_index(references['development_samples'], 'take')
    if set(refs) != set(inputs):
        raise ValueError('InterCap dev reference/request cohort mismatch')
    if any(x['native_units_origin_projection_gate'] != 'verified-camera-metres-native-origin-v1' for x in refs.values()):
        raise ValueError('native units/origin/projection gate is not verified')
    paths = {x['GT_annotation_source'] for x in refs.values()}
    if len(paths) != 1:
        raise ValueError('one pinned native annotation source required')
    native = next(iter(paths))
    if sha256(native) != references['native_annotations_sha256']:
        raise ValueError('native annotations hash mismatch')
    sources[str(Path(native).resolve())] = references['native_annotations_sha256']
    images, annotations = selected_native_records(native,
        [i for r in refs.values() for i in r['image_ids']],
        [i for r in refs.values() for i in r['annotation_ids']])
    truth = {}
    for take, ref in refs.items():
        source = dict(inputs[take]); frame_ids = source['native_frame_numbers']
        if source.get('source_frame_cadence') != 5 or ref['camera'] != 1 or source['camera'] != 1:
            raise ValueError('only frozen camera1 cadence5 InterCap development references supported')
        if not (len(frame_ids) == len(ref['file_names']) == len(ref['annotation_ids']) == len(ref['image_ids'])):
            raise ValueError('native reference arrays differ in length')
        if sha256(ref['scorer_mesh']) != ref['scorer_mesh_sha256']:
            raise ValueError('native model hash mismatch')
        sources[str(Path(ref['scorer_mesh']).resolve())] = ref['scorer_mesh_sha256']
        poses = []
        for i, (frame, aid, iid, name) in enumerate(zip(frame_ids, ref['annotation_ids'], ref['image_ids'], ref['file_names'])):
            image, ann = images[iid], annotations[aid]
            if (ann['image_id'] != iid or image['file_name'] != name or not name.startswith(take + '/Frames_Cam1/color/')
                    or int(Path(name).stem) != frame or not source['rgb_paths'][i].endswith('/' + name)
                    or ann['obj_param']['name'] != take.split('/')[1]):
                raise ValueError('native dev image/annotation/object/frame join mismatch')
            if set(image['cam_param']) != {'focal', 'princpt'}:
                raise ValueError('new camera extrinsics require explicit contract review')
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(finite_array(ann['obj_param']['pose'], (3,), 'native axis angle')).as_matrix()
            pose[:3, 3] = finite_array(ann['obj_param']['trans'], (3,), 'native translation metres')
            poses.append(pose)
        source.update(sequence=take, video_frame_indices=frame_ids,
                      timestamps_seconds=(np.asarray(frame_ids, np.float64) / fps).tolist())
        normalized.append(source)
        truth[take] = {'poses': np.asarray(poses), 'mesh': ref['scorer_mesh'],
                       'rows': list(range(len(frame_ids))), 'cadence': 5}
    return normalized, truth, sources


def score(prediction_path, request_path, reference_path, *, allow_partial=False, samples=4096, dataset='BEHAVE', intercap_source_fps=None):
    prediction = json.loads(Path(prediction_path).read_text())
    request = json.loads(Path(request_path).read_text())
    references = json.loads(Path(reference_path).read_text())
    if prediction.get('schema') != 'opencv-metric-object-predictions-v1':
        raise ValueError('prediction schema must explicitly declare metric OpenCV poses')
    if not isinstance(prediction.get('sequences'), list) or not prediction['sequences']:
        raise ValueError('nonempty prediction sequences required')
    if not 16 <= samples <= 200000:
        raise ValueError('surface samples outside [16,200000]')
    intercap_truth, extra_sources = {}, {}
    if dataset == 'InterCap':
        records, intercap_truth, extra_sources = intercap_references(request, references, intercap_source_fps or float('nan'))
        inputs = unique_index(records, 'sequence')
        refs = intercap_truth
    elif dataset == 'BEHAVE':
        inputs = unique_index(request['BEHAVE'], 'sequence')
        refs = unique_index(references, 'sequence')
    else:
        raise ValueError('unsupported dataset')
    predictions = {}
    for item in prediction['sequences']:
        seq = item['sequence']
        if seq in predictions or seq not in inputs or seq not in refs:
            raise ValueError('duplicate or unfrozen prediction sequence')
        predictions[seq] = item
    if not allow_partial and set(predictions) != set(inputs):
        raise ValueError('full sequence coverage required; use --allow-partial only for diagnostics')
    sources = {str(Path(p).resolve()): sha256(p) for p in [prediction_path, request_path, reference_path, __file__]}
    sources.update(extra_sources)
    results = []
    for seq, item in predictions.items():
        source, ref = inputs[seq], refs[seq]
        if item.get('camera') != source['camera'] or source['camera'] != 1:
            raise ValueError('only explicitly matched camera1 references are currently supported')
        if item.get('mesh_units') != 'metres' or item.get('pose_convention') != POSE_CONVENTION:
            raise ValueError('require explicit metre mesh and exact object-to-camera pose convention')
        if not isinstance(item.get('source'), dict) or not item['source'] or not item.get('origin_convention'):
            raise ValueError('require nonempty source provenance and origin convention')
        correspondence = item.get('origin_correspondence_to_gt')
        if correspondence not in ('unverified', 'verified'):
            raise ValueError('origin_correspondence_to_gt must be unverified or verified')
        if correspondence == 'verified' and not item.get('origin_correspondence_source'):
            raise ValueError('verified origin correspondence requires evidence/source')
        scale = float(item.get('object_scale', float('nan')))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('one finite positive object_scale per sequence is required')
        path = Path(item['mesh_path'])
        if sha256(path) != item['mesh_sha256']:
            raise ValueError('predicted mesh hash mismatch')
        sources[str(path.resolve())] = item['mesh_sha256']
        pred_vertices, pred_faces = load_surface(path)
        if dataset == 'BEHAVE':
            for key, expected in [('gt_npz', 'gt_sha256'), ('calibration_json', 'calibration_sha256'),
                                  ('object_mesh_original', 'object_mesh_original_sha256'),
                                  ('object_mesh_simplified', 'object_mesh_simplified_sha256')]:
                if sha256(ref[key]) != ref[expected]:
                    raise ValueError('scorer reference hash mismatch')
                sources[str(Path(ref[key]).resolve())] = ref[expected]
            gt = np.load(ref['gt_npz'], allow_pickle=False)
            if int(gt['kid']) != 1:
                raise ValueError('GT camera mismatch')
            simplified_vertices, _ = load_surface(ref['object_mesh_simplified'])
            offset = simplified_vertices.mean(0)
            if not np.allclose(offset, ref['mesh_origin_subtracted_mean'], rtol=0, atol=1e-9):
                raise ValueError('GT mesh origin convention changed')
            gt_vertices, gt_faces = load_surface(ref['object_mesh_original'], offset)
        else:
            gt_vertices, gt_faces = load_surface(ref['mesh'])  # native origin, no centering
        pred_points, pred_center = surface_samples(pred_vertices, pred_faces, samples)
        gt_points, gt_center = surface_samples(gt_vertices, gt_faces, samples)
        expected_frames = source['video_frame_indices']
        if len(set(expected_frames)) != len(expected_frames):
            raise ValueError('duplicate request frames')
        lookup = {frame: i for i, frame in enumerate(expected_frames)}
        frames = {}
        for record in item['frames']:
            frame = record['video_frame_index']
            if type(frame) is not int or frame not in lookup or frame in frames:
                raise ValueError('prediction frame duplicate, invalid or outside frozen cohort')
            frames[frame] = record
        if not frames or (not allow_partial and set(frames) != set(expected_frames)):
            raise ValueError('missing predictions; partial diagnostics require explicit --allow-partial')
        ordered = sorted(frames)
        positions = [lookup[frame] for frame in ordered]
        if dataset == 'BEHAVE':
            gt_rows = np.array([ref['GT_row_indices'][i] for i in positions])
            if any(str(gt['frames'][row]) != ref['GT_frame_names'][i] for i, row in zip(positions, gt_rows)):
                raise ValueError('GT row/name join mismatch')
            gt_poses = gt['pose_abs']
            cadence = 1
        else:
            gt_rows = np.array([ref['rows'][i] for i in positions])
            gt_poses = ref['poses']
            cadence = ref['cadence']
        times = np.array([source['timestamps_seconds'][i] for i in positions], np.float64)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError('invalid timestamps')
        cds, pred_t, gt_t, pred_centers, gt_centers = [], [], [], [], []
        for frame, row in zip(ordered, gt_rows):
            r = validate_rotation(frames[frame]['R_obj2cam'])
            t = finite_array(frames[frame]['t_obj2cam_m'], (3,), 'translation')
            transform = finite_array(gt_poses[row], (4, 4), 'GT pose')
            rg = validate_rotation(transform[:3, :3]); tg = transform[:3, 3]
            cds.append(surface_cd_sum_cm(scale * pred_points @ r.T + t, gt_points @ rg.T + tg))
            pred_t.append(t); gt_t.append(tg)
            pred_centers.append(scale * pred_center @ r.T + t); gt_centers.append(gt_center @ rg.T + tg)
        local_acc = motion_errors(np.array(pred_t), np.array(gt_t), times, np.array(ordered), gt_rows, cadence=cadence)
        center_acc = motion_errors(np.array(pred_centers), np.array(gt_centers), times, np.array(ordered), gt_rows, cadence=cadence)
        results.append({'sequence': seq, 'camera': 1, 'expected_frames': len(expected_frames), 'scored_frames': len(ordered),
                        'coverage': len(ordered) / len(expected_frames), 'frame_ids': ordered,
                        'missing_frame_ids': sorted(set(expected_frames) - set(frames)),
                        'surface_cd_sum_cm': float(np.mean(cds)), 'surface_cd_per_frame_cm': cds,
                        'local_origin_translation_ACC': local_acc, 'surface_centroid_ACC_diagnostic': center_acc,
                        'origin_correspondence_to_gt': correspondence,
                        'local_origin_ACC_interpretable_as_corresponding_point_error': correspondence == 'verified',
                        'max_GT_video_timestamp_join_error_seconds': max(ref['timestamp_errors_seconds'][i] for i in positions) if dataset == 'BEHAVE' else None,
                        'source_frame_cadence': cadence, 'timestamp_basis': 'measured video elapsed seconds' if dataset == 'BEHAVE' else f'native frame number / declared {intercap_source_fps}fps; nominal',
                        'prediction_source': item['source'], 'origin_convention': item['origin_convention']})
    complete = set(predictions) == set(inputs) and all(x['coverage'] == 1 for x in results)
    return {'schema': 'labeled-object-development-score-v1', 'dataset': dataset + '_engineering_development',
            'alignment': 'none; raw metric OpenCV camera coordinates; no human or object GT similarity fit',
            'metric_definition': 'CD=sum of mean Euclidean nearest-sample distances in both directions, cm; deterministic area-weighted surfaces',
            'surface_samples_per_mesh': samples, 'surface_sampling_seed': 314159,
            'human_alignment_present': False, 'exact_hosted_metrics': False,
            'nominal_ACC_disclosure': 'Raw second-difference error in cm/retained-sample² on original declared-cadence triples; source cadence is recorded. BEHAVE uses measured joined video times; InterCap uses explicitly declared nominal source fps. Interior triples only; kit endpoint padding absent.',
            'centroid_ACC_disclosure': 'Surface-area centroid is invariant to local recentering but differing shapes need not have identical physical centroids.',
            'complete_coverage': complete, 'partial_diagnostic': allow_partial,
            'missing_sequences': sorted(set(inputs) - set(predictions)),
            'mean_sequence_surface_cd_sum_cm': float(np.mean([x['surface_cd_sum_cm'] for x in results])),
            'per_sequence': results, 'source_sha256': sources, 'self_test': self_test()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--surface-samples', type=int, default=4096)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--dataset', choices=['BEHAVE', 'InterCap'], default='BEHAVE')
    parser.add_argument('--intercap-source-fps', type=float)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if any(getattr(args, field) is None for field in ['prediction', 'requests', 'references', 'out']):
        parser.error('--prediction/--requests/--references/--out are required')
    if args.out.exists() or not 16 <= args.surface_samples <= 200000:
        parser.error('output must be new; surface-samples must be in [16,200000]')
    report = score(args.prediction, args.requests, args.references,
                   allow_partial=args.allow_partial, samples=args.surface_samples,
                   dataset=args.dataset, intercap_source_fps=args.intercap_source_fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({'complete_coverage': report['complete_coverage'],
                      'mean_sequence_surface_cd_sum_cm': report['mean_sequence_surface_cd_sum_cm'],
                      'scored_sequences': len(report['per_sequence'])}, indent=2))


if __name__ == '__main__':
    main()
