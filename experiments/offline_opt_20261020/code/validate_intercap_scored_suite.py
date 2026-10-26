#!/usr/bin/env python3
"""Independent actual score/composition audit; selected native development IDs only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def surface(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    triangles = np.asarray(mesh.vertices, float)[np.asarray(mesh.faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    keep = area > 1e-14; triangles = triangles[keep]; area = area[keep]
    rng = np.random.default_rng(314159); idx = rng.choice(len(triangles), 4096, p=area/area.sum())
    uv = rng.random((4096, 2)); flip = uv.sum(1) > 1; uv[flip] = 1-uv[flip]
    tri = triangles[idx]
    points = tri[:, 0]+uv[:, :1]*(tri[:, 1]-tri[:, 0])+uv[:, 1:]*(tri[:, 2]-tri[:, 0])
    center = (triangles.mean(1)*area[:, None]).sum(0)/area.sum()
    return points, center


def main():
    from labeled_object_scorer import selected_native_records
    base = ROOT/'intercap_results_v1'; complete = json.loads((base/'COMPLETE.json').read_text())
    gate = json.loads((ROOT/'LAUNCH_intercap_results_v1.json').read_text())
    for path, expected in gate['source_sha256'].items():
        assert sha(path) == expected, path
    assert len(complete['records']) == 6 and complete['all_arms_retained']
    assert complete['frames_per_arm'] == 246 and complete['takes_per_arm'] == 8
    assert not complete['exact_hosted_metrics'] and not complete['report_reserve_included']
    transforms_path = ROOT/'intercap_human_alignment_sidecars_v1/TRANSFORMS.json'
    assert sha(transforms_path) == complete['transform_receipt_sha256']
    transforms = {r['sequence']: r['primary_transform'] for r in json.loads(transforms_path.read_text())['records']}
    references = json.loads((ROOT/'benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json').read_text())
    refs = references['development_samples']
    images, anns = selected_native_records(refs[0]['GT_annotation_source'],
        [i for r in refs for i in r['image_ids']], [i for r in refs for i in r['annotation_ids']])
    assert len(images) == len(anns) == 246
    requests = {r['take']: r for r in json.loads((ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json').read_text())['InterCap']}
    truth = {}; surfaces = {}
    for ref in refs:
        path = ref['scorer_mesh']
        if path not in surfaces:
            surfaces[path] = surface(path)
        params = [anns[i]['obj_param'] for i in ref['annotation_ids']]
        truth[ref['take']] = dict(R=Rotation.from_rotvec([p['pose'] for p in params]).as_matrix(),
            t=np.asarray([p['trans'] for p in params]), surface=surfaces[path])
    rows = []; cd_errors = []; acc_errors = []; composition_errors = []
    for record in complete['records']:
        for key in ['prediction', 'raw_camera_scores', 'human_aligned_prediction', 'human_aligned_scores']:
            assert sha(record[key]) == record[key+'_sha256']
        original = {r['sequence']: r for r in json.loads(Path(record['prediction']).read_text())['sequences']}
        for mode in ['raw_camera', 'human_aligned']:
            score = json.loads(Path(record[mode+'_scores']).read_text())
            pred_path = record['prediction'] if mode == 'raw_camera' else record['human_aligned_prediction']
            prediction = {r['sequence']: r for r in json.loads(Path(pred_path).read_text())['sequences']}
            assert set(prediction) == set(original) == set(requests)
            assert score['complete_coverage'] and not score['exact_hosted_metrics'] and not score['partial_diagnostic']
            assert score['human_alignment_present'] == (mode == 'human_aligned')
            slookup = {r['sequence']: r for r in score['per_sequence']}
            assert len(slookup) == 8
            for take, seq in prediction.items():
                saved = slookup[take]; native = truth[take]; frame_ids = requests[take]['native_frame_numbers']
                assert [f['video_frame_index'] for f in seq['frames']] == saved['frame_ids'] == frame_ids
                assert saved['coverage'] == 1 and not saved['missing_frame_ids']
                assert saved['source_frame_cadence'] == 5 and not saved['local_origin_ACC_interpretable_as_corresponding_point_error']
                ppath = seq['mesh_path']
                if ppath not in surfaces:
                    surfaces[ppath] = surface(ppath)
                p, pc = surfaces[ppath]; g, gc = native['surface']
                R = np.array([f['R_obj2cam'] for f in seq['frames']]); t = np.array([f['t_obj2cam_m'] for f in seq['frames']]); scale = seq['object_scale']
                if mode == 'human_aligned':
                    old = original[take]; h = transforms[take]; H = np.array(h['R']); shift = np.array(h['t'])
                    assert seq['mesh_path'] == old['mesh_path'] and seq['mesh_sha256'] == old['mesh_sha256']
                    err = max(abs(scale-h['scale']*old['object_scale']),
                        np.max(abs(R-H@np.array([f['R_obj2cam'] for f in old['frames']]))),
                        np.max(abs(t-(h['scale']*np.array([f['t_obj2cam_m'] for f in old['frames']])@H.T+shift))))
                    composition_errors.append(float(err)); assert err < 1e-12
                for i in [0, len(frame_ids)-1]:
                    pp = scale*p@R[i].T+t[i]; gg = g@native['R'][i].T+native['t'][i]
                    cd = 100*(cKDTree(gg).query(pp)[0].mean()+cKDTree(pp).query(gg)[0].mean())
                    error = abs(cd-saved['surface_cd_per_frame_cm'][i]); cd_errors.append(float(error)); assert error < 1e-9
                assert abs(np.mean(saved['surface_cd_per_frame_cm'])-saved['surface_cd_sum_cm']) < 1e-10
                times = np.asarray(frame_ids, float)/30.; dt = np.diff(times)
                for label, pv, gv in [('local_origin_translation_ACC', t, native['t']),
                    ('surface_centroid_ACC_diagnostic', scale*np.einsum('nij,j->ni', R, pc)+t,
                     np.einsum('nij,j->ni', native['R'], gc)+native['t'])]:
                    pa = 2*np.diff(np.diff(pv, axis=0)/dt[:, None], axis=0)/(times[2:]-times[:-2])[:, None]
                    ga = 2*np.diff(np.diff(gv, axis=0)/dt[:, None], axis=0)/(times[2:]-times[:-2])[:, None]
                    actual = 100*np.linalg.norm(pa-ga, axis=1)
                    nominal = 100*np.linalg.norm(np.diff(pv, n=2, axis=0)-np.diff(gv, n=2, axis=0), axis=1)
                    a = saved[label]; assert a['eligible_triples'] == len(times)-2
                    error = max(float(abs(actual-a['per_triple_timestamp_error_cm_s2']).max()),
                        float(abs(nominal-a['per_triple_nominal_error_cm_retained_sample2']).max()))
                    acc_errors.append(error); assert error < 1e-7
            mean = float(np.mean([s['surface_cd_sum_cm'] for s in score['per_sequence']]))
            assert abs(mean-score['mean_sequence_surface_cd_sum_cm']) < 1e-10
            rows.append(dict(arm=record['arm'], alignment=mode, frames=246, takes=8,
                mean_sequence_surface_cd_sum_cm=mean,
                per_sequence_surface_cd_sum_cm={s['sequence']: s['surface_cd_sum_cm'] for s in score['per_sequence']}))
    report = dict(score_passes_checked=len(rows), gate_hashes_passed=len(gate['source_sha256']),
        manifest_artifact_hashes_passed=24, native_selected_image_records=246,
        native_selected_annotation_records=246, reserve_payloads_decoded=0,
        aligned_pose_composition_max_error=max(composition_errors),
        independent_CD_frame_checks=len(cd_errors), independent_CD_max_error_cm=max(cd_errors),
        independent_ACC_sequence_diagnostic_checks=len(acc_errors), independent_ACC_max_error=max(acc_errors),
        full_cohort_rows=rows, exact_hosted_metrics=False,
        limitation='engineering development; nominal30fps/stride5; mapped GT-human alignment surrogate; origin-correspondence unverified')
    dest = ROOT.parents[1]/'validation/OPT_INTERCAP_ACTUAL_SCORE_SUITE_20261020.json'
    dest.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
