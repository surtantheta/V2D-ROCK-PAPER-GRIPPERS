#!/usr/bin/env python3
"""Read-only output audit; object GT is accessed only for score recomputation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')


def samples(path, offset=None):
    mesh = trimesh.load(str(path), force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, np.float64)
    if offset is not None:
        vertices = vertices-offset
    tri = vertices[np.asarray(mesh.faces)]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    good = area > 1e-14
    tri = tri[good]; area = area[good]
    rng = np.random.default_rng(314159)
    chosen = rng.choice(len(tri), 4096, p=area/area.sum())
    uv = rng.random((4096, 2)); flip = uv.sum(1) > 1
    uv[flip] = 1-uv[flip]
    selected = tri[chosen]
    return selected[:, 0]+uv[:, :1]*(selected[:, 1]-selected[:, 0])+uv[:, 1:]*(selected[:, 2]-selected[:, 0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    output = ROOT/'human_aligned_surrogate_scores_v1'
    transforms = json.loads((ROOT/'human_alignment_sidecars_v1/TRANSFORMS.json').read_text())
    lookup = {r['sequence']: r['primary_transform'] for r in transforms['records']}
    gate_path = ROOT/'LAUNCH_human_alignment_surrogate.json'
    gate = json.loads(gate_path.read_text())
    for path, expected in gate['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    sources = [ROOT/'labeled_predictions_v2/human_anchored_comparison',
        ROOT/'labeled_predictions_v2/temporal_continuity_comparison',
        ROOT/'labeled_hunyuan_placement_v1/temporal_comparison']
    references = {r['sequence']: r for r in json.loads((ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json').read_text())}
    checks = []; cds = []
    for directory in sources:
        for original in sorted(directory.glob('*_PREDICTIONS.json')):
            label = directory.parent.name+'_'+directory.name+'_'+original.name
            prediction_path = output/label; score_path = output/label.replace('_PREDICTIONS', '_SCORES')
            if not prediction_path.exists() or not score_path.exists():
                assert not args.require_complete, label
                continue
            before = json.loads(original.read_text()); after = json.loads(prediction_path.read_text())
            score = json.loads(score_path.read_text())
            assert score['complete_coverage'] and not score['exact_hosted_metrics']
            assert {s['sequence'] for s in before['sequences']} == set(lookup)
            before_lookup = {s['sequence']: s for s in before['sequences']}
            score_lookup = {s['sequence']: s for s in score['per_sequence']}
            maximum = 0.; frame_count = 0
            for seq in after['sequences']:
                old = before_lookup[seq['sequence']]; x = lookup[seq['sequence']]
                scale = x['scale']; rotation = np.array(x['R']); shift = np.array(x['t'])
                assert seq['mesh_path'] == old['mesh_path'] and seq['mesh_sha256'] == old['mesh_sha256']
                assert abs(seq['object_scale']-scale*old['object_scale']) < 1e-12
                assert len(seq['frames']) == len(old['frames']) == 96
                assert [f['video_frame_index'] for f in seq['frames']] == [f['video_frame_index'] for f in old['frames']]
                for f, g in zip(seq['frames'], old['frames']):
                    err = max(float(abs(np.array(f['R_obj2cam'])-rotation@np.array(g['R_obj2cam'])).max()),
                        float(abs(np.array(f['t_obj2cam_m'])-(scale*rotation@np.array(g['t_obj2cam_m'])+shift)).max()))
                    maximum = max(maximum, err); assert err < 1e-12
                    frame_count += 1
                # Recompute one full first-frame surface CD per sequence and family.
                if original.name == 'foundationpose_PREDICTIONS.json':
                    ref = references[seq['sequence']]
                    center = np.asarray(trimesh.load(ref['object_mesh_simplified'], force='mesh', process=False).vertices).mean(0)
                    gt = np.load(ref['gt_npz'], allow_pickle=False)['pose_abs'][ref['GT_row_indices'][0]]
                    f = seq['frames'][0]
                    p = seq['object_scale']*samples(seq['mesh_path'])@np.array(f['R_obj2cam']).T+f['t_obj2cam_m']
                    g = samples(ref['object_mesh_original'], center)@gt[:3, :3].T+gt[:3, 3]
                    cd = float(100*(cKDTree(g).query(p)[0].mean()+cKDTree(p).query(g)[0].mean()))
                    actual = score_lookup[seq['sequence']]['surface_cd_per_frame_cm'][0]
                    assert abs(cd-actual) < 1e-9
                    cds.append(dict(label=label, sequence=seq['sequence'], cd_cm=cd, error_cm=abs(cd-actual)))
            assert frame_count == 384
            mean = float(np.mean([s['surface_cd_sum_cm'] for s in score['per_sequence']]))
            assert abs(mean-score['mean_sequence_surface_cd_sum_cm']) < 1e-10
            checks.append(dict(label=label, frames=frame_count, pose_error_max=maximum,
                full_cohort_mean_sequence_cd_sum_cm=mean,
                per_sequence_cd_sum_cm={s['sequence']: s['surface_cd_sum_cm'] for s in score['per_sequence']}))
    if args.require_complete:
        assert (output/'COMPLETE.json').exists() and len(checks) == 18
        manifest = json.loads((output/'COMPLETE.json').read_text())
        assert len(manifest['records']) == 18
        assert manifest['scorer_only'] and not manifest['exact_hosted_metrics']
        for record in manifest['records']:
            for key in ['original', 'prediction', 'scores']:
                assert hashlib.sha256(Path(record[key]).read_bytes()).hexdigest() == record[key+'_sha256']
    report = dict(frozen_gate_source_count=len(gate['source_sha256']),
        prediction_score_pairs_checked=len(checks), expected_pairs=18,
        checks=checks, independent_first_frame_CD_checks=cds,
        all_expected_complete=len(checks) == 18 and (output/'COMPLETE.json').exists(),
        object_GT_access='scoring validation only; no transform estimation')
    path = ROOT.parents[1]/'validation/OPT_HUMAN_ALIGNED_OUTPUT_CHECKS_20261020.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
