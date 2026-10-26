#!/usr/bin/env python3
"""Frozen optimizer on matched legacy FP; same v2 optimization-mask decisions.

No GT/scorer imports, no new tracking/inference and no parameter changes.
Only the trajectory and its own FP-IoU prior originate in matched_control.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import trimesh

import run_identity_scale_v2 as frozen

ROOT = frozen.ROOT
ARM_MAP = dict(legacy_temporal='identity_temporal',
               legacy_fixedscale_geometry='fixedscale_geometry',
               legacy_shared_scale='shared_scale',
               legacy_shared_scale_contact='shared_scale_contact')
ARMS = list(ARM_MAP)
FROZEN_SOURCE_SHA256 = {
    ROOT/'code/run_identity_scale_v2.py': 'aeeecd12046fd4ddd91792b964e68fc5e6d8a9aa4c77cb7b53b885908e226bc7',
    ROOT/'code/centroid_scale_trajectory_v2.py': 'ba15496a8d4057cb6ab1cbf12422591a3c127ae18ac6d2c798fce16553422a20',
}


def observation_sequence(legacy, identity):
    """Copy legacy prediction; replace only the optimization mask manifest pair."""
    assert legacy['sequence'] == identity['sequence']
    assert legacy['mesh_path'] == identity['mesh_path']
    assert legacy['mesh_sha256'] == identity['mesh_sha256']
    assert legacy['object_scale'] == identity['object_scale']
    assert legacy['source']['human_anchor']['alpha'] == identity['source']['human_anchor']['alpha']
    assert [f['video_frame_index'] for f in legacy['frames']] == [f['video_frame_index'] for f in identity['frames']]
    row = copy.deepcopy(legacy)
    for key in ['identity_mask_manifest', 'identity_mask_manifest_sha256']:
        row['source'][key] = identity['source'][key]
    assert frozen.sha(row['source']['identity_mask_manifest']) == row['source']['identity_mask_manifest_sha256']
    assert row['frames'] == legacy['frames']
    assert row['source']['poses_path'] == legacy['source']['poses_path']
    assert row['source']['poses_sha256'] == legacy['source']['poses_sha256']
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'identity_scale_v2/legacy_optimized')
    args = ap.parse_args()
    for path, expected in FROZEN_SOURCE_SHA256.items():
        assert frozen.sha(path) == expected, str(path)
    args.out.mkdir(exist_ok=False)
    request_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    requests = json.loads(request_path.read_text())
    sources = {str(p): frozen.sha(p) for p in [Path(__file__), *FROZEN_SOURCE_SHA256,
        Path(frozen.previous.__file__), ROOT/'code/run_constrained_intercap_v1.py', request_path]}
    reports = []
    for dataset in ['BEHAVE', 'InterCap']:
        cfg = frozen.configuration(dataset)
        legacy_path = ROOT/f'identity_scale_v2/matched_control/{dataset}_PREDICTIONS.json'
        identity_path = ROOT/f'identity_scale_v2/replay/{dataset}_PREDICTIONS.json'
        for path in [legacy_path, identity_path, cfg['inputs'], cfg['contacts']]:
            sources[str(path)] = frozen.sha(path)
        legacy = json.loads(legacy_path.read_text()); identity = json.loads(identity_path.read_text())
        item_rows = [x for x in json.loads(cfg['inputs'].read_text())['items'] if x['dataset'] == dataset]
        items = {x['take'] if dataset == 'InterCap' else x['name']: x for x in item_rows}
        contact_rows = json.loads(cfg['contacts'].read_text())['records']
        contacts = {x['sequence']: x for x in contact_rows}
        request_rows = requests[dataset]
        req = {x['take'] if dataset == 'InterCap' else x['sequence']: x for x in request_rows}
        identity_map = {x['sequence']: x for x in identity['sequences']}
        names = [x['sequence'] for x in legacy['sequences']]
        assert len(names) == len(set(names)) == cfg['count']
        assert len(item_rows) == len(items) == len(contact_rows) == len(contacts) == len(request_rows) == len(req) == cfg['count']
        assert len(identity_map) == len(identity['sequences']) == cfg['count']
        assert set(names) == set(identity_map) == set(items) == set(contacts) == set(req)
        outputs = {arm: copy.deepcopy(legacy) for arm in ARMS}
        folder = args.out/dataset; folder.mkdir(); summaries=[]
        for original in legacy['sequences']:
            name = original['sequence']; item=items[name]
            sequence=observation_sequence(original,identity_map[name])
            ids=req[name]['native_frame_numbers' if dataset == 'InterCap' else 'video_frame_indices']
            assert [f['video_frame_index'] for f in sequence['frames']] == ids
            times=np.asarray(ids)/30. if dataset == 'InterCap' else np.asarray(req[name]['timestamps_seconds'])
            obs,audit=frozen.observe(sequence,item,cfg['depth']/item['name'],contacts[name],times,cfg['step'],sources)
            mesh=trimesh.load(sequence['mesh_path'],process=False)
            vertices_metric=np.asarray(mesh.vertices)*sequence['object_scale']
            # Compare frozen selection/coverage/contact audit to the completed main v2 arm.
            main_audit_path=ROOT/f'identity_scale_v2/optimized/{dataset}/{item["name"]}_OBSERVATIONS.json'
            sources[str(main_audit_path)]=frozen.sha(main_audit_path)
            assert audit == json.loads(main_audit_path.read_text()), str(main_audit_path)
            (folder/f'{item["name"]}_OBSERVATIONS.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
            summary=dict(sequence=name,frames=len(ids),same_main_optimization_observation_audit=True,arms={})
            for arm,mapped_arm in ARM_MAP.items():
                t,ratio,diagnostics=frozen.fit_arm(mapped_arm,obs,times,ids,cfg['step'])
                t,ratio,diagnostics=frozen.export_validated(t,ratio,diagnostics,obs,vertices_metric)
                row=next(x for x in outputs[arm]['sequences'] if x['sequence']==name)
                row['object_scale']=float(sequence['object_scale']*ratio)
                for i,frame in enumerate(row['frames']):
                    frame['t_obj2cam_m']=t[i].tolist()
                poses=np.tile(np.eye(4),(len(t),1,1));poses[:,:3,:3]=obs['rotations'];poses[:,:3,3]=t
                pose_path=folder/f'{item["name"]}_{arm}_POSES.npz'
                np.savez_compressed(pose_path,T_obj2cam=poses,relative_object_scale=ratio)
                row['source']['legacy_scale_v2']=dict(arm=arm,frozen_arm=mapped_arm,
                    relative_object_scale=float(ratio),protocol=frozen.previous.PROTOCOL if mapped_arm=='identity_temporal' else frozen.core.PROTOCOL,
                    area_centroid_metric=obs['mu_metric'].tolist(),export_validation=diagnostics['export_validation'],
                    original_legacy_prediction_sha256=frozen.sha(legacy_path),
                    original_legacy_tracking_poses_path=original['source']['poses_path'],
                    original_legacy_tracking_poses_sha256=original['source']['poses_sha256'],
                    original_legacy_tracking_mask_manifest=original['source']['identity_mask_manifest'],
                    original_legacy_tracking_mask_manifest_sha256=original['source']['identity_mask_manifest_sha256'],
                    optimization_mask_manifest=sequence['source']['identity_mask_manifest'],
                    optimization_mask_manifest_sha256=sequence['source']['identity_mask_manifest_sha256'],
                    main_identity_prediction_sha256=frozen.sha(identity_path),
                    main_observation_audit_sha256=frozen.sha(main_audit_path),
                    optimization_observation_audit_identical_to_main=True,
                    fp_prior_quality_from='matched legacy raw track IoU; same frozen prior rule',
                    human_and_observed_depth_fixed_metric=True,rotation_fixed=True,
                    fixed_core_coefficients=True,GT_inputs=False)
                row['source'].update(identity_mask_manifest=sequence['source']['identity_mask_manifest'],
                    identity_mask_manifest_sha256=sequence['source']['identity_mask_manifest_sha256'],
                    poses_path=str(pose_path),poses_sha256=frozen.sha(pose_path),
                    poses_units='camera metres; human alpha already applied',optimizer=arm)
                summary['arms'][arm]=diagnostics
            summaries.append(summary);print(dataset,name,'finished',flush=True)
        records=[]
        for arm,prediction in outputs.items():
            assert sum(len(s['frames']) for s in prediction['sequences'])==cfg['frames']
            path=folder/f'{arm}_PREDICTIONS.json';path.write_text(json.dumps(prediction,indent=2,allow_nan=False))
            records.append(dict(arm=arm,prediction=str(path),prediction_sha256=frozen.sha(path)))
        reports.append(dict(dataset=dataset,records=records,sequences=summaries,
            frames_per_arm=cfg['frames'],takes_per_arm=cfg['count']))
    for path,expected in sources.items():assert frozen.sha(path)==expected,path
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(records=reports,arms=ARMS,
        source_sha256=sources,GT_inputs=False,reserve_included=False,full_frames_per_arm=630,
        development_after_failure_analysis=True,extension='frozen fit on matched legacy trajectories; same v2 observation masks',
        original_main_arms_and_outputs_unchanged=True),indent=2,allow_nan=False))


if __name__=='__main__':main()
