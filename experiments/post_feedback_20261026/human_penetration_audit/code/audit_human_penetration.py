#!/usr/bin/env python3
"""Fixed-pointset CPU penetration and predicted-human diagnostics. No inference.

Human acceleration is a prediction/change diagnostic, NOT GT ACC-H error.
Penetration uses the same 5840 finger-driven vertices across all arms, not the
unavailable hosted 512-point role asset. No native annotations are opened.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path('/home/tanuj/V2D_challenge')
HERE = PROJECT/'experiments/post_feedback_20261026'
KIT = PROJECT/'data/assets/submission_kit/v2d_submission_kit'
ARMS = ['main', 'moge_full', 'hybrid']
ROOTS = {
    'main': PROJECT/'work_labeled/submission',
    'moge_full': PROJECT/'work_labeled_moge/submission',
    'hybrid': HERE/'predicted_gauge_hybrid/behave_v1/submission',
}
SIDE = {
    'main': PROJECT/'work_labeled/align_BEHAVE/TRANSFORMS.json',
    'moge_full': HERE/'moge_audit/scoring/align_BEHAVE/TRANSFORMS.json',
    'hybrid': PROJECT/'work_labeled/align_BEHAVE/TRANSFORMS.json',
}
PROTOCOL = dict(
    schema='fixed-handset-penetration-predicted-human-diagnostics-v1',
    episodes=[900, 901, 902, 903], arms=ARMS,
    penetration_local_frames='32 fixed rounded linspace indices from 0 to95, same all arms',
    human_local_frames='all96 eachcase; original source IDs/timestamps, no gaps bridged',
    pointset='finger-driven MHR lod1 vertices: pose parameters68:122 each0.3, zero shape/pose; displacement>1e-4 metres',
    expected_hand_vertices=5840,
    body_joints='official MHR_TABLE3_BODY_JOINT_INDICES, 22 joints',
    decoder='TorchScript MHR, pose136+scales68, shape45, zero expression72, correctives=True, cm/100*[1,-1,-1]',
    geometry='exact kit4096vertices4096faces, no recentering; original s,R,t',
    penetration='kit PEN._penetration_depth((hand-t)@R/object_scale) * object_scale * own fixed humanSim3_scale *100cm',
    human_change='camera-space and own-human-aligned differences relative MAIN; not GT error',
    shape='coefficient changes and canonical rest mesh/joint differences; not GT shape error',
    exact_hosted_PEN=False, exact_hosted_ACC_H=False,
    native_annotation_records_opened=0, reserve_included=False,
    parameters_or_frame_selection_tuned_from_scores=False,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def stats(values):
    v = np.asarray(values, float).reshape(-1)
    assert np.isfinite(v).all()
    return dict(count=len(v), mean=float(v.mean()) if len(v) else None,
        median=float(np.median(v)) if len(v) else None,
        p95=float(np.percentile(v, 95)) if len(v) else None,
        maximum=float(v.max()) if len(v) else None)


def acceleration(values, times):
    a = np.asarray(values, np.float64); t = np.asarray(times, np.float64)
    assert (np.diff(t) > 0).all()
    dt = np.diff(t).reshape((-1,)+(1,)*(a.ndim-1))
    total = (t[2:]-t[:-2]).reshape((-1,)+(1,)*(a.ndim-1))
    return 2*np.diff(np.diff(a, axis=0)/dt, axis=0)/total


def own_motion(joints, times, eligible):
    nominal = np.linalg.norm(np.diff(joints, n=2, axis=0), axis=-1)[eligible]*100
    physical = np.linalg.norm(acceleration(joints, times), axis=-1)[eligible]*100
    return dict(nominal_cm_per_source_frame_squared=stats(nominal),
        timestamp_cm_per_second_squared=stats(physical), eligible_triples=int(eligible.sum()),
        diagnostic_only_not_GT_error=True)


def map_points(points, transform):
    return transform['scale']*(points@np.asarray(transform['R']).T)+np.asarray(transform['t'])


def to_local(hands, scale, rotations, translations):
    return np.einsum('tpi,tij->tpj', hands-translations[:, None], rotations)/scale


def self_test():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(261026);v=rng.normal(size=(7,11,3));
    R=np.repeat(Rotation.from_rotvec([.2,-.3,.1]).as_matrix()[None],7,axis=0)
    t=rng.normal(size=(7,3));s=.73;local=to_local(v,s,R,t)
    H=Rotation.from_rotvec([-.4,.2,.3]).as_matrix();a=1.4;b=np.array([.2,-.5,.8])
    again=to_local(a*v@H.T+b,a*s,H@R,a*t@H.T+b)
    equiv=float(abs(again-local).max());assert equiv<1e-12
    offset=np.array([.7,-.3,.5]);shifted=to_local(v,s,R,t+s*np.einsum('tij,j->ti',R,offset))
    origin=float(abs(shifted-(local-offset)).max());assert origin<1e-12
    times=np.array([0.,.03,.07,.12,.18,.25,.33]);x=np.zeros((7,22,3));x[:,:,0]=times[:,None]**2
    err=float(abs(acceleration(x,times)[:,:,0]-2).max());assert err<1e-12
    return dict(scale_alignment_once_inverse_max_error_m=equiv,
        original_mesh_origin_invariance_max_error_m=origin,
        nonuniform_time_acceleration_max_error=err)


def decode(model, data, torch, ids=None):
    ids=np.arange(len(data['pose'])) if ids is None else np.asarray(ids)
    params=np.concatenate([data['pose'][ids],np.repeat(data['scales'][None],len(ids),axis=0)],axis=1).astype(np.float32)
    return decode_parameters(model,params,data['shape'],torch)


def decode_parameters(model, params, shape, torch):
    vs=[];js=[]
    with torch.inference_mode():
        for start in range(0,len(params),8):
            p=np.asarray(params[start:start+8],np.float32)
            sh=np.repeat(np.asarray(shape,np.float32).reshape(1,45),len(p),axis=0)
            v,j=model(torch.from_numpy(sh),torch.from_numpy(p),torch.zeros(len(p),72),True)
            vs.append(v.double().numpy()/100*np.array([1.,-1.,-1.]))
            js.append(j[...,:3].double().numpy()/100*np.array([1.,-1.,-1.]))
    v,j=np.concatenate(vs),np.concatenate(js)
    assert v.shape==(len(params),18439,3) and j.shape==(len(params),127,3)
    assert np.isfinite(v).all() and np.isfinite(j).all()
    return v,j


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--self-test-only',action='store_true');a=ap.parse_args();tests=self_test()
    if a.self_test_only: print(json.dumps(tests));return
    assert not a.out.exists();a.out.mkdir(parents=True);start=time.monotonic()
    write(a.out/'PROTOCOL.json',dict(PROTOCOL,synthetics=tests))
    source_paths={Path(__file__).resolve(),a.out/'PROTOCOL.json',
        PROJECT/'models/mhr/mhr_model.pt', KIT/'metric_code/track_1/PEN.py',
        KIT/'v2dlb/mhr_metrics.py', KIT/'v2dlb/mesh_budget.py',
        PROJECT/'work_labeled/LABELED_EPISODES.json',
        HERE/'predicted_gauge_hybrid/behave_v1/COMPLETE.json',
        HERE/'moge_audit/FINAL_VALIDATION.json'}
    source_paths.update(SIDE.values())
    for root in ROOTS.values():
        for ep in PROTOCOL['episodes']:
            source_paths.update([root/f'episode_{ep:06d}.npz',root/f'episode_{ep:06d}_object.ply'])
    guards={str(p):sha(p) for p in sorted(source_paths)}
    write(a.out/'SOURCE_GUARDS.json',guards)
    sys.path.insert(0,str(KIT))
    from v2dlb.mesh_budget import budget_mesh
    from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES
    joints=np.asarray(MHR_TABLE3_BODY_JOINT_INDICES);assert joints.shape==(22,)
    import torch,numba
    torch.set_num_threads(2);torch.set_num_interop_threads(1);numba.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('fixed_kit_penetration',KIT/'metric_code/track_1/PEN.py')
    pen=importlib.util.module_from_spec(spec);spec.loader.exec_module(pen)
    assert pen._PENETRATION_KERNELS is not None,'Require numba kernels; avoid unbounded fallback worker count'
    model=torch.jit.load(str(PROJECT/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
    v0,_=decode_parameters(model,np.zeros((1,204),np.float32),np.zeros(45),torch)
    params=np.zeros((54,204),np.float32);params[np.arange(54),np.arange(68,122)]=.3
    changed,_=decode_parameters(model,params,np.zeros(45),torch)
    hand=np.flatnonzero(np.linalg.norm(changed-v0,axis=-1).max(0)>1e-4)
    assert len(hand)==5840,('fixed pointset mismatch',len(hand))
    np.save(a.out/'HAND_VERTEX_IDS.npy',hand)
    metadata=read(PROJECT/'work_labeled/LABELED_EPISODES.json')
    maps={arm:{r['sequence']:r['primary_transform'] for r in read(path)['records']} for arm,path in SIDE.items()}
    selected=np.round(np.linspace(0,95,32)).astype(int);assert len(np.unique(selected))==32
    records=[]
    for ep in PROTOCOL['episodes']:
        name=f'episode_{ep:06d}';meta=metadata[str(ep)];seq=meta['sequence']
        assert meta['dataset']=='BEHAVE' and meta['frames']==96
        native=np.asarray(meta['frame_ids']);times=np.asarray(meta['times'],np.float64)
        assert len(native)==len(times)==96 and len(np.unique(native))==96
        eligible=(np.diff(native)[:-1]==1)&(np.diff(native)[1:]==1)
        data={arm:dict(np.load(root/(name+'.npz'),allow_pickle=False)) for arm,root in ROOTS.items()}
        assert all(np.array_equal(data['main'][k],data['hybrid'][k]) for k in ['pose','scales','shape'])
        decoded={};rest={}
        for arm in ['main','moge_full']:
            decoded[arm]=decode(model,data[arm],torch)
            p=np.zeros((1,204),np.float32);p[0,136:]=data[arm]['scales']
            rest[arm]=decode_parameters(model,p,data[arm]['shape'],torch)
        decoded['hybrid']=decoded['main'];rest['hybrid']=rest['main']
        mainV,mainJ=decoded['main'];mainJ=mainJ[:,joints]
        mainAlignedV=map_points(mainV,maps['main'][seq]);mainAlignedJ=map_points(mainJ,maps['main'][seq])
        case=dict(episode=ep,sequence=seq,frames=96,penetration_frames=32,hand_vertices=len(hand),arms={})
        for arm in ARMS:
            d=data[arm];V,J=decoded[arm];J=J[:,joints];transform=maps[arm][seq]
            assert transform['scale']>0
            alignedV=map_points(V,transform);alignedJ=map_points(J,transform)
            human=dict(MAIN_parameter_arrays_bitexact=all(np.array_equal(d[k],data['main'][k]) for k in ['pose','scales','shape']),
                raw_vertex_change_vs_MAIN_cm=stats(np.linalg.norm(V-mainV,axis=-1)*100),
                own_aligned_vertex_change_vs_MAIN_cm=stats(np.linalg.norm(alignedV-mainAlignedV,axis=-1)*100),
                shape_coefficient_delta_L2=float(np.linalg.norm(d['shape']-data['main']['shape'])),
                scale_coefficient_delta_L2=float(np.linalg.norm(d['scales']-data['main']['scales'])),
                canonical_rest_vertex_change_vs_MAIN_cm=stats(np.linalg.norm(rest[arm][0]-rest['main'][0],axis=-1)*100),
                own_camera_22joint_motion=own_motion(J,times,eligible),
                own_aligned_22joint_motion=own_motion(alignedJ,times,eligible),
                relative_MAIN_22joint_motion=own_motion(alignedJ-mainAlignedJ,times,eligible),
                diagnostic_not_GT_human_accuracy=True)
            if arm in ['main','hybrid']:
                assert human['MAIN_parameter_arrays_bitexact'] and human['own_aligned_vertex_change_vs_MAIN_cm']['maximum']==0
            vertices,faces=budget_mesh(ROOTS[arm]/(name+'_object.ply'),4096,4096)
            scale=float(d['object_scale']);assert scale>0
            R=d['object_rotation'][selected];t=d['object_translation'][selected]
            local=to_local(V[selected][:,hand],scale,R,t)
            depth=pen._penetration_depth(local.reshape(-1,3),vertices,faces).reshape(32,len(hand))
            assert np.isfinite(depth).all() and (depth>=0).all()
            camera_cm=depth*scale*100;aligned_cm=camera_cm*transform['scale']
            perframe=[]
            for j,local_index in enumerate(selected):
                perframe.append(dict(local_index=int(local_index),native_frame=int(native[local_index]),
                    mean_depth_cm=float(aligned_cm[j].mean()),p95_point_depth_cm=float(np.percentile(aligned_cm[j],95)),
                    max_point_depth_cm=float(aligned_cm[j].max()),inside_fraction=float((depth[j]>0).mean())))
            penetration=dict(proxy_not_hosted_PEN=True,sampled_local_indices=selected.tolist(),
                object_scale=scale,human_alignment_scale=float(transform['scale']),
                camera_frame_mean_depth_cm=float(camera_cm.mean()),
                aligned_mean_depth_cm=float(aligned_cm.mean()),
                aligned_frame_mean_summary_cm=stats(aligned_cm.mean(1)),
                aligned_point_depth_summary_cm=stats(aligned_cm),
                mean_inside_fraction=float((depth>0).mean()),
                sampled_frame_fraction_any_inside=float((depth>0).any(1).mean()),per_frame=perframe)
            case['arms'][arm]=dict(human=human,penetration=penetration)
            print('PEN_DONE',ep,arm,'mean_cm',penetration['aligned_mean_depth_cm'],flush=True)
        out=a.out/f'CASE_{ep}.json';write(out,case);records.append(case)
    summary=[]
    for arm in ARMS:
        penrows=[c['arms'][arm]['penetration'] for c in records]
        summary.append(dict(arm=arm,cases=4,human_frames=384,penetration_frames=128,
            mean_sequence_penetration_proxy_cm=float(np.mean([x['aligned_mean_depth_cm'] for x in penrows])),
            worst_sequence_penetration_proxy_cm=float(max(x['aligned_mean_depth_cm'] for x in penrows)),
            maximum_point_depth_cm=float(max(x['aligned_point_depth_summary_cm']['maximum'] for x in penrows)),
            human_MAIN_bitexact=all(c['arms'][arm]['human']['MAIN_parameter_arrays_bitexact'] for c in records),
            mean_sequence_own_aligned_22joint_nominal_motion=float(np.mean([
                c['arms'][arm]['human']['own_aligned_22joint_motion']['nominal_cm_per_source_frame_squared']['mean'] for c in records]))))
    write(a.out/'SUMMARY.json',dict(protocol=PROTOCOL,summary=summary,
        full_12_case_adoption_gate_not_satisfied_by_BEHAVE_only=True,
        exact_hosted_metrics=False,GT_error_metrics_computed=False))
    for p,h in guards.items():assert sha(p)==h,('source changed during audit',p)
    outputs={str(p):sha(p) for p in a.out.iterdir() if p.is_file()}
    write(a.out/'COMPLETE.json',dict(status='PASS',cases=4,human_frames_per_arm=384,
        penetration_frames_per_arm=128,hand_point_count=5840,arms=ARMS,source_sha256=guards,
        output_sha256=outputs,elapsed_seconds=time.monotonic()-start,exact_hosted_metrics=False,
        native_annotation_records_opened=0,reserve_included=False,GPU_used=False))
    print('COMPLETE',a.out,flush=True)


if __name__=='__main__':main()
