#!/usr/bin/env python3
"""Scorer-only coarse-point coordinate audit; not a decoded SAM3D mesh evaluation."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
CACHE=Path('/rhome/tanuj/unicon3rpp_runs/sam3d_contract/gate3')
SEQUENCES=['Date01_Sub01_stool_move','Date01_Sub01_monitor_hand','Date01_Sub01_chairblack_hand','Date01_Sub01_boxsmall_hand']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cd(a,b):return float(cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())*100

def main():
    t0=time.time();out=ROOT/'runs/cached_stage1_coordinate_audit_v2';out.mkdir()
    helper=Path('/home/tanuj/projects/UniCon3Rpp_M5_src/tools/sam3d_behave_gate3.py');spec=importlib.util.spec_from_file_location('gate_helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    manifest=CACHE/'gate3_manifest.json';rows=json.loads(manifest.read_text())['samples'];sources={str(p):sha(p) for p in [Path(__file__),helper,manifest]};records=[]
    for s in rows:
        if s['sequence'] not in SEQUENCES:continue
        template=h.load_template(s,count=4000);target=template@np.array(s['gt_rotation_cam']).T+np.array(s['gt_translation_cam'])
        for k in ['template_original','template_simplified']:sources[s[k]]=sha(s[k])
        for seed in [11,29,47,83]:
            p=CACHE/'predictions'/f'{s["sample_id"]}__seed{seed}.npz';sources[str(p)]=sha(p);pred=np.load(p)
            cloud=np.array(pred['voxel'],float);rng=np.random.default_rng(20261020)
            if len(cloud)>4000:cloud=cloud[rng.choice(len(cloud),4000,replace=False)]
            scale=np.array(pred['scale'],float);q=h.rotation_from_quaternion_wxyz(pred['rotation']);flip=np.diag([-1.,-1.,1.]);tt=flip@np.array(pred['translation'],float)
            # Official Transform3d uses row points @ Rq, hence OpenCV column rotation F @ Rq.T.
            correct=(cloud*scale)@(flip@q.T).T+tt
            legacy=(cloud*scale)@(flip@q).T+tt
            record={'sequence':s['sequence'],'sample_id':s['sample_id'],'camera':s['kid'],'seed':seed,'points_sampled':len(cloud),'correct_rowform_cd_point_proxy_cm_sum':cd(correct,target),'legacy_columnform_cd_point_proxy_cm_sum':cd(legacy,target),
                    'predicted_point_centroid_distance_to_GT_surface_centroid_cm':float(np.linalg.norm(correct.mean(0)-target.mean(0))*100),
                    'predicted_to_GT_median_depth_ratio':float(np.median(correct[:,2])/np.median(target[:,2])),
                    'predicted_to_GT_centered_mean_radius_ratio':float(np.linalg.norm(correct-correct.mean(0),axis=1).mean()/np.linalg.norm(target-target.mean(0),axis=1).mean()),
                    'predicted_to_GT_centered_RMS_radius_ratio':float(np.sqrt(np.mean(np.sum((correct-correct.mean(0))**2,axis=1)))/np.sqrt(np.mean(np.sum((target-target.mean(0))**2,axis=1)))),
                    'predicted_scale_vector':scale.tolist(),'gt_transform_used_for_initialization':False}
            records.append(record)
    summary={}
    for seq in SEQUENCES:
        rr=[x for x in records if x['sequence']==seq]
        keys=['correct_rowform_cd_point_proxy_cm_sum','legacy_columnform_cd_point_proxy_cm_sum','predicted_point_centroid_distance_to_GT_surface_centroid_cm','predicted_to_GT_median_depth_ratio','predicted_to_GT_centered_mean_radius_ratio','predicted_to_GT_centered_RMS_radius_ratio']
        summary[seq]={k:{'median':float(np.median([x[k] for x in rr])),'min':float(min(x[k] for x in rr)),'max':float(max(x[k] for x in rr))} for k in keys}
    report={'schema':'stage1-coarse-point-coordinate-audit-v1','records':records,'summary':summary,'sources_sha256':sources,'scope':'Existing stage1voxelpointoccupancy vs GTsurface samples, oracle-mask source; NOT decodedmesh surfaceCD, NOT primaryRGBpredictedmask pipeline. No GTregistration/alignment. Correct convention is fixed by source, not selected by score. Varying physical cameras and distant timestamps, noACC.', 'seconds':time.time()-t0}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(summary,indent=2));print('seconds',report['seconds'])
if __name__=='__main__':main()
