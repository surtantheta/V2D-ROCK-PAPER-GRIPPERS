"""PBS-only fixed-human joint trajectory experiment and predicted fidelity gates."""
from pathlib import Path
import argparse,json,os,shutil,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import temporal_v2 as util
from joint_core_v1 import Config,solve
R=util.ROOT;L=util.LARGE;P=util.PRIOR;OLD=util.OLD
OLD_INPUT=Path('/rhome/tanuj/v2d_identity_20261029/mesh_fit/old_pi3x_v3/INPUTS.json')

class ImageDepthGuard:
    def __init__(self,case,data,human,human_faces,v,f,base):
        from PIL import Image
        self.base=base;self.v=v;self.f=f;self.r=data['object_rotation'];self.t0=data['object_translation'];self.s=float(data['object_scale'])
        k=np.load(case['paths']['poses'])['K_model'].astype(float);self.k=k[0] if k.ndim==3 else k
        assert case['calibration']['MAIN_camera']['S6_model_wh']==[512,384]
        assert np.allclose(self.k,case['calibration']['MAIN_camera']['S6_K'],atol=1e-6)
        self.kr=np.diag([.5,.5,1.])@self.k
        tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        self.center=(tri.mean(axis=1)*area[:,None]).sum(axis=0)/area.sum()
        self.c0=self.s*np.einsum('tij,j->ti',self.r,self.center)+self.t0
        self.uv0=self.project(self.c0,self.k)
        geometry=np.load(case['paths']['geometry']);self.obs=[]
        for sample in case['samples']:
            if not sample['selected'] or sample['state'] in ['absent','uncertain','ambiguous']:continue
            i=sample['frame']['local_index'];position=sample['position'];fr=sample['frame']
            assert int(geometry['frame_local_indices'][position])==i and int(geometry['native_frame_ids'][position])==fr['native_frame_id']
            mask=np.asarray(Image.open(sample['mask']).convert('L').resize((256,192),Image.Resampling.NEAREST))>127
            if not mask.any():continue
            hz,_,_=base.raster(human[i],human_faces,self.kr,192,256)
            z,_,_=base.raster(self.s*(self.v@self.r[i].T)+self.t0[i],self.f,self.kr,192,256)
            visible=np.isfinite(z)&((z<=hz+.002)|mask)
            baseline_iou=self.iou(visible,mask)
            cal=case['calibration'];body=cal['frames'][position]
            reliable=bool(cal['calibration_metric_diagnostic_eligible'] and body['calibrated_body_residual_cm'] is not None and body['calibrated_body_residual_cm']['mean']<=15 and body['body']['body_pixels']>=64)
            points=np.zeros((0,3))
            if reliable:
                points,_=base.sample_cloud(geometry['local_camera_points'][position],geometry['rgb_to_model'][position],sample['mask'],fr['rgb_wh'],cal['calibration'])
            xy=np.zeros((0,2),int);cloud_z=np.zeros(0);baseline_depth=None
            if len(points):
                pix=np.rint(self.project(points,self.kr)).astype(int)
                good=(pix[:,0]>=0)&(pix[:,0]<256)&(pix[:,1]>=0)&(pix[:,1]<192)&(points[:,2]>.001)
                pix=pix[good];points=points[good]
                good=visible[pix[:,1],pix[:,0]]
                if good.sum()>=16:
                    xy=pix[good];cloud_z=points[good,2]
                    baseline_depth=float(np.minimum(np.abs(z[xy[:,1],xy[:,0]]-cloud_z),.25).mean())
            self.obs.append(dict(frame=i,mask=mask,human_z=hz,baseline_iou=baseline_iou,xy=xy,cloud_z=cloud_z,baseline_depth=baseline_depth,metric_calibration_reliable=reliable))

    @staticmethod
    def project(points,k):
        h=points@k.T;assert np.isfinite(h).all() and (h[:,2]>.001).all()
        return h[:,:2]/h[:,2,None]

    @staticmethod
    def iou(a,b):
        return float((a&b).sum()/max(1,(a|b).sum()))

    def __call__(self,delta):
        c=self.c0+delta
        shift=np.linalg.norm(self.project(c,self.k)-self.uv0,axis=1)
        minz=min(float((self.s*(self.v@r.T)+tr+off)[:,2].min()) for r,tr,off in zip(self.r,self.t0,delta))
        reasons=[]
        if shift.max()>16:reasons.append('surface_centroid_projection_change_over_16px_at512')
        if minz<=.001:reasons.append('object_vertex_not_in_front')
        rows=[]
        for ob in self.obs:
            i=ob['frame'];z,_,_=self.base.raster(self.s*(self.v@self.r[i].T)+self.t0[i]+delta[i],self.f,self.kr,192,256)
            visible=np.isfinite(z)&((z<=ob['human_z']+.002)|ob['mask'])
            iou=self.iou(visible,ob['mask']);depth=None
            xy=ob['xy']
            if len(xy):
                valid=visible[xy[:,1],xy[:,0]]
                losses=np.full(len(xy),.25)
                losses[valid]=np.minimum(np.abs(z[xy[valid,1],xy[valid,0]]-ob['cloud_z'][valid]),.25)
                depth=float(losses.mean())
            rows.append(dict(frame=i,baseline_iou=ob['baseline_iou'],candidate_iou=iou,baseline_masked_depth_residual_m=ob['baseline_depth'],candidate_masked_depth_residual_m=depth,frozen_baseline_cloud_points=len(xy),metric_calibration_reliable=ob['metric_calibration_reliable']))
        if rows:
            b=float(np.mean([a['baseline_iou'] for a in rows]));a=float(np.mean([a['candidate_iou'] for a in rows]))
            if b>=.1 and a<b-.02:reasons.append('mean_visible_mask_iou_drop_over_.02')
            depths=[a for a in rows if a['baseline_masked_depth_residual_m'] is not None]
            if depths and np.mean([a['candidate_masked_depth_residual_m']-a['baseline_masked_depth_residual_m'] for a in depths])>.01:reasons.append('mean_masked_depth_residual_increase_over_10mm')
        return not reasons,dict(reasons=reasons,maximum_surface_centroid_projection_change_px=float(shift.max()),maximum_translation_camera_z_change_m=float(np.abs(delta[:,2]).max()),minimum_fullmesh_camera_z_m=minz,observations=rows,unobserved_frames_have_projection_bound_only=True)

def fit_case(case,data,model,hand,human_faces,pen,base,budget_mesh,output):
    ep=case['episode'];name=f'episode_{ep:06d}';start=time.monotonic()
    n=len(data['pose']);human=base.decode_human(model,data,np.arange(n),__import__('torch'))
    hands=human[:,hand]
    v,f=budget_mesh(util.LOCAL/f'{name}_object.ply',4096,4096)
    guard=ImageDepthGuard(case,data,human,human_faces,v,f,base)
    meta=json.loads((OLD/'work_labeled/LABELED_EPISODES.json').read_text())[str(ep)]
    updated,diag,initial,final=solve(v,f,data['object_rotation'],float(data['object_scale']),hands,data['object_translation'],meta['frame_ids'],pen,Config(),image_guard=guard)
    offset=updated-data['object_translation'];ok,visibility=guard(offset);assert ok
    result=dict(data);result['object_translation']=updated
    for key in data:
        if key!='object_translation':assert result[key].dtype==data[key].dtype and result[key].tobytes()==data[key].tobytes()
    dest=output/'submission';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{name}.npz',**result)
    shutil.copyfile(util.LOCAL/f'{name}_object.ply',dest/f'{name}_object.ply')
    assert util.sha(dest/f'{name}_object.ply')==util.sha(util.LOCAL/f'{name}_object.ply')
    before=np.maximum(-initial,0);after=np.maximum(-final,0)
    changed=dict(newly_penetrating_frames=int((~(before>1e-6).any(axis=1)&(after>1e-6).any(axis=1)).sum()),cleared_frames=int(((before>1e-6).any(axis=1)&~(after>1e-6).any(axis=1)).sum()),maximum_depth_worsened_frames=int((after.max(axis=1)>before.max(axis=1)+1e-8).sum()),strict_new_inside_points=int(((before<=0)&(after>0)).sum()))
    np.savez_compressed(output/f'{name}_PHYSICAL_STATE.npz',native_frame_ids=np.array(meta['frame_ids']),hand_vertex_ids=hand,hands_scene_m=hands,initial_signed_distance_scene_m=initial,final_signed_distance_scene_m=final,baseline_translation=data['object_translation'],candidate_translation=updated)
    util.dump(output/f'{name}_JOINT.json',dict(episode=ep,frames=n,diagnostics=diag,final_image_depth_guard=visibility,penetration_changes=changed,human_mesh_rotation_scale_bitexact=True,GT_used=False,seconds=time.monotonic()-start))
    print('GENUINE_JOINT_CASE',ep,n,diag['maximum_correction_m'],diag['initial_merit'],diag['final_merit'],diag['initial_penetration'],diag['final_penetration'],flush=True)
    return dict(episode=ep,frames=n,maximum_correction_m=diag['maximum_correction_m'],accepted_outer_steps=len(diag['history']),initial_merit=diag['initial_merit'],final_merit=diag['final_merit'],penetration_changes=changed,diagnostic=str(output/f'{name}_JOINT.json'))

def generate(stage,pilot_audit=None):
    assert os.environ.get('PBS_JOBID')
    policy=json.loads((R/'JOINT_INPUTS_V3.json').read_text());util.verify(policy['sha256'])
    tests=json.loads((R/'JOINT_CORE_TEST_V1.json').read_text());assert tests['status']=='PASS';util.verify(tests['source_sha256'])
    if stage=='full':
        assert pilot_audit is not None
        audit=json.loads(Path(pilot_audit).read_text())
        assert audit['status']=='PASS' and audit['stage']=='GENUINE_JOINT_PILOT_V3'
        assert audit['producer_completion_sha256']==util.sha(L/'joint_v3/pilot/COMPLETE.json')
        assert audit['protocol_sha256']==util.sha(R/'JOINT_INPUTS_V3.json')
    import torch,numba
    torch.set_num_threads(2);torch.set_num_interop_threads(1);numba.set_num_threads(2)
    pe=util.module(P/'code/evaluate_penetration.py','joint_pe');pen=util.module(pe.PEN,'joint_official_pen')
    base=util.module(P/'code/run_real_mesh_fit_v3.py','joint_decode')
    sys.path.insert(0,str(pe.KIT));from v2dlb.mesh_budget import budget_mesh
    hand=np.load(pe.HAND);assert len(hand)==5840
    human_faces=np.load(OLD/'models/mhr/mhr_faces_lod1.npy').astype(np.int64)
    model=torch.jit.load(str(OLD/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
    old=json.loads(OLD_INPUT.read_text());cases={c['episode']:c for c in old['cases']}
    output=L/'joint_v3'/stage;assert not output.exists();output.mkdir(parents=True)
    rows=[]
    for ep in ([900,904] if stage=='pilot' else list(range(900,912))):
        name=f'episode_{ep:06d}'
        if stage=='full' and ep in [900,904]:
            pilot=L/'joint_v3/pilot';receipt=json.loads((pilot/'COMPLETE.json').read_text());util.verify(receipt['output_sha256'])
            assert receipt['pilot_technical_gate_passed']
            (output/'submission').mkdir(exist_ok=True)
            for suffix in ['.npz','_object.ply']:shutil.copyfile(pilot/'submission'/f'{name}{suffix}',output/'submission'/f'{name}{suffix}')
            for suffix in ['_PHYSICAL_STATE.npz','_JOINT.json']:shutil.copyfile(pilot/f'{name}{suffix}',output/f'{name}{suffix}')
            rows.append(dict(next(r for r in receipt['records'] if r['episode']==ep),reused_identical_pilot=True));continue
        data=dict(np.load(util.LOCAL/f'{name}.npz',allow_pickle=False))
        rows.append(fit_case(cases[ep],data,model,hand,human_faces,pen,base,budget_mesh,output))
    util.verify(policy['sha256'])
    if stage=='pilot':
        action=any(r['maximum_correction_m']>1e-4 and r['final_merit']<r['initial_merit']-1e-9 for r in rows)
        passed=len(rows)==2 and sum(r['frames'] for r in rows)==124 and action
    else:passed=len(rows)==12 and sum(r['frames'] for r in rows)==630
    assert passed,('genuine joint stage failed technical/productive gate',stage,rows)
    pins={str(p):util.sha(p) for p in (output/'submission').iterdir() if p.is_file()}
    aux={str(p):util.sha(p) for p in output.iterdir() if p.is_file()}
    util.dump(output/'COMPLETE.json',dict(status='GENUINE_JOINT_TRANSLATION_'+stage.upper()+'_COMPLETE',records=rows,cases=len(rows),frames=sum(r['frames'] for r in rows),pilot_technical_gate_passed=passed,output_sha256=pins,auxiliary_output_sha256=aux,GT_used_for_inference=False,MAIN_human_bitexact=True,mesh_bitexact=True,rotation_scale_bitexact=True,posthoc_smoothing=False,protocol_sha256=util.sha(R/'JOINT_INPUTS_V3.json'),CPU_job=os.environ['PBS_JOBID']))

def evaluate():
    candidate=L/'joint_v3/full'
    ev=util.module(P/'code/evaluate_candidate.py','genuine_joint_cd');ev.H=R
    ev.run(argparse.Namespace(candidate=candidate/'submission',completion=candidate/'COMPLETE.json',out=R/'genuine_joint_scores_v3',baseline_parity=False,label='Genuine joint translation λ64, frozen contact support C10'))
    pe=util.module(P/'code/evaluate_penetration.py','genuine_joint_pen');pe.H=R
    pe.run(argparse.Namespace(candidate=candidate/'submission',completion=candidate/'COMPLETE.json',out=R/'genuine_joint_pen_v3'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','full','evaluate']);ap.add_argument('--pilot-audit',type=Path);a=ap.parse_args()
    assert os.environ.get('PBS_JOBID')
    if a.stage=='evaluate':evaluate()
    else:generate(a.stage,a.pilot_audit)
