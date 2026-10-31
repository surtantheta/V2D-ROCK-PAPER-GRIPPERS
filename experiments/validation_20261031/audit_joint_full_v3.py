#!/usr/bin/env python3
"""Independent actual pilot hands/SDF/joint-merit/trajectory contract audit."""
import datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
import numpy as np
B=Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments');STUDY=B/'physical_refinement_20261031';OUT=B/'validation_20261031'
RUN=Path('/rhome/tanuj/v2d_physical_20261031/joint_v3/full');LOCAL=Path('/rhome/tanuj/v2d_identity_20261029/mesh_fit/old_pi3x_v3/submission');OLD=Path('/home/tanuj/V2D_challenge');KIT=OLD/'data/assets/submission_kit/v2d_submission_kit'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(4194304),b''):h.update(z)
 return h.hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def decode(model,data,torch):
 result=[]
 for start in range(0,len(data['pose']),8):
  p=data['pose'][start:start+8];n=len(p);params=np.concatenate([p,np.repeat(data['scales'][None],n,0)],axis=1).astype(np.float32);shape=np.repeat(data['shape'].astype(np.float32).reshape(1,45),n,0)
  with torch.inference_mode():v,_=model(torch.from_numpy(shape),torch.from_numpy(params),torch.zeros(n,72),True)
  result.append(v.double().numpy()/100*np.array([1.,-1.,-1.]))
 return np.concatenate(result)
def signed(hands,t,R,s,v,f,pen,radius):
 tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);tri=tri[area>0]
 lo=tri.min(axis=(0,1))-radius/s;hi=tri.max(axis=(0,1))+radius/s
 faces=[np.ascontiguousarray(tri[:,k]) for k in range(3)];winding,distance=pen._PENETRATION_KERNELS
 out=np.ones(hands.shape[:2],float)
 for i in range(len(t)):
  q=(hands[i]-t[i])@R[i]/s;take=((q>=lo)&(q<=hi)).all(axis=1);p=np.ascontiguousarray(q[take])
  if len(p):
   inside=np.abs(winding(p,*faces))>.5;dist=distance(p,*faces)*s;out[i,take]=np.where(inside,-dist,dist)
 return out
def merit(delta,t0,ids,sdf,norm,cfg):
 steps,counts=np.unique(np.diff(ids),return_counts=True);stride=int(steps[np.argmax(counts)]);vals=[]
 for i in range(1,len(ids)-1):
  if ids[i]-ids[i-1]==stride and ids[i+1]-ids[i]==stride:vals.append(((t0+delta)[i-1]-2*(t0+delta)[i]+(t0+delta)[i+1])/(stride*stride))
 fidelity=float(np.square(delta).sum());temporal=cfg['temporal_native_strength']*float(np.square(vals).sum());collision=cfg['collision_strength']*float((np.square(np.maximum(cfg['clearance_margin_m']-sdf,0)).sum(axis=1)/norm).sum())
 return fidelity+temporal+collision

def main():
 assert os.environ.get('PBS_JOBID')
 import torch,numba
 torch.set_num_threads(2);torch.set_num_interop_threads(1);numba.set_num_threads(2)
 protocol=STUDY/'JOINT_INPUTS_V3.json';m=json.loads(protocol.read_text());cfg=m['config'];guards={str(protocol):sha(protocol),str(__file__):sha(__file__)}
 for p,h in m['sha256'].items():assert sha(p)==h,('input changed',p);guards[p]=h
 complete=RUN/'COMPLETE.json';c=json.loads(complete.read_text());assert c['cases']==12 and c['frames']==630 and c['pilot_technical_gate_passed'] is True and c['GT_used_for_inference'] is False
 assert c['protocol_sha256']==sha(protocol) and [r['episode'] for r in c['records']]==list(range(900,912))
 for group in [c['output_sha256'],c['auxiliary_output_sha256']]:
  for p,h in group.items():assert sha(p)==h,('output changed',p);guards[p]=h
 pen=load(KIT/'metric_code/track_1/PEN.py','independent_joint_pen');assert pen._PENETRATION_KERNELS is not None
 sys.path.insert(0,str(KIT));from v2dlb.mesh_budget import budget_mesh
 hand=np.load(OLD/'experiments/post_feedback_20261026/human_penetration_audit/run_v2_cari4d/HAND_VERTEX_IDS.npy');assert len(hand)==5840
 model=torch.jit.load(str(OLD/'models/mhr/mhr_model.pt'),map_location='cpu').eval();meta=json.loads((OLD/'work_labeled/LABELED_EPISODES.json').read_text());oldcases={r['episode']:r for r in json.loads((LOCAL.parent/'INPUTS.json').read_text())['cases']}
 rows=[];active=False;total=0
 for record in c['records']:
  ep=record['episode'];stem=f'episode_{ep:06d}';src=dict(np.load(LOCAL/f'{stem}.npz',allow_pickle=False));a=dict(np.load(RUN/'submission'/f'{stem}.npz',allow_pickle=False));p=dict(np.load(RUN/f'{stem}_PHYSICAL_STATE.npz',allow_pickle=False));r=json.loads((RUN/f'{stem}_JOINT.json').read_text());diag=r['diagnostics'];n=len(src['pose'])
  assert n==record['frames']==r['frames'] and r['GT_used'] is False and set(a)==set(src)
  for k in src:
   if k!='object_translation':assert src[k].dtype==a[k].dtype and src[k].shape==a[k].shape and src[k].tobytes()==a[k].tobytes(),(ep,k)
  assert sha(LOCAL/f'{stem}_object.ply')==sha(RUN/'submission'/f'{stem}_object.ply')
  ids=np.asarray(meta[str(ep)]['frame_ids']);assert np.array_equal(p['native_frame_ids'],ids) and np.array_equal(p['hand_vertex_ids'],hand)
  human=decode(model,src,torch);hands=human[:,hand];herror=float(np.max(np.abs(hands-p['hands_scene_m'])));assert herror<1e-12
  assert np.array_equal(p['baseline_translation'],src['object_translation']) and np.array_equal(p['candidate_translation'],a['object_translation'])
  delta=a['object_translation']-src['object_translation'];maximum=float(np.linalg.norm(delta,axis=1).max());assert maximum<=cfg['maximum_correction_m']+1e-12 and abs(maximum-diag['maximum_correction_m'])<1e-12
  v,f=budget_mesh(LOCAL/f'{stem}_object.ply',4096,4096);R=a['object_rotation'];s=float(a['object_scale']);assert s>0 and R.shape==(n,3,3) and np.isfinite(delta).all()
  initial=signed(hands,src['object_translation'],R,s,v,f,pen,cfg['clearance_margin_m']+cfg['maximum_correction_m']);final=signed(hands,a['object_translation'],R,s,v,f,pen,cfg['clearance_margin_m']+cfg['maximum_correction_m'])
  serr=max(float(np.max(np.abs(initial-p['initial_signed_distance_scene_m']))),float(np.max(np.abs(final-p['final_signed_distance_scene_m']))));assert serr<1e-9
  counts=(initial<cfg['clearance_margin_m']).sum(axis=1);normalizers=np.maximum(counts,len(hand)/cfg['reference_hand_sampling_density']);assert np.array_equal(counts,diag['baseline_active_counts']) and np.array_equal(normalizers,diag['frozen_perframe_normalizers'])
  initial_cost=merit(np.zeros_like(delta),src['object_translation'],ids,initial,normalizers,cfg);final_cost=merit(delta,src['object_translation'],ids,final,normalizers,cfg)
  assert abs(initial_cost-diag['initial_merit'])<1e-8 and abs(final_cost-diag['final_merit'])<1e-8 and final_cost<=initial_cost+1e-10
  before=np.maximum(-initial,0);after=np.maximum(-final,0);assert (after*after).mean()<=(before*before).mean()+cfg['mean_penetration_squared_tolerance_m2']+1e-14 and after.max()<=before.max()+cfg['maximum_point_depth_increase_m']+1e-10
  change=dict(newly_penetrating_frames=int((~(before>1e-6).any(axis=1)&(after>1e-6).any(axis=1)).sum()),cleared_frames=int(((before>1e-6).any(axis=1)&~(after>1e-6).any(axis=1)).sum()),maximum_depth_worsened_frames=int((after.max(axis=1)>before.max(axis=1)+1e-8).sum()),strict_new_inside_points=int(((before<=0)&(after>0)).sum()));assert change==r['penetration_changes']==record['penetration_changes']
  tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);mu=(tri.mean(axis=1)*area[:,None]).sum(axis=0)/area.sum();C=s*np.einsum('tij,j->ti',R,mu)+src['object_translation'];K=np.load(oldcases[ep]['paths']['poses'])['K_model'];K=K[0] if K.ndim==3 else K
  def project(x):
   y=x@K.T;return y[:,:2]/y[:,2,None]
  pixel=float(np.linalg.norm(project(C+delta)-project(C),axis=1).max());visibility=r['final_image_depth_guard'];assert pixel<=16+1e-10 and abs(pixel-visibility['maximum_surface_centroid_projection_change_px'])<1e-9 and not visibility['reasons']
  minz=min(float((s*(v@rr.T)+tt)[:,2].min()) for rr,tt in zip(R,a['object_translation']));assert minz>.001 and abs(minz-visibility['minimum_fullmesh_camera_z_m'])<1e-10
  observations=visibility['observations']
  if observations:
   b=np.mean([q['baseline_iou'] for q in observations]);z=np.mean([q['candidate_iou'] for q in observations]);assert b<.1 or z>=b-.02-1e-12
   d=[q for q in observations if q['baseline_masked_depth_residual_m'] is not None]
   if d:assert np.mean([q['candidate_masked_depth_residual_m']-q['baseline_masked_depth_residual_m'] for q in d])<=.01+1e-12
  active=active or(maximum>1e-4 and final_cost<initial_cost-1e-9);total+=n;rows.append(dict(episode=ep,frames=n,decoded_hands_max_error_m=herror,independent_signed_distance_max_error_m=serr,maximum_correction_m=maximum,initial_merit=initial_cost,final_merit=final_cost,maximum_projected_centroid_change_px=pixel,minimum_fullmesh_camera_z_m=minz,penetration_changes=change));print('INDEPENDENT_JOINT_FULL',ep,maximum,initial_cost,final_cost,serr,flush=True)
 assert total==630 and active
 report=dict(status='PASS',stage='GENUINE_JOINT_FULL_V3',producer_completion_sha256=sha(complete),protocol_sha256=sha(protocol),PBS_JOBID=os.environ['PBS_JOBID'],timestamp_sgt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),cases=12,frames=630,records=rows,productive_nontrivial_motion=True,MAIN_human_mesh_R_scale_bytes_preserved=True,GT_loaded=False,source_and_output_sha256=guards,limits=['Technical and objective-consistency gate only; no GT/hosted accuracy.','5840proxyhandpoints differ from hosted512roles.','Near-surface signed distances exact; faroutside expandedAABB is an irrelevant positive sentinel.','Saved final IoU/depth residual guards and scalar aggregations checked; this audit does not rerender all stored image observations.'])
 target=OUT/'GENUINE_JOINT_FULL_V3_OUTPUT_REVIEW.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_and_output_sha256']}),flush=True)
if __name__=='__main__':main()
