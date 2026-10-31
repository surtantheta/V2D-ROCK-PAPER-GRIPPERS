#!/usr/bin/env python3
"""Independent point-level official-distance/Trimesh-nearest diagnosis."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/tanuj/V2D_challenge')
KIT=OLD/'data/assets/submission_kit/v2d_submission_kit'
BASE=Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 assert os.environ.get('PBS_JOBID')
 import torch,numba,trimesh
 torch.set_num_threads(2);torch.set_num_interop_threads(1);numba.set_num_threads(2)
 sys.path.insert(0,str(KIT));from v2dlb.mesh_budget import budget_mesh
 pen=load(KIT/'metric_code/track_1/PEN.py','independent_probe_pen');winding,distance=pen._PENETRATION_KERNELS
 core=load(ROOT.parent/'physical_refinement_20261031/code/joint_core_v1.py','unchanged_joint_core_for_closest_probe')
 replay=load(ROOT.parent/'validation_20261031/audit_joint_full_v3.py','independent_probe_decoder')
 hand=np.load(OLD/'experiments/post_feedback_20261026/human_penetration_audit/run_v2_cari4d/HAND_VERTEX_IDS.npy')
 model=torch.jit.load(str(OLD/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
 rows=[]
 for ep,frame,point in [(12,163,3564),(13,175,3661)]:
  data=dict(np.load(BASE/f'episode_{ep:06d}.npz',allow_pickle=False));one=dict(data);one['pose']=data['pose'][frame:frame+1]
  human=replay.decode(model,one,torch);h=human[0,hand[point]]
  v,f=budget_mesh(BASE/f'episode_{ep:06d}_object.ply',4096,4096);r=data['object_rotation'][frame];s=float(data['object_scale']);t=data['object_translation'][frame]
  sm=core.SignedMesh(v,f,r[None],s,h[None,None],pen)
  for delta in [np.zeros(3),*[(sign*.02)*np.eye(3)[axis] for axis in range(3) for sign in [-1,1]]]:
   q=(h-(t+delta))@r/s;q_left=((h-t)-delta)@r/s
   oracle=float(distance(np.ascontiguousarray(q[None]),sm.v0,sm.v1,sm.v2)[0])*s
   p=sm.closest(q);closest=float(np.linalg.norm(q-p))*s
   full=trimesh.triangles.closest_point(sm.tri,np.repeat(q[None],len(sm.tri),axis=0));d=np.linalg.norm(q-full,axis=1);best=int(d.argmin())
   row=dict(episode=ep,frame=frame,hand_proxy_index=point,translation_delta_m=delta.tolist(),local_point=q.tolist(),local_association_point_grouping_difference=float(np.max(abs(q-q_left))),official_distance_m=oracle,nearest_distance_m=closest,global_nearest_distance_m=float(d[best]*s),distance_mismatch_m=closest-oracle,nearest_triangle_area=float(np.linalg.norm(np.cross(sm.tri[best,1]-sm.tri[best,0],sm.tri[best,2]-sm.tri[best,0]))/2),nearest_triangle=sm.tri[best].tolist())
   rows.append(row)
   print(json.dumps({k:v for k,v in row.items() if k!='nearest_triangle'}),flush=True)
 target=ROOT/'validation/KERNEL_POINT_PROBE_V1.json';assert not target.exists();target.write_text(json.dumps(dict(status='DIAGNOSTIC_BASELINE_PLUS_AXIS_OFFSETS_ONLY',records=rows,model_inference='fixed predicted MHR decode only',new_reconstruction=False,GT_loaded=False,PBS_JOBID=os.environ['PBS_JOBID']),indent=2)+'\n')
if __name__=='__main__':main()
