#!/usr/bin/env python3
"""Read-only CPU coverage/provenance validation for fresh InterCap FP assets."""
import hashlib,json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 base=ROOT/'labeled_intercap_v1';inputs=json.loads((base/'RUN_INPUTS.json').read_text());data=json.loads((base/'FP_RUNTIME.json').read_text());shapes=json.loads((base/'shapes/RESULT.json').read_text());expected={x['take']:x for x in inputs['items']};assert len(expected)==8 and sum(len(x['source_paths']) for x in expected.values())==246
 assert data['GT_inputs_provided'] is False and data['native_CAD_input'] is False and data['reserve_takes_included'] is False
 assert data['complete_take_coverage'] is True and len(data['records'])==8 and {x['take'] for x in data['records']}==set(expected)
 for p,h in data['source_sha256'].items():assert sha(p)==h,p
 summary=[];frames=0
 for rec in data['records']:
  take=rec['take'];item=expected[take];n=len(item['source_paths']);assert rec['sequence']==take and rec['name']==item['name'];assert rec['source_paths']==item['source_paths'] and rec['source_sha256']==item['source_sha256'];assert rec['native_frame_numbers']==item['native_frame_numbers'];kf=json.loads((Path(rec['directory'])/'keyframe.json').read_text());k=int(kf['keyframe']);assert 0<=k<n and k==rec['keyframe'];assert kf['video_frame_index']==rec['video_frame_index']==item['native_frame_numbers'][k]
  mesh=Path(rec['mesh_path']);assert sha(mesh)==rec['mesh_sha256']==kf['mesh_sha256']==kf['budgeted_mesh_sha256'];assert str(mesh)==kf['mesh_path'];m=trimesh.load(mesh,process=False);v=np.asarray(m.vertices);f=np.asarray(m.faces);assert v.shape==f.shape==(4096,3) and np.isfinite(v).all() and f.min()>=0 and f.max()<4096;positive=int((m.area_faces>1e-14).sum());assert positive>0
  R=np.asarray(kf['R_obj2cam']);t=np.asarray(kf['t_obj2cam_m']);assert np.isfinite(t).all() and t.shape==(3,) and t[2]>0;assert np.max(abs(R.T@R-np.eye(3)))<1e-5 and abs(np.linalg.det(R)-1)<1e-5;assert np.array_equal(t,np.asarray(kf['t_obj2cam']))
  dr=Path(rec['depth']);run=json.loads((dr/'RUN.json').read_text());K=np.asarray(run['K_fixed_source']);assert K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0;assert np.max(abs(K-np.asarray(kf['K_predicted'])))<1e-5 and run['frames']==n and run['model_wh']==[512,384] and run['source_wh']==[1920,1080]
  positive_depth=[]
  for i in range(n):
   d=np.load(dr/f'{i:06d}.npz')['depth'];assert d.shape==(384,512) and np.isfinite(d).all();positive_depth.append(float((d>0).mean()));assert positive_depth[-1]>.5
   assert sha(Path(rec['frames'])/f'{i:06d}.jpg')==item['source_sha256'][i]
  masks=json.loads((Path(rec['masks'])/'MASKS.json').read_text());assert len(masks['rows'])==n;assert masks['rows'][k]['selected'] and not masks['rows'][k]['person_duplicate_gate']['rejected'];assert sha(kf['source_rgb'])==kf['source_rgb_sha256']==item['source_sha256'][k];assert sha(kf['source_predicted_mask'])==kf['source_predicted_mask_sha256'];assert sha(kf['source_person_mask'])==kf['source_person_mask_sha256'];assert sha(kf['source_proposal_mesh_path'])==kf['source_proposal_mesh_sha256']
  for row in masks['rows']:
   assert sha(row['mask'])==row['mask_sha256'] and sha(row['person_mask'])==row['person_mask_sha256']
   object_pixels=np.asarray(Image.open(row['mask']))>127;person_pixels=np.asarray(Image.open(row['person_mask']))>127;assert object_pixels.shape==person_pixels.shape==(1080,1920);intersection=int((object_pixels&person_pixels).sum());overlap=intersection/max(int((object_pixels|person_pixels).sum()),1);inside=intersection/max(int(object_pixels.sum()),1);gate=row['person_duplicate_gate'];assert abs(overlap-gate['object_person_iou'])<1e-15 and abs(inside-gate['object_pixels_inside_person_fraction'])<1e-15;assert not(row['selected'] and overlap>=.8 and inside>=.95)
  proposal=next(x for x in shapes['records'] if x['take']==take);assert proposal['status']=='decoded' and proposal['image_selected_proposal_local_index']==k;assert k==max(proposal['proposals'],key=lambda p:p['image_mask_iou'])['local_frame_index']
  summary.append({'take':take,'frames':n,'keyframe_local':k,'keyframe_native':rec['video_frame_index'],'positive_faces':positive,'minimum_positive_depth_fraction':min(positive_depth),'proposal_count':len(proposal['proposals']),'mask_iou_selected':kf['image_mask_iou'],'fallback':kf['mask_confidence_fallback']});frames+=n
 assert frames==246
 out={'schema':'intercap-FP-runtime-independent-coverage-QA-v1','takes':8,'frames':frames,'complete':True,'records':summary,'GT_read':False,'reserve_read':False,'source_sha256':{str(p):sha(p) for p in [Path(__file__),base/'FP_RUNTIME.json',base/'RUN_INPUTS.json',base/'shapes/RESULT.json']},'cadence_disclosure':'native frame/30 nominal; stride5; exact hosted ACC not established'}
 dest=base/'preflight/FP_RUNTIME_QA.json';dest.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
