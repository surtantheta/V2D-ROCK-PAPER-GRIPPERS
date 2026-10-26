#!/usr/bin/env python3
"""CPU kit4096 budget/FP assets for generated InterCap development proposals."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 kit=ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit';sys.path.insert(0,str(kit));from v2dlb.mesh_budget import budget_mesh
 source=ROOT/'labeled_intercap_v1/shapes/RESULT.json';manifest=ROOT/'labeled_intercap_v1/RUN_INPUTS.json';data=json.loads(source.read_text());runtime=json.loads(manifest.read_text());items={x['take']:x for x in runtime['items']};records=[]
 assert not data['GT_inputs_provided'] and not data['reserve_takes_included'];sources={str(p):sha(p) for p in [Path(__file__),source,manifest,kit/'v2dlb/mesh_budget.py']}
 for seq in data['records']:
  if seq['status']!='decoded':continue
  item=items[seq['take']];proposal=next(x for x in seq['proposals'] if x['local_frame_index']==seq['image_selected_proposal_local_index']);src=Path(proposal['mesh_path']);assert sha(src)==proposal['mesh_sha256'];v,f=budget_mesh(src,faces=4096,vertices=4096)
  assert v.shape==f.shape==(4096,3) and np.isfinite(v).all();tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;assert (area>1e-14).any()
  dest=ROOT/'labeled_intercap_v1/budgeted'/item['name'];dest.mkdir(parents=True,exist_ok=False);mesh=dest/'object_mesh.ply';trimesh.Trimesh(v,f,process=False).export(mesh)
  key=dict(proposal,source_proposal_mesh_path=str(src),source_proposal_mesh_sha256=sha(src),mesh_path=str(mesh),mesh_sha256=sha(mesh),keyframe=proposal['local_frame_index'],focal=proposal['K_predicted'][0][0],t_obj2cam=proposal['t_obj2cam_m'],budgeted_mesh_sha256=sha(mesh),budget_faces=4096,budget_vertices=4096,budget_origin_preserved=True,selection_uses_GT=False,source_frame_cadence=5,nominal_source_fps=30.,source_timestamp_basis='native frame/30 nominal, stride5 retained samples')
  (dest/'keyframe.json').write_text(json.dumps(key,indent=2));records.append({'sequence':seq['take'],'take':seq['take'],'name':item['name'],'directory':str(dest),'mesh_path':str(mesh),'mesh_sha256':sha(mesh),'frames':item['frames'],'masks':item['masks'],'depth':item['depth'],'source_paths':item['source_paths'],'source_sha256':item['source_sha256'],'native_frame_numbers':item['native_frame_numbers'],'keyframe':proposal['local_frame_index'],'video_frame_index':proposal['video_frame_index']});sources[str(src)]=sha(src)
 report={'schema':'intercap-dev-FP-runtime-v1','records':records,'source_sha256':sources,'GT_inputs_provided':False,'native_CAD_input':False,'reserve_takes_included':False,'complete_take_coverage':len(records)==8,'nominal_source_fps':30.,'source_frame_cadence':5,'cadence_disclosure':'native RGB retained every5sourceframes; actual hardware timestamps unavailable; do not report exact hosted ACC'}
 (ROOT/'labeled_intercap_v1/BUDGETED.json').write_text(json.dumps({'records':[{k:x[k] for k in ['sequence','take','name','directory','mesh_path','mesh_sha256']} for x in records],'source_sha256':sources},indent=2));(ROOT/'labeled_intercap_v1/FP_RUNTIME.json').write_text(json.dumps(report,indent=2));print(json.dumps({'budgeted_takes':len(records),'complete':len(records)==8},indent=2))
if __name__=='__main__':main()
