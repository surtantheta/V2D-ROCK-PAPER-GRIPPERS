#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    kit=ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit';sys.path.insert(0,str(kit));from v2dlb.mesh_budget import budget_mesh
    result=ROOT/'labeled_predictions_v2/shapes/RESULT.json';data=json.loads(result.read_text());records=[]
    old_budget=ROOT/'labeled_predictions_v1/BUDGETED.json';reuse={x['sequence']:x for x in json.loads(old_budget.read_text())['records']}
    audit=ROOT/'labeled_predictions_v2/SEMANTIC_MASK_AUDIT.json';changed={x['sequence'] for x in json.loads(audit.read_text())['records'] if x['rejected_count']>0}
    for seq in data['records']:
        if seq['sequence'] not in changed:records.append(reuse[seq['sequence']]);continue
        if seq['status']!='decoded':continue
        proposal=next(x for x in seq['proposals'] if x['local_frame_index']==seq['image_selected_proposal_local_index'])
        src=Path(proposal['mesh_path']);assert sha(src)==proposal['mesh_sha256'];v,f=budget_mesh(src,faces=4096,vertices=4096)
        dest=ROOT/'labeled_predictions_v2/budgeted'/seq['sequence'];dest.mkdir(parents=True,exist_ok=False);mesh=dest/'object_mesh.ply';trimesh.Trimesh(v,f,process=False).export(mesh)
        record=dict(proposal,keyframe=proposal['local_frame_index'],focal=proposal['K_predicted'][0][0],t_obj2cam=proposal['t_obj2cam_m'],budgeted_mesh_sha256=sha(mesh),budget_faces=4096,budget_vertices=4096,budget_origin_preserved=True,selection_uses_GT=False)
        (dest/'keyframe.json').write_text(json.dumps(record,indent=2));records.append({'sequence':seq['sequence'],'directory':str(dest),'mesh_path':str(mesh),'mesh_sha256':sha(mesh)})
    (ROOT/'labeled_predictions_v2/BUDGETED.json').write_text(json.dumps({'records':records,'source_sha256':{str(x):sha(x) for x in [Path(__file__),result,old_budget,audit,kit/'v2dlb/mesh_budget.py']}},indent=2))
if __name__=='__main__':main()
