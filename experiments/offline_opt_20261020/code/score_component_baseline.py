#!/usr/bin/env python3
"""Early full-cohort FP baseline check while independent pose refinement runs."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from labeled_object_scorer import POSE_CONVENTION
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    runtime=ROOT/'labeled_predictions_v1';out=runtime/'component_baseline';out.mkdir(exist_ok=False);inputs=runtime/'RUN_INPUTS.json';manifest=json.loads(inputs.read_text());bp=runtime/'BUDGETED.json';assets={x['sequence']:x for x in json.loads(bp.read_text())['records']};hp=runtime/'human_scale/RESULT.json';anchors={x['sequence']:x['alpha'] for x in json.loads(hp.read_text())['records']};sources={str(x):sha(x) for x in [Path(__file__),inputs,bp,hp]}
    for arm in ['raw','predicted_human_scale']:
        records=[]
        for item in manifest['items']:
            if item['dataset']!='BEHAVE' or item['name'] not in assets:continue
            asset=assets[item['name']];mesh=Path(asset['mesh_path']);track=runtime/'tracks'/item['name']/'poses.npz';T=np.load(track)['T_obj2cam'];sources[str(track)]=sha(track);sources[str(mesh)]=sha(mesh);alpha=1. if arm=='raw' else anchors[item['name']];assert alpha is not None and alpha>0
            records.append({'sequence':item['name'],'camera':1,'mesh_path':str(mesh),'mesh_sha256':sha(mesh),'mesh_units':'metres','object_scale':alpha,'pose_convention':POSE_CONVENTION,'origin_convention':'generated mesh mean recentered and exact pose compensation','origin_correspondence_to_gt':'unverified','source':{'early_component_baseline':True,'raw_poses_sha256':sha(track),'human_anchor_result_sha256':sha(hp),'alpha':alpha,'no_GT_used_by_inference':True},'frames':[{'video_frame_index':int(Path(path).stem),'R_obj2cam':T[i,:3,:3].tolist(),'t_obj2cam_m':(alpha*T[i,:3,3]).tolist()} for i,path in enumerate(item['source_paths'])]})
        pred=out/f'{arm}_PREDICTIONS.json';pred.write_text(json.dumps({'schema':'opencv-metric-object-predictions-v1','sequences':records},indent=2));scores=out/f'{arm}_SCORES.json'
        subprocess.run(['/home/tanuj/.conda/envs/cari4d/bin/python',str(ROOT/'code/labeled_object_scorer.py'),'--prediction',str(pred),'--requests',str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),'--references',str(ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'),'--out',str(scores)],check=True)
    (out/'SOURCE_RECEIPT.json').write_text(json.dumps(sources,indent=2));print('completed full384-frame paired baseline component check',flush=True)
if __name__=='__main__':main()
