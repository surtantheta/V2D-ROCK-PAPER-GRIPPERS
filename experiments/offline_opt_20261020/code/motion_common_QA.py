#!/usr/bin/env python3
"""Common visibility metrics and overlays for previously frozen motion fits."""
import json
from pathlib import Path
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image,ImageDraw
import silhouette_pilot_v6 as core
from budget_and_motion import projected_human_band,smooth_translation

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020');PROJECT=ROOT.parents[1]
def main():
    source=ROOT/'runs/budget_and_motion_v1/RESULT.json';data=json.loads(source.read_text())
    out=ROOT/'runs/motion_common_QA_v1';out.mkdir()
    assets=json.loads((ROOT/'assets/budget_comparison_v1/ASSETS.json').read_text())
    mesh=np.load(assets['assets']['generated']['array_path']);v=torch.tensor(mesh['vertices'],device='cuda');f=torch.tensor(mesh['faces'],device='cuda')
    ctx=dr.RasterizeCudaContext();frames=list(range(432,464));hashes={str(source):core.digest(source),str(Path(__file__)):core.digest(__file__)}
    focal=json.loads((PROJECT/'work/s3_focal/episode_000000/FOCAL.json').read_text())['focal_source_px']
    images,target_np,kk,_,metadata=core.load_frames(PROJECT,0,frames,focal,256,hashes)
    target=torch.tensor(target_np,device='cuda');k=torch.tensor(kk,device='cuda')
    body=np.load(PROJECT/'work/s3db/episode_000000/sam3d_body.npz')['vertices_cam'];hf=torch.tensor(np.load(PROJECT/'models/mhr/mhr_faces_lod1.npy'),device='cuda',dtype=torch.int32)
    band=projected_human_band(ctx,body,hf,frames,metadata,k,256);visible=~(band & (target<.5));truth=target>.5
    def metrics(pred):
        pp=pred>.5
        def vals(t):return t.cpu().tolist()
        inter=(pp&truth).sum((1,2));mass=truth.sum((1,2)).clamp_min(1)
        return {'common_masked_iou':vals(inter/((pp|truth)&visible).sum((1,2)).clamp_min(1)),
                'observed_positive_recall':vals(inter/mass),
                'outside_target_and_humanband_mass_over_target':vals((pp&~truth&~band).sum((1,2))/mass),
                'predicted_hidden_area_over_target':vals((pp&band&~truth).sum((1,2))/mass),
                'full_projected_iou':core.ious(pred,target)}
    result=[];renders=[]
    for record in data['records']:
        if record['cohort']!='motion':continue
        rr=torch.tensor(record['R_obj2cam'],device='cuda',dtype=torch.float32)
        tt=np.array(record['t_obj2cam_m']);raw=core.render(ctx,v[None]@rr.transpose(-1,-2)+torch.tensor(tt,device='cuda',dtype=torch.float32)[:,None],f,k,[256,256])
        renders.append(raw.cpu().numpy())
        weights=np.maximum(np.asarray(record['final_iou']),.1)
        for penalty in [0.,.1,1.,10.,100.]:
            translation=smooth_translation(tt,weights,penalty)
            pred=core.render(ctx,v[None]@rr.transpose(-1,-2)+torch.tensor(translation,device='cuda',dtype=torch.float32)[:,None],f,k,[256,256])
            result.append({'occlusion_fit':record['occlusion_band'],'penalty':penalty,'metrics':metrics(pred),
                           'acceleration_magnitude_cm_per_frame2':float(np.linalg.norm(np.diff(translation,n=2,axis=0),axis=1).mean()*100)})
    # Target white, plain red, occlusion fit green, uncertainty band blue; same view/crop for both.
    canvas=Image.new('RGB',(4*256,3*280),(20,20,20));draw=ImageDraw.Draw(canvas)
    for col,idx in enumerate([0,10,20,31]):
        base=np.array(images[idx]).astype(np.float32)
        for row,mask in enumerate([target_np[idx],renders[0][idx],renders[1][idx]]):
            img=base.copy();bb=band[idx].cpu().numpy();img[bb]=.75*img[bb]+.25*np.array([60,100,255])
            color=np.array([[255,255,255],[255,70,50],[40,255,80]][row]);edge=(mask>.5)
            img[edge]=.5*img[edge]+.5*color
            canvas.paste(Image.fromarray(np.uint8(img.clip(0,255))),(col*256,row*280+24));draw.text((col*256+4,row*280+4),f'{["target","plain","human band fit"][row]} frame {frames[idx]}',fill='white')
    canvas.save(out/'MOTION_QA.png')
    report={'schema':'v2d-motion-common-QA-v1','records':result,'source_sha256':hashes,'definition':'Same target, predicted-human band and known-visible set for both arms; retain all target positive pixels. Band is a heuristic without depth ordering.',
            'metric_scale_recovered':False,'GT_accuracy_measured':False,'smoothing_weights':'raw fit IoU; retained as declared diagnostic, may penalize amodal coverage','no_model_promotion_from_these_metrics':True}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for x in result:print(x['occlusion_fit'],x['penalty'],{k:float(np.mean(v)) for k,v in x['metrics'].items()},'acc',x['acceleration_magnitude_cm_per_frame2'],flush=True)
if __name__=='__main__':main()
