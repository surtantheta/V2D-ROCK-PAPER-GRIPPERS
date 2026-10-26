#!/usr/bin/env python3
"""Compare budgeted shapes and occlusion-aware pose fitting on actual V2D motion."""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as functional
import nvdiffrast.torch as dr
import silhouette_pilot_v6 as core


def projected_human_band(context,vertices,faces,frame_ids,metadata,intrinsics,size):
    result=[]
    for j,frame in enumerate(frame_ids):
        vv=torch.as_tensor(vertices[frame:frame+1],device='cuda',dtype=torch.float32)
        mask=core.render(context,vv,faces,intrinsics[j:j+1],[size,size])
        result.append(functional.max_pool2d(mask[:,None],11,stride=1,padding=5)[0,0]>.1)
    return torch.stack(result)


def smooth_translation(observed,weights,penalty):
    n=len(observed)
    second=np.diff(np.eye(n),n=2,axis=0)
    lhs=np.diag(weights)+penalty*(second.T@second)
    return np.linalg.solve(lhs,weights[:,None]*observed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('output exists')
    project=Path('/home/tanuj/V2D_challenge');t0=time.time()
    source_paths=[Path(__file__),Path(core.__file__),args.assets/'ASSETS.json',project/'work/s6/episode_000000/poses.npz',
                  project/'work/s3_focal/episode_000000/FOCAL.json',project/'models/mhr/mhr_faces_lod1.npy',
                  project/'work/s3db/episode_000000/sam3d_body.npz']
    sources={str(p):core.digest(p) for p in source_paths}
    assets=json.loads((args.assets/'ASSETS.json').read_text())
    focal=json.loads(source_paths[4].read_text())['focal_source_px']
    baseline=np.load(source_paths[3])['T_obj2cam']
    context=dr.RasterizeCudaContext();gates=core.renderer_gate(context,'cuda');gates.update(core.optimizer_gate(context,'cuda'))
    args.out.mkdir(parents=True)
    records=[]
    static=[0,35,90,743,15,60,730,770]
    original_loss=core.loss_silhouette
    for cohort,frames in [('static',static),('motion',list(range(432,464)))]:
        images,target_np,kk,_,metadata=core.load_frames(project,0,frames,focal,256,sources)
        target=torch.as_tensor(target_np,device='cuda');k=torch.as_tensor(kk,device='cuda')
        rotations=baseline[frames,:3,:3]
        aa=Rotation.from_matrix(rotations).as_euler('xyz')
        translation0=baseline[frames,:3,3].copy();translation0[:,2]=np.log(translation0[:,2])
        human_band=None
        if cohort=='motion':
            body=np.load(source_paths[6])['vertices_cam']
            human_faces=torch.as_tensor(np.load(source_paths[5]),device='cuda',dtype=torch.int32)
            human_band=projected_human_band(context,body,human_faces,frames,metadata,k,256)
            del body
        for name,asset in assets['assets'].items():
            if cohort=='motion' and name!='generated':continue
            artifact=Path(asset['array_path']);assert core.digest(artifact)==asset['sha256']
            sources[str(artifact)]=asset['sha256'];data=np.load(artifact)
            v=torch.as_tensor(data['vertices'],device='cuda');f=torch.as_tensor(data['faces'],device='cuda')
            for occlusion in ([False] if cohort=='static' else [False,True]):
                if occlusion:
                    visibility=torch.where(human_band & (target<.5),0.,1.)
                    def visible_loss(pred,observed,distance):
                        # Core fits each frame separately; match the stable observed tensor's storage offset.
                        offset=(observed.data_ptr()-target.data_ptr())//(target.element_size()*target.shape[1]*target.shape[2])
                        weights=visibility[offset:offset+len(observed)]
                        mass=observed.sum((1,2)).clamp_min(1)
                        return ((((pred-observed).square()*weights).sum((1,2))+.2*(pred*distance*weights).sum((1,2)))/mass).mean()
                    core.loss_silhouette=visible_loss
                else:core.loss_silhouette=original_loss
                fitted,history=core.fit(context,v,f,target,k,aa,translation0,fit_shape=False)
                initial=core.render(context,v[None]@torch.tensor(rotations,device='cuda',dtype=torch.float32).transpose(-1,-2)+torch.tensor(baseline[frames,:3,3],device='cuda',dtype=torch.float32)[:,None],f,k,[256,256])
                record={'cohort':cohort,'asset':name,'occlusion_band':occlusion,'frames':frames,
                        'initial_iou':core.ious(initial,target),'final_iou':core.ious(fitted[3],target),
                        'loss':fitted[0],'history':history,'R_obj2cam':fitted[1].cpu().tolist(),'t_obj2cam_m':fitted[2].cpu().tolist(),
                        'geometry_frozen':True,'all_frame_poses_fit_to_masks':True}
                if cohort=='motion':
                    observed=fitted[2].cpu().numpy();weights=np.maximum(np.asarray(record['final_iou']),.1)
                    record['translation_regularization_sweep']=[]
                    for penalty in [0.,.1,1.,10.,100.]:
                        tt=smooth_translation(observed,weights,penalty)
                        vv=v[None]@fitted[1].transpose(-1,-2)+torch.tensor(tt,device='cuda',dtype=torch.float32)[:,None]
                        rendered=core.render(context,vv,f,k,[256,256])
                        record['translation_regularization_sweep'].append({'penalty':penalty,'mask_iou':core.ious(rendered,target),
                            'acceleration_magnitude_cm_per_frame2':float(np.linalg.norm(np.diff(tt,n=2,axis=0),axis=1).mean()*100),
                            'mean_shift_cm':float(np.linalg.norm(tt-observed,axis=1).mean()*100),
                            'GT_acceleration_error_measured':False})
                    record['baseline_acceleration_magnitude_cm_per_frame2']=float(np.linalg.norm(np.diff(baseline[frames,:3,3],n=2,axis=0),axis=1).mean()*100)
                    record['human_uncertainty_band_fraction']=float(human_band.float().mean())
                    record['observed_object_positive_fraction_in_human_band']=float((human_band*target).sum()/target.sum())
                records.append(record)
                np.savez_compressed(args.out/f'{cohort}_{name}_occlusion{int(occlusion)}.npz',R=fitted[1].cpu().numpy(),t=fitted[2].cpu().numpy(),frames=frames)
                print(json.dumps({'cohort':cohort,'asset':name,'occlusion':occlusion,'initial_mean_iou':float(np.mean(record['initial_iou'])),
                                  'final_mean_iou':float(np.mean(record['final_iou']))}),flush=True)
    core.loss_silhouette=original_loss
    report={'schema':'v2d-budget-and-motion-v1','gates':gates,'records':records,'source_sha256':sources,
            'metric_scale_recovered':False,'GT_accuracy_measured':False,'mesh_origin_preserved':True,
            'occlusion_semantics':'Ignore negative pixels in dilated predicted-human projection; retain every positive object pixel. No front/back ordering asserted.',
            'regularization_semantics':'Observation-anchored translation solve; acceleration magnitude is not error against GT.',
            'seconds':time.time()-t0,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (args.out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('complete',args.out,'seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
