#!/usr/bin/env python3
"""Paired fixed-mesh mask refinement after RGB-only FoundationPose tracking."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as functional
# Shared nvdiffrast supports newer GPUs; isolated build is needed on Turing only.
if torch.cuda.get_device_capability()==(7,5):sys.path.insert(0,'/home/tanuj/V2D_challenge/experiments/offline_opt_20261020/deps/sm75')
import nvdiffrast.torch as dr
import trimesh
import silhouette_pilot_v6 as core
from budget_and_motion import smooth_translation
from labeled_object_scorer import POSE_CONVENTION
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def crop_observations(item,rows,K):
    frames=[];targets=[];bands=[];intrinsics=[]
    for i,row in enumerate(rows):
        if not row['selected'] or row['score']<.3:continue
        mask=np.asarray(Image.open(row['mask']))>127;ys,xs=np.nonzero(mask)
        if len(xs)<50:continue
        side=int(np.ceil(1.5*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)));left=int(np.floor(xs.mean()+.5-side/2));top=int(np.floor(ys.mean()+.5-side/2));box=(left,top,left+side,top+side)
        target=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).crop(box).resize((256,256),Image.Resampling.NEAREST),np.float32)/255
        person=Image.open(row['person_mask']).crop(box).resize((256,256),Image.Resampling.NEAREST)
        person=torch.tensor(np.asarray(person)>127,device='cuda',dtype=torch.float32)[None,None]
        band=functional.max_pool2d(person,11,stride=1,padding=5)[0,0]>.1
        affine=np.array([[256/side,0,-left*256/side],[0,256/side,-top*256/side],[0,0,1.]])
        frames.append(i);targets.append(target);bands.append(band);intrinsics.append(affine@K)
    if not frames:return [],None,None,None
    return frames,torch.tensor(np.stack(targets),device='cuda'),torch.stack(bands),torch.tensor(np.stack(intrinsics),device='cuda',dtype=torch.float32)
def main():
    out=ROOT/'labeled_predictions_v1/comparison';out.mkdir(exist_ok=False)
    inputs=ROOT/'labeled_predictions_v1/RUN_INPUTS.json';budget=ROOT/'labeled_predictions_v1/BUDGETED.json';manifest=json.loads(inputs.read_text());budgeted={x['sequence']:x for x in json.loads(budget.read_text())['records']}
    context=dr.RasterizeCudaContext();gates=core.renderer_gate(context,'cuda');gates.update(core.optimizer_gate(context,'cuda'));original_loss=core.loss_silhouette
    arms={x:[] for x in ['foundationpose','plain','person_band','person_band_smooth1','person_band_high_confidence','person_band_high_confidence_smooth1']};summaries=[]
    sources={str(x):sha(x) for x in [Path(__file__),Path(core.__file__),inputs,budget,ROOT/'code/budget_and_motion.py']}
    for item in manifest['items']:
        if item['dataset']!='BEHAVE' or item['name'] not in budgeted:continue
        asset=budgeted[item['name']];objdir=Path(asset['directory']);meshfile=Path(asset['mesh_path']);assert sha(meshfile)==asset['mesh_sha256'];mesh=trimesh.load(meshfile,process=False)
        vv=torch.tensor(np.asarray(mesh.vertices),device='cuda',dtype=torch.float32);ff=torch.tensor(np.asarray(mesh.faces),device='cuda',dtype=torch.int32)
        kf=json.loads((objdir/'keyframe.json').read_text());K=np.array(kf['K_predicted']);mp=Path(item['masks'])/'MASKS.json';rows=json.loads(mp.read_text())['rows']
        track=ROOT/'labeled_predictions_v1/tracks'/item['name']/'poses.npz';base=np.load(track)['T_obj2cam'];assert len(base)==len(rows) and np.isfinite(base).all()
        for path in [track,mp,meshfile,objdir/'keyframe.json']:sources[str(path)]=sha(path)
        frames,target,band,k=crop_observations(item,rows,K);indices=np.array(frames,dtype=np.int64)
        if frames:
            rotation=torch.tensor(base[indices,:3,:3],device='cuda',dtype=torch.float32);translation=torch.tensor(base[indices,:3,3],device='cuda',dtype=torch.float32)
            vc=vv[None]@rotation.transpose(-1,-2)+translation[:,None];rendered=core.render(context,vc,ff,k,[256,256]);eligible=(vc[...,2].min(1).values>.05)&(rendered.sum((1,2))>=.1*target.sum((1,2)))
            idx=torch.where(eligible)[0];indices=indices[idx.cpu().numpy()];target=target[idx].contiguous();band=band[idx].contiguous();k=k[idx].contiguous();initial_iou=core.ious(rendered[idx],target)
        else:indices=np.array([],dtype=np.int64);initial_iou=[]
        candidates={'foundationpose':base.copy()};history={};raw_iou={}
        if len(indices):
            aa=Rotation.from_matrix(base[indices,:3,:3]).as_euler('xyz');tt=base[indices,:3,3].copy();tt[:,2]=np.log(tt[:,2])
            for name in ['plain','person_band']:
                if name=='person_band':
                    visible=~(band&(target<.5))
                    def visible_loss(pred,observed,distance):
                        offset=(observed.data_ptr()-target.data_ptr())//(target.element_size()*256*256);weights=visible[offset:offset+len(observed)];mass=observed.sum((1,2)).clamp_min(1)
                        return (((pred-observed).square()*weights).sum((1,2))/mass+.2*(pred*distance*weights).sum((1,2))/mass).mean()
                    core.loss_silhouette=visible_loss
                else:core.loss_silhouette=original_loss
                fitted,h=core.fit(context,vv,ff,target,k,aa,tt,fit_shape=False);poses=base.copy();poses[indices,:3,:3]=fitted[1].cpu().numpy();poses[indices,:3,3]=fitted[2].cpu().numpy();candidates[name]=poses;history[name]=h;raw_iou[name]=core.ious(fitted[3],target)
        else:
            candidates['plain']=base.copy();candidates['person_band']=base.copy()
        core.loss_silhouette=original_loss
        high=candidates['person_band'].copy();low=np.array([r['score']<.8 for r in rows]);high[low]=base[low];candidates['person_band_high_confidence']=high
        high_weights=np.full(len(base),.1)
        if len(indices):high_weights[indices]=np.maximum(raw_iou['person_band'],.1)
        high_weights[low]=.1
        high_smoothed=high.copy();high_smoothed[:,:3,3]=smooth_translation(high[:,:3,3],high_weights,1.);candidates['person_band_high_confidence_smooth1']=high_smoothed
        smoothed=candidates['person_band'].copy();weights=np.full(len(base),.1)
        if len(indices):weights[indices]=np.maximum(raw_iou['person_band'],.1)
        smoothed[:,:3,3]=smooth_translation(smoothed[:,:3,3],weights,1.);candidates['person_band_smooth1']=smoothed
        seqsummary={'sequence':item['name'],'total_frames':len(base),'mask_threshold_0_3_frames':len(frames),'mask_threshold_0_8_frames':sum(r['selected'] and r['score']>=.8 for r in rows),'refinement_eligible_frames':indices.tolist(),'fallback_frames_before_smoothing':len(base)-len(indices),'initial_eligible_mean_iou':float(np.mean(initial_iou)) if initial_iou else None,'fit_history':history,'mask_metrics':{}}
        seqsummary['changed_frame_counts_by_arm']={name:int((np.max(np.abs(poses-base),axis=(1,2))>1e-7).sum()) for name,poses in candidates.items()}
        seqsummary['smoothing_modifies_fallback_translations']=True
        for name,poses in candidates.items():
            if len(indices):
                rr=torch.tensor(poses[indices,:3,:3],device='cuda',dtype=torch.float32);tt=torch.tensor(poses[indices,:3,3],device='cuda',dtype=torch.float32)
                pred=core.render(context,vv[None]@rr.transpose(-1,-2)+tt[:,None],ff,k,[256,256]);pp=pred>.5;truth=target>.5;visible=~(band&~truth)
                seqsummary['mask_metrics'][name]={'full_iou':core.ious(pred,target),'common_masked_iou':((pp&truth).sum((1,2))/((pp|truth)&visible).sum((1,2)).clamp_min(1)).cpu().tolist()}
            npz=out/f'{item["name"]}_{name}.npz';np.savez_compressed(npz,T_obj2cam=poses)
            record={'sequence':item['name'],'camera':1,'mesh_path':str(meshfile),'mesh_sha256':sha(meshfile),'mesh_units':'metres','object_scale':1.,'pose_convention':POSE_CONVENTION,'origin_convention':kf['origin_convention'],'origin_correspondence_to_gt':'unverified','source':{'run_inputs_sha256':sha(inputs),'poses_path':str(npz),'poses_sha256':sha(npz),'geometry_frozen':True,'native_GT_inputs_provided':False,'K_depth_source':'RGB-predicted MoGe; uncalibrated metric scale','optimizer':name},'frames':[{'video_frame_index':int(Path(item['source_paths'][i]).stem),'R_obj2cam':poses[i,:3,:3].tolist(),'t_obj2cam_m':poses[i,:3,3].tolist()} for i in range(len(poses))]}
            arms[name].append(record)
        summaries.append(seqsummary);print('completed paired compare',item['name'],'eligible',len(indices),'/',len(base),flush=True)
    for name,records in arms.items():(out/f'{name}_PREDICTIONS.json').write_text(json.dumps({'schema':'opencv-metric-object-predictions-v1','sequences':records},indent=2,allow_nan=False))
    (out/'RESULT.json').write_text(json.dumps({'gates':gates,'sequences':summaries,'source_sha256':sources,'geometry_frozen':True,'GT_used_by_optimization':False,'smoothing_penalty_predeclared':1.,'person_band':'predicted SAM3 person mask, dilated5crop pixels; negative target pixels only; no depth ordering'},indent=2,allow_nan=False))
if __name__=='__main__':main()
