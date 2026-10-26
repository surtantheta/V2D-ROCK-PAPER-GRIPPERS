#!/usr/bin/env python3
"""Sparse RGB-predicted human surface anchors a constant scene scale per clip."""
import hashlib,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020');PROJECT=ROOT.parents[1]
LIB=PROJECT/'video_to_data/reconstruction/modules/v2d_sam3d_body/lib'
WEIGHTS=PROJECT/'models/sam3d_body/sam-3d-body-dinov3'
BARY=np.array([(a,b,1-a-b) for a in np.linspace(0,1,5) for b in np.linspace(0,1,5) if a+b<=1+1e-9])
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def zbuffer(points,K,width,height):
    points=points[np.isfinite(points).all(1)&(points[:,2]>.001)];z=points[:,2]
    u=np.floor(points[:,0]/z*K[0,0]+K[0,2]).astype(int);v=np.floor(points[:,1]/z*K[1,1]+K[1,2]).astype(int)
    inside=(u>=0)&(u<width)&(v>=0)&(v<height);result=np.full((height,width),np.inf);np.minimum.at(result,(v[inside],u[inside]),z[inside]);return result

def self_test():
    K=np.array([[80.,0,30.],[0,75.,40.],[0,0,1.]])
    x,y=np.meshgrid(np.linspace(-.8,.8,100),np.linspace(-1,1,100));points=np.stack([x.ravel(),y.ravel(),np.full(x.size,3.)],axis=-1)
    buf=zbuffer(points,K,64,80);valid=np.isfinite(buf);alpha=np.median(buf[valid]/1.2);assert abs(alpha-2.5)<1e-10
    vertices=np.array([[.1,.2,.3],[-.2,.1,.2]]);t=np.array([.2,-.3,3.]);world=vertices+t;scaled=alpha*vertices+alpha*t
    assert np.max(abs(world[:,:2]/world[:,2:]-scaled[:,:2]/scaled[:,2:]))<1e-12
    return {'known_depth_factor_recovered':float(alpha),'uniform_camera_scale_preserves_projection':True,'independent_fx_fy_and_offcenter_K':True}

def main():
    gates=self_test();out=ROOT/'labeled_predictions_v1/human_scale';out.mkdir(exist_ok=False)
    receipt=ROOT/'labeled_predictions_v1/HUMAN_MODEL_SOURCE_RECEIPT.json';pins=json.loads(receipt.read_text())
    for path,expected in pins['source_sha256'].items():assert sha(path)==expected,path
    os.environ.setdefault('TORCH_HOME',str(PROJECT/'models/sam3d_body/torch_home'));sys.path.insert(0,str(LIB))
    from sam_3d_body import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
    model,cfg=load_sam_3d_body(checkpoint_path=str(WEIGHTS/'model.ckpt'),device='cuda',mhr_path=str(WEIGHTS/'assets/mhr_model.pt'));est=SAM3DBodyEstimator(sam_3d_body_model=model,model_cfg=cfg)
    torch.set_num_threads(4);faces_path=PROJECT/'models/mhr/mhr_faces_lod1.npy';kit_faces=np.load(faces_path);faces=np.asarray(est.faces,dtype=np.int64)
    assert faces.ndim==2 and faces.shape[1]==3 and faces.min()>=0
    gates['authoritative_model_head_topology']={'faces':len(faces),'max_vertex_index':int(faces.max()),'kit_faces_exactly_equal':bool(np.array_equal(faces,kit_faces))}
    inp=ROOT/'labeled_predictions_v1/RUN_INPUTS.json';inputs=json.loads(inp.read_text());sources={str(p):sha(p) for p in [Path(__file__),receipt,inp,faces_path]};records=[]
    for item in inputs['items']:
        if item['dataset']!='BEHAVE':continue
        depthdir=ROOT/'labeled_predictions_v1/depth'/item['name'];runpath=depthdir/'RUN.json'
        if not runpath.exists():records.append({'sequence':item['name'],'status':'no_predicted_depth','alpha':None});continue
        sources[str(runpath)]=sha(runpath);run=json.loads(runpath.read_text());K=np.array(run['K_fixed_source'],np.float32);W,H=run['source_wh'];rw,rh=run['model_wh'];kr=np.diag([rw/W,rh/H,1.])@K
        maskpath=Path(item['masks'])/'MASKS.json';rows=json.loads(maskpath.read_text())['rows'];sources[str(maskpath)]=sha(maskpath)
        seqdir=out/item['name'];seqdir.mkdir();ratios=[];perframe=[]
        for i in range(0,len(rows),8):
            rgb=Path(item['source_paths'][i]);assert sha(rgb)==item['source_sha256'][i]
            pp=Path(rows[i]['person_mask']);assert sha(pp)==rows[i]['person_mask_sha256'];person=np.asarray(Image.open(pp))>127
            n,labels,stats,_=cv2.connectedComponentsWithStats(person.astype(np.uint8),8)
            if n<=1:perframe.append({'frame':i,'status':'empty_predicted_person'});continue
            component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);yy,xx=np.nonzero(labels==component);lo=np.array([xx.min(),yy.min()],float);hi=np.array([xx.max()+1,yy.max()+1],float);center=(lo+hi)/2;extent=(hi-lo)*1.15
            box=np.r_[center-extent/2,center+extent/2].astype(np.float32);box[[0,2]]=box[[0,2]].clip(0,W-1);box[[1,3]]=box[[1,3]].clip(0,H-1)
            t0=time.time();image=np.asarray(Image.open(rgb).convert('RGB'))
            with torch.inference_mode():pred=est.process_one_image(img=image,bboxes=box[None],masks=None,cam_int=torch.tensor(K)[None],inference_type='full')
            if len(pred)!=1:perframe.append({'frame':i,'status':'unexpected_body_output_count','count':len(pred)});continue
            def arr(x):return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x,np.float32)
            verts=arr(pred[0]['pred_vertices']).reshape(-1,3)+arr(pred[0]['pred_cam_t']).reshape(1,3)
            assert faces.max()<len(verts)
            if not np.isfinite(verts).all():perframe.append({'frame':i,'status':'nonfinite_body_prediction'});continue
            triangles=verts[faces];points=np.einsum('sk,fkd->fsd',BARY,triangles).reshape(-1,3);buf=zbuffer(points,kr,rw,rh)
            depthpath=depthdir/f'{i:06d}.npz';depth=np.load(depthpath)['depth'].astype(float);sources[str(depthpath)]=sha(depthpath)
            objectmask=Path(rows[i]['mask']);assert sha(objectmask)==rows[i]['mask_sha256']
            person_small=cv2.resize(person.astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST)>0
            obj_small=cv2.resize((np.asarray(Image.open(objectmask))>127).astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST)
            exclusion=cv2.dilate(obj_small,np.ones((5,5),np.uint8))>0
            finite=np.isfinite(buf);interior=cv2.erode(finite.astype(np.uint8),np.ones((3,3),np.uint8))>0
            valid=interior&person_small&~exclusion&np.isfinite(depth)&(depth>1e-3)
            r=buf[valid]/depth[valid];r=r[np.isfinite(r)&(r>0)]
            rec={'frame':i,'pixels':len(r),'box_xyxy':box.tolist(),'ratio_median':float(np.median(r)) if len(r)>=200 else None,'status':'used' if len(r)>=200 else 'insufficient_surface_pixels','seconds':time.time()-t0};perframe.append(rec)
            if len(r)>=200:ratios.append(r)
            np.savez_compressed(seqdir/f'body_{i:06d}.npz',vertices_cam=verts,K=K,box=box)
            sources[str(pp)]=sha(pp);sources[str(objectmask)]=sha(objectmask)
            print(item['name'],i,'pixels',len(r),'alpha',rec['ratio_median'],flush=True)
        alpha=float(np.median(np.concatenate(ratios))) if ratios else None
        record={'sequence':item['name'],'status':'estimated' if alpha else 'no_valid_anchor','alpha':alpha,'frames':perframe,'accepted_frames':sum(x['status']=='used' for x in perframe),'frame_median_std':float(np.std([x['ratio_median'] for x in perframe if x['status']=='used'])) if ratios else None,'frame_median_MAD':float(np.median(np.abs(np.array([x['ratio_median'] for x in perframe if x['status']=='used'])-np.median([x['ratio_median'] for x in perframe if x['status']=='used'])))) if ratios else None,'ratio_pixel_percentiles':{str(q):float(np.percentile(np.concatenate(ratios),q)) for q in [5,25,50,75,95]} if ratios else None,'no_anchor_fallback_policy':'identity scale1 applied explicitly by paired rescoring','fixed_scale_per_clip':True,'GT_used':False,'anchor':'RGB-predicted SAM3D Body visible surface vs RGB-predicted MoGe depth, intersect predicted person and exclude predicted object','origin':'uniform camera-origin scale; multiply fixed mesh and every t by same alpha','sampled_frames_step':8};records.append(record);(seqdir/'SCALE.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    (out/'RESULT.json').write_text(json.dumps({'schema':'predicted-human-global-scene-scale-v1','gates':gates,'records':records,'source_sha256':sources,'model_provenance':pins,'GT_inputs_provided':False,'native_object_geometry_provided':False},indent=2,allow_nan=False));print('completed predicted human metric anchor',flush=True)
if __name__=='__main__':main()
