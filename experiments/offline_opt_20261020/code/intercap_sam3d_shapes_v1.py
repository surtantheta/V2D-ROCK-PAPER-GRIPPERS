#!/usr/bin/env python3
"""Fresh eight-take InterCap development SAM3D geometry; predicted RGB/person/object masks only."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import trimesh

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
SAM=Path('/rhome/tanuj/projects/sam-3d-objects')
F=np.diag([-1.,-1.,1.]);GLB_ROT=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quat(q):
    w,x,y,z=np.asarray(q,float).reshape(4)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
    sys.path[:0]=[str(SAM),str(SAM/'notebook')]
    from inference import Inference
    from sam3d_mesh_contract_snapshot_v1 import decimate,raster_iou
    manifest=ROOT/'labeled_intercap_v1/RUN_INPUTS.json';inp=json.loads(manifest.read_text());assert not inp['GT_fields_included']
    outdir=ROOT/'labeled_intercap_v1/shapes';outdir.mkdir(exist_ok=False)
    source={str(p):sha(p) for p in [Path(__file__),manifest,SAM/'checkpoints/hf/pipeline.yaml',SAM/'notebook/inference.py',SAM/'sam3d_objects/pipeline/inference_pipeline_pointmap.py',ROOT/'code/sam3d_mesh_contract_snapshot_v1.py',ROOT/'code/budget_intercap_shapes_v1.py']}
    print('GPU capability',torch.cuda.get_device_capability(),'stock decoded path previously ran on RTX8000(sm75); compiled PyTorch3D ops excluded',flush=True)
    from model_source_guard import verify_model_group
    model_provenance=verify_model_group('SAM3D_Objects_and_MoGe')
    assert inp.get('reserve_takes_included') is False and inp.get('native_CAD_input') is False
    assert len(inp['items'])==8 and sum(len(i['source_paths']) for i in inp['items'])==246
    person_receipt=Path(inp['person_masks_complete_receipt'])
    assert sha(person_receipt)==inp['person_masks_complete_receipt_sha256']
    person_data=json.loads(person_receipt.read_text());assert person_data['takes']==8 and person_data['frames']==246
    for entry in person_data['records']:assert sha(entry['mask_manifest'])==entry['mask_manifest_sha256']
    source[str(person_receipt)]=sha(person_receipt)
    model=Inference(str(SAM/'checkpoints/hf/pipeline.yaml'),compile=False)
    original=model._pipeline.compute_pointmap;capture={};fixed_camera={}
    def recorded(image,pointmap=None):
        if pointmap is None:result=original(image,None)
        else:
            # Supplied P3D pointmap already has known predicted K. Avoid the installed
            # pointmap intrinsics-recovery wrapper's swapped (focal,shift) return bug.
            loaded=torch.from_numpy(model._pipeline.image_to_float(image))
            points=pointmap.to(model._pipeline.device).permute(2,0,1)
            points=model._pipeline._clip_pointmap(points,loaded[...,-1])
            result={'pts_color':loaded.permute(2,0,1).contiguous()[:3],
                    'pointmap':points,'intrinsics':torch.tensor(fixed_camera['K_normalized'],device=model._pipeline.device,dtype=torch.float32)}
        capture.clear();capture['K_normalized']=result['intrinsics'].detach().cpu().numpy();capture['pointmap_P3D']=result['pointmap'].detach().cpu().numpy();return result
    model._pipeline.compute_pointmap=recorded
    records=[]
    for item in inp['items']:
        if item['dataset']!='InterCap':continue
        mp=Path(item['masks'])/'MASKS.json';source[str(mp)]=sha(mp);rows=json.loads(mp.read_text())['rows'];eligible=[(r['area'],i) for i,r in enumerate(rows) if r['selected'] and r['score']>=.8];confidence_fallback=not bool(eligible)
        if confidence_fallback:eligible=[(r['area'],i) for i,r in enumerate(rows) if r['selected'] and r['score']>=.3]
        picked=[]
        spread=max(1,len(rows)//3)
        for area,i in sorted(eligible,reverse=True):
            if all(abs(i-j)>=spread for j in picked):picked.append(i)
            if len(picked)==3:break
        assert all('person_duplicate_gate' in row and row['person_mask'] for row in rows)
        for row in rows:
            assert not (row['selected'] and row['person_duplicate_gate']['rejected'])
        seqdir=outdir/item['name'];seqdir.mkdir();proposals=[]
        # Independent RGB-only MoGe depth for every frame, used by fixed-mesh tracking later.
        depthdir=ROOT/'labeled_intercap_v1/depth'/item['name'];depthdir.mkdir(parents=True)
        raw_intrinsics=[]
        for frame_index,rgbpath in enumerate(item['source_paths']):
            assert sha(rgbpath)==item['source_sha256'][frame_index],rgbpath
            im=np.asarray(Image.open(rgbpath).convert('RGB'));hh,ww=im.shape[:2]
            rgba_depth=np.concatenate([im,np.full((hh,ww,1),255,np.uint8)],axis=-1)
            with torch.inference_mode():point_prediction=original(rgba_depth)
            pm=point_prediction['pointmap'].detach().float().cpu().numpy();kn_depth=point_prediction['intrinsics'].detach().float().cpu().numpy().reshape(3,3)
            raw_intrinsics.append(np.diag([ww,hh,1.])@kn_depth)
            d=np.asarray(Image.fromarray(pm[2]).resize((512,384),Image.Resampling.BILINEAR));d=np.where(np.isfinite(d)&(d>0),d,0.)
            np.savez_compressed(depthdir/f'{frame_index:06d}.npz',depth=d.astype(np.float16),K_source=np.diag([ww,hh,1.])@kn_depth)
            del point_prediction;torch.cuda.empty_cache()
            if frame_index%16==0:print('predicted_MoGe_depth',item['name'],frame_index,'/',len(item['source_paths']),flush=True)
        K_fixed=np.median(np.asarray(raw_intrinsics),axis=0);K_fixed[2]=[0.,0.,1.]
        assert np.isfinite(K_fixed).all() and K_fixed[0,0]>0 and K_fixed[1,1]>0
        fixed_camera['K_normalized']=np.diag([1/ww,1/hh,1.])@K_fixed
        (depthdir/'RUN.json').write_text(json.dumps({'K_fixed_source':K_fixed.tolist(),'raw_K_source_per_frame':np.asarray(raw_intrinsics).tolist(),'model_wh':[512,384],'source_wh':[ww,hh],'source':'RGB-predicted MoGe via SAM3D internal compute_pointmap','metric_scale_recovered':False,'external_human_calibration':False,'frames':len(item['source_paths'])},indent=2))
        if not picked:
            records.append({'sequence':item['take'],'take':item['take'],'name':item['name'],'status':'no_predicted_mask','proposals':[]})
            (seqdir/'PROPOSALS.json').write_text(json.dumps(records[-1],indent=2))
            continue
        for i in picked:
            t0=time.time();image_path=Path(item['source_paths'][i]);mask_path=Path(rows[i]['mask']);assert sha(image_path)==item['source_sha256'][i] and sha(mask_path)==rows[i]['mask_sha256']
            image=np.asarray(Image.open(image_path).convert('RGB'));mask=np.asarray(Image.open(mask_path))>127;H,W=mask.shape
            assert sha(rows[i]['person_mask'])==rows[i]['person_mask_sha256']
            person=np.asarray(Image.open(rows[i]['person_mask']))>127
            overlap=(mask&person).sum()/max((mask|person).sum(),1);inside=(mask&person).sum()/max(mask.sum(),1)
            assert not (overlap>=.8 and inside>=.95),'selected proposal duplicates predicted person'
            rgba=model.merge_mask_to_rgba(image,mask)
            depth_input=np.load(depthdir/f'{i:06d}.npz')['depth'].astype(np.float32)
            depth_full=np.asarray(Image.fromarray(depth_input).resize((W,H),Image.Resampling.BILINEAR));yy,xx=np.mgrid[:H,:W];xx=xx+.5;yy=yy+.5
            pointmap=np.stack([-(xx-K_fixed[0,2])/K_fixed[0,0]*depth_full,-(yy-K_fixed[1,2])/K_fixed[1,1]*depth_full,depth_full],axis=-1).astype(np.float32)
            assert ((depth_full>0)&mask).sum()>=.5*mask.sum(),'insufficient predicted depth under object mask'
            pointmap[depth_full<=0]=np.nan
            with torch.inference_mode():prediction=model._pipeline.run(rgba,None,seed=42,stage1_only=False,with_mesh_postprocess=False,with_texture_baking=False,with_layout_postprocess=False,use_vertex_color=False,pointmap=torch.from_numpy(pointmap))
            glb=prediction['glb'];v=np.asarray(glb.vertices,float)@GLB_ROT.T;faces=np.asarray(glb.faces,np.int64)
            scale=prediction['scale'].detach().cpu().numpy().reshape(-1).astype(float);assert scale.size in [1,3] and (scale>0).all()
            vv=v*scale;vv,ff=decimate(vv,faces,20000);center=vv.mean(0);vv=vv-center
            rotation=F@quat(prediction['rotation'].detach().cpu().numpy()).T
            translation=F@prediction['translation'].detach().cpu().numpy().reshape(3)+rotation@center
            assert np.linalg.det(rotation)>.99999 and np.isfinite(vv).all()
            mesh_path=seqdir/f'proposal_{i:06d}.ply';trimesh.Trimesh(vv,ff,process=False).export(mesh_path)
            kn=np.asarray(capture['K_normalized']).reshape(3,3);K=K_fixed.copy();K_inferred=np.diag([W,H,1.])@kn
            pp=capture['pointmap_P3D'];depth=pp[2] if pp.shape[0]==3 else pp[...,2]
            np.savez_compressed(seqdir/f'proposal_{i:06d}_predicted_depth.npz',depth=depth.astype(np.float16),K=K)
            iou=raster_iou(vv@rotation.T+translation,ff,K,mask)
            rec={'local_frame_index':i,'video_frame_index':int(image_path.stem),'mesh_path':str(mesh_path),'mesh_sha256':sha(mesh_path),'R_obj2cam':rotation.tolist(),'t_obj2cam_m':translation.tolist(),'K_predicted':K.tolist(),'K_inferred_from_fixed_pointmap':K_inferred.tolist(),'K_policy':'sequence median RGB-predicted MoGe; pointmap reconstructed with same fixed K used by tracker/optimizer','scale_baked_into_mesh':scale.tolist(),'image_mask_iou':iou,'border_mask_allowed':True,
                 'mask_score':rows[i]['score'],'mask_confidence_fallback':confidence_fallback,'mask_confidence_policy':'prefer>=.8; use>=.3 only if no high-confidence observation exists, retain explicit fallback flag','mesh_vertices':len(vv),'mesh_faces':len(ff),'source_rgb':str(image_path),'source_rgb_sha256':sha(image_path),'source_predicted_mask':str(mask_path),'source_predicted_mask_sha256':sha(mask_path),'seconds':time.time()-t0,'source_person_mask':rows[i]['person_mask'],'source_person_mask_sha256':rows[i]['person_mask_sha256'],'person_duplicate_gate':rows[i]['person_duplicate_gate'],'candidate_spread_local_frames':spread,'source_frame_cadence':5,'nominal_source_fps':30.,'origin_convention':'generated-mesh vertex mean recentered, pose compensated; no correspondence to native GT mesh origin','metric_scale_recovered':False,'depth_scale_source':'internal RGB-predicted MoGe; no human/native metric calibration'}
            proposals.append(rec);(seqdir/f'proposal_{i:06d}.json').write_text(json.dumps(rec,indent=2,allow_nan=False));print(item['name'],i,'maskiou',iou,'extent',np.ptp(vv,axis=0),'seconds',rec['seconds'],flush=True)
            del prediction,glb;torch.cuda.empty_cache()
        best=max(proposals,key=lambda x:x['image_mask_iou']);records.append({'sequence':item['take'],'take':item['take'],'name':item['name'],'status':'decoded','proposals':proposals,'image_selected_proposal_local_index':best['local_frame_index'],'selection_uses_GT':False})
        (seqdir/'PROPOSALS.json').write_text(json.dumps(records[-1],indent=2))
    assert any(x['status']=='decoded' for x in records),'no sequence decoded; stop comparison chain'
    report={'schema':'intercap-dev-fresh-SAM3D-proposals-v1','records':records,'source_sha256':source,'GT_inputs_provided':False,'model_provenance':model_provenance,'layout_post_optimization':False,'mesh_postprocess':False,'seed':42,'metric_scale_recovered':False,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    report.update(reserve_takes_included=False,frames=246,takes=8,source_frame_cadence=5,nominal_source_fps=30.,person_masks_are_fresh=True)
    assert len(records)==8
    (outdir/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('complete fresh InterCap shapes',flush=True)
    import subprocess
    budget_source=ROOT/'code/budget_intercap_shapes_v1.py'
    subprocess.run(['/usr/bin/env','PYTHONPATH='+str(ROOT/'deps/hunyuan21'),'/home/tanuj/.conda/envs/grail/bin/python',str(budget_source)],check=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        failure=ROOT/'labeled_intercap_v1/SHAPES_FAILURE.json'
        failure.write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc(),'GT_inputs_provided':False},indent=2))
        raise
