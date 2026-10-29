"""GPU-only RGB geometry proposal cache. No GT/scorer/CAD/body decoder imports."""
import argparse, hashlib, json, math, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from geometry_contract import as_w2c, estimate_pinhole, resize_map, unproject, validate

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,j):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');temp.replace(path)

def verify(pins):
    for p,h in pins.items():assert sha(p)==h,('source changed',p)

def npcpu(x):return x.detach().float().cpu().numpy()

def pi3_images(paths):
    # Exact official load_images_as_tensor sizing and resize (without directory scan).
    import torch
    from torchvision.transforms.functional import to_tensor
    w,h=Image.open(paths[0]).size;scale=math.sqrt(255000/(w*h))
    k,m=round(w*scale/14),round(h*scale/14)
    while k*m*196>255000:
        if k/m>w/h:k-=1
        else:m-=1
    wh=(max(1,k)*14,max(1,m)*14)
    return torch.stack([to_tensor(Image.open(p).convert('RGB').resize(wh,Image.Resampling.LANCZOS)) for p in paths])

def diagnostics(case,frames,depth,K,local,w2c,A,extra,variant):
    import cv2
    n,h,w=depth.shape;kn=[];knative=[];observations=[]
    for i,row in enumerate(frames):
        ki=np.linalg.inv(A[i])@K[i];kn.append(ki)
        crop=case['native_to_staged_crop_xyxy'];native=ki.copy()
        if crop is not None:native=np.array([[1,0,crop[0]],[0,1,crop[1]],[0,0,1.]])@ki
        knative.append(native)
        observation={'local_index':row['local_index'],'native_frame_id':row['native_frame_id'],
                     'time_seconds':row['time_seconds'],'mask_selected':row['mask_selected'],'mask_score':row['mask_score']}
        if row['mask']:
            mask=np.array(Image.open(row['mask']).convert('L'))>127
            assert list(mask.shape[::-1])==row['rgb_wh'],('mask/source RGB mismatch',row['mask'])
            # A maps staged-RGB continuous pixel coordinates into model coordinates.
            warped=cv2.warpPerspective(mask.astype(np.uint8),A[i],(w,h),flags=cv2.INTER_NEAREST)>0
            valid=warped&np.isfinite(local[i]).all(-1)&(depth[i]>0)
            observation['mask_pixels_model']=int(warped.sum());observation['valid_mask_points']=int(valid.sum())
            if valid.sum()>=20:
                points=local[i][valid]
                observation.update({'visible_surface_camera_median':np.median(points,axis=0).tolist(),
                                    'visible_surface_depth_q10_q50_q90':np.quantile(points[:,2],[.1,.5,.9]).tolist(),
                                    'warning':'Visible surface statistic, not full object centre or metric accuracy.'})
        observations.append(observation)
    c2w=np.linalg.inv(w2c);relative=w2c@c2w[0:1]
    tr=np.linalg.norm(relative[:,:3,3],axis=1)
    angles=np.degrees(np.arccos(np.clip((np.trace(relative[:,:3,:3],axis1=1,axis2=2)-1)/2,-1,1)))
    valid=depth[np.isfinite(depth)&(depth>0)]
    extra['predicted_camera_motion']={'translation_max_divided_by_median_depth':float(tr.max()/np.median(valid)),
                                      'rotation_max_deg':float(angles.max()),
                                      'interpretation':'Duplicate-first control should have zero motion; normal clips can be dynamic/static.'}
    return np.array(kn),np.array(knative),observations

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--model',required=True);ap.add_argument('--weights',required=True)
    args=ap.parse_args();root=Path(args.root);manifest=json.loads((root/'PILOT_INPUTS.json').read_text());freeze=json.loads((root/'SOURCE_FREEZE.json').read_text())
    verify(freeze['sha256']);spec=next(x for x in manifest['models'] if x['name']==args.model)
    import torch
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    sys.path.insert(0,spec['source']);weights=Path(args.weights)
    weight_pins={str(p):sha(p) for p in weights.iterdir() if p.name in spec['files']}
    assert len(weight_pins)==len(spec['files'])
    out=Path(manifest['large_output_root'])/args.model;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    if args.model=='vggt1b':
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        model=VGGT().eval();model.load_state_dict(torch.load(weights/'model.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda()
    elif args.model=='pi3x':
        from pi3.models.pi3x import Pi3X
        model=Pi3X.from_pretrained(str(weights),local_files_only=True).cuda().eval()
    else:
        from depth_anything_3.api import DepthAnything3
        model=DepthAnything3.from_pretrained(str(weights),local_files_only=True).cuda().eval()
    loaded=time.monotonic()-start
    receipt={'schema':'geometry-model-pilot-v1','model':spec,'weight_sha256':weight_pins,
             'source_freeze_sha256':sha(root/'SOURCE_FREEZE.json'),'torch':torch.__version__,
             'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
             'dtype':str(dtype),'model_load_seconds':loaded,'records':[],
             'no_gt_or_calibrated_camera_conditioning':True,'no_static_world_object_fusion':True}
    todo=[(c,'normal') for c in manifest['cases']]
    for case,variant in todo:
        label=f"episode_{case['episode']:06d}_{variant}";case_dir=out/label;case_dir.mkdir(exist_ok=True)
        frames=list(case['frames'])
        if variant=='duplicate_first':frames=[frames[0]]*16
        if variant=='reverse':frames=frames[::-1]
        paths=[x['rgb'] for x in frames];pins={p:manifest['input_sha256'][p] for p in set(paths)}
        pins.update({x['mask']:x['mask_sha256'] for x in frames if x['mask']})
        verify(pins);torch.cuda.reset_peak_memory_stats();t=time.monotonic();print('START',args.model,label,flush=True)
        try:
            extra={};aux_arrays={};sizes=[x['rgb_wh'] for x in frames]
            if args.model=='vggt1b':
                images=load_and_preprocess_images(paths,mode='crop').cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):pred=model(images)
                e,k=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:]);K=npcpu(k[0]);w2c=as_w2c(npcpu(e[0]))
                depth=npcpu(pred['depth'][0,...,0]);conf=npcpu(pred['depth_conf'][0]);local=unproject(depth,K).astype(np.float32)
                # Preserve independent point-head geometry separately from depth-unprojected camera points.
                direct_world=npcpu(pred['world_points'][0]);extra['point_head_available']=True
                aux_arrays['direct_world_points_confidence']=npcpu(pred['world_points_conf'][0])
                h,w=depth.shape[-2:];A=[]
                for sw,sh in sizes:
                    pre_h=round(sh*(518/sw)/14)*14;top=max(0,(pre_h-518)//2)
                    A.append(resize_map((sw,sh),(518,pre_h),top))
                extra['preprocess']='official crop: width518, rounded14 height, centre crop only if height>518'
                del pred,images,e,k
            elif args.model=='pi3x':
                images=pi3_images(paths).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):pred=model(imgs=images[None])
                local=npcpu(pred['local_points'][0]);depth=local[...,2];K,fit=estimate_pinhole(local)
                conf=npcpu(torch.sigmoid(pred['conf'][0,...,0]));w2c=np.linalg.inv(npcpu(pred['camera_poses'][0])).astype(np.float64)
                direct_world=npcpu(pred['points'][0]);extra.update({'metric_multiplier_already_applied_by_model':float(pred['metric'][0]),'pinhole_fit_residual_px':fit,
                                                                 'preprocess':'official255000pixel budget nearest14 grid; direct resize, no crop',
                                                                 'intrinsics_kind':'least-squares pinhole diagnostic from model local rays, not supplied intrinsics'})
                h,w=depth.shape[-2:];A=[resize_map(s,(w,h)) for s in sizes]
                del pred,images
            else:
                with torch.inference_mode():pred=model.inference(paths,process_res=504,process_res_method='upper_bound_resize',infer_gs=False,export_dir=None)
                depth=pred.depth.astype(np.float32);conf=pred.conf.astype(np.float32) if pred.conf is not None else np.ones_like(depth)
                if pred.sky is not None:aux_arrays['predicted_sky']=np.asarray(pred.sky)
                K=np.asarray(pred.intrinsics,dtype=np.float64);w2c=as_w2c(pred.extrinsics);local=unproject(depth,K).astype(np.float32)
                h,w=depth.shape[-2:];A=[resize_map(s,(w,h)) for s in sizes];direct_world=None
                extra.update({'model_is_metric':int(pred.is_metric),'model_scale_factor':float(pred.scale_factor) if pred.scale_factor is not None else None,
                              'preprocess':'official upper_bound_resize504 nearest14; no crop; no extrinsic/intrinsic inputs'})
                del pred
            torch.cuda.synchronize();inference_seconds=time.monotonic()-t
            A=np.asarray(A);qa=validate(depth,K,w2c)
            assert np.isfinite(local).mean()>.95
            K_staged,K_native,observations=diagnostics(case,frames,depth,K,local,w2c,A,extra,variant)
            arrays={'depth':depth.astype(np.float32),'confidence':conf.astype(np.float32),'local_camera_points':local.astype(np.float32),
                    'K_model':K,'K_staged_rgb':K_staged,'K_native_rgb':K_native,'w2c':w2c,'rgb_to_model':A,
                    'frame_local_indices':np.array([x['local_index'] for x in frames]),'native_frame_ids':np.array([x['native_frame_id'] for x in frames]),
                    'time_seconds':np.array([x['time_seconds'] for x in frames])}
            if direct_world is not None:arrays['direct_world_points']=direct_world
            arrays.update(aux_arrays)
            output=case_dir/'GEOMETRY.npz';np.savez_compressed(output,**arrays)
            result={'status':'PASS','episode':case['episode'],'dataset':case['dataset'],'sequence':case['sequence'],'variant':variant,
                    'frames':frames,'native_to_staged_crop_xyxy':case['native_to_staged_crop_xyxy'],
                    'metric_status':spec['metric_status'],'camera_convention':'OpenCV +Xright +Ydown +Zforward, w2c 4x4',
                    'pixel_convention':'official model grids integer0..W-1; K scaled with official continuous-coordinate resize, PIL sampling centres differ',
                    'model_hw':[h,w],'inference_and_postprocess_seconds':inference_seconds,
                    'total_seconds_before_save_receipt':time.monotonic()-t,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host_process_peak_RSS_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'qa':qa,'extra':extra,'observations':observations,'output_sha256':{str(output):sha(output)},'input_sha256':pins}
            verify(pins);verify(freeze['sha256']);dump(case_dir/'RESULT.json',result)
            del arrays,aux_arrays,depth,local,conf,direct_world
            print('PASS',args.model,label,'seconds',round(time.monotonic()-t,2),flush=True)
        except Exception as exc:
            result={'status':'FAIL','episode':case['episode'],'variant':variant,'error':repr(exc),'traceback':traceback.format_exc()}
            dump(case_dir/'FAILED.json',result);print('FAIL',args.model,label,repr(exc),flush=True)
            # Capacity failure is a model-level smoke failure; do not change frame count or silently retune.
            if isinstance(exc,torch.cuda.OutOfMemoryError):
                receipt['capacity_abort_after']=label;receipt['records'].append(result);break
        receipt['records'].append(result);torch.cuda.empty_cache()
    verify(freeze['sha256']);verify(weight_pins)
    receipt['total_seconds']=time.monotonic()-start;receipt['expected_windows']=len(todo)
    receipt['all_expected_pass']=len(receipt['records'])==len(todo) and all(x['status']=='PASS' for x in receipt['records'])
    dump(out/'COMPLETE.json',receipt)
    if not receipt['all_expected_pass']:raise SystemExit(2)

if __name__=='__main__':main()
