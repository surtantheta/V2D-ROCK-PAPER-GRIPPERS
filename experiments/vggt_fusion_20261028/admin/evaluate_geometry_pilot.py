#!/usr/bin/env python3
"""CPU diagnostic: predicted-human calibration, then separate partial-surface GT scoring.

Not an object reconstruction/host CD-O evaluator. No GT participates in fitting.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

P=Path('/home/tanuj/V2D_challenge');R=P/'experiments/vggt_fusion_20261028'
W=P/'work_labeled';BR=P/'experiments/offline_opt_20261020'
KIT=P/'data/assets/submission_kit/v2d_submission_kit'
INPUTS=R/'PILOT_INPUTS.json';FREEZE=R/'SOURCE_FREEZE.json';OWN=R/'admin/DEPTH_DIAGNOSTIC_INPUTS.json'
OUT=R/'depth_diagnostic_v1';REQ=BR/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
REFS={'BEHAVE':BR/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json',
      'InterCap':BR/'benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json'}
PROTOCOL=dict(schema='geometry-pilot-partial-surface-diagnostic-v1',cases=list(range(900,912)),
    models=['vggt1b','da3nested11','pi3x'],frames_per_case=16,fit_frame_positions=list(range(8)),heldout_frame_positions=list(range(8,16)),
    human_fit='one proper positive weighted Sim3 per model/clip; predicted MAIN MHR only; no ICP/object/GT fit',
    body_visibility='approximate vertex Z-buffer, closest official22-joint torso region; filled/closed vertex footprint eroded3modelpixels; object mask dilated3 excluded',
    torso_joint_positions=[0,1,2,3,6,9,12,13,14],body_max_pixels_per_frame=512,
    minimum_body_pixels_per_frame=64,minimum_fit_frames=6,minimum_fit_points=512,
    confidence_policy='finite positive confidence at least within-frame median among candidate points; ties retained; normalized equal frame fit weights',
    human_Huber_delta_m=.03,IRLS_iterations=5,minimum_covariance_ratio=1e-5,
    maximum_fit_body_mean_cm=10.,maximum_heldout_body_mean_cm=15.,minimum_heldout_sufficient_frames=6,
    object_sampling='selected mask with score>=.5; erosion3, else1, else0 explicitly flagged; minimum16 valid pixels; deterministic evenly spaced max1024',
    object_minimum_pixels=16,object_max_pixels=1024,surface_samples=4096,surface_seed=314159,
    coordinates='Use authoritative local_camera_points. MAIN K=S6K scaled by TRACK.model_wh to actual RGB, then exact rgb_to_model; integer-grid nearest pixel, no +.5 shift.',
    Pi3_metric='already applied once; K is diagnostic only, never replace local points by K-unprojected depth',
    score='ONE-SIDED foreground points to deterministic4096GT surface samples; separate raw-camera and fixed MAIN GT-human-surrogate alignment',
    VGGT_raw_metric_score=False,fit_data_frozen_before_native_GT_values=True,challenge_GT=False,reserve=False,
    existing_development_results_and_GT_derived_sidecars_available=True,analyst_blinded_to_previous_GT_results=False,
    new_partial_surface_GT_scores_computed_only_after_fit_exports_frozen=True,
    limitations=['Partial visible foreground is not full object CD-O; sparse surface sampling has a distance floor.',
      'Predicted MHR pose/projection, masks, depth and approximate visibility can be wrong.',
      'A Sim3 cannot generally undo different focal distortions; held-out body errors must be reported.',
      'Unsupported frames/cases remain explicit; do not rank models on differing supported subsets.'])

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def dump(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def verify(pins):
    for p,h in pins.items():assert sha(p)==h,('source changed',p)
def sample_indices(n,count):return np.unique(np.rint(np.linspace(0,n-1,min(n,count))).astype(int))
def apply(x,s,r,t):return s*(np.asarray(x)@np.asarray(r).T)+np.asarray(t)
def stats(x):
    a=np.asarray(x,float)
    return None if not a.size else dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def fit_sim3(x,y,w):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    assert x.shape==y.shape and x.ndim==2 and x.shape[1]==3 and w.shape==(len(x),)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all()
    w=w/w.sum();xm=w@x;ym=w@y;xc=x-xm;yc=y-ym
    u,d,vt=np.linalg.svd((xc*w[:,None]).T@yc);sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(vt.T@u.T))
    rot=vt.T@np.diag(sign)@u.T;scale=float(d@sign/np.sum(w[:,None]*xc*xc));trans=ym-scale*(xm@rot.T)
    assert scale>0 and np.isfinite(scale) and np.linalg.det(rot)>0
    return scale,rot,trans

def robust_fit(x,y,w):
    robust=np.ones(len(x))
    for _ in range(PROTOCOL['IRLS_iterations']):
        s,r,t=fit_sim3(x,y,w*robust);err=np.linalg.norm(apply(x,s,r,t)-y,axis=1)
        robust=np.minimum(1.,PROTOCOL['human_Huber_delta_m']/np.maximum(err,1e-12))
    return fit_sim3(x,y,w*robust)

def warp_mask(path,A,hw):
    import cv2
    from PIL import Image
    h,w=hw;mask=np.asarray(Image.open(path).convert('L'))>127
    return cv2.warpPerspective(mask.astype(np.uint8),A,(w,h),flags=cv2.INTER_NEAREST)>0

def main_camera(case,pins):
    name=f"episode_{case['episode']:06d}";pose=W/f's6/{name}/poses.npz';track=pose.parent/'TRACK.json';body=W/f's3db/{name}/sam3d_body.npz'
    for p in [pose,track,body]:pins[str(p)]=sha(p)
    K=np.load(pose,allow_pickle=False)['K_model'].astype(float)
    wh=json.loads(track.read_text())['model_wh'];rgb=case['frames'][0]['rgb_wh']
    K_rgb=np.diag([rgb[0]/wh[0],rgb[1]/wh[1],1.])@K
    kb=np.load(body,allow_pickle=False)['K'].astype(float)
    assert np.allclose(K_rgb,kb,atol=1e-3,rtol=0),('MAIN K/source RGB inconsistency',name)
    crop=case['native_to_staged_crop_xyxy']
    if crop is not None:
        assert crop==[240,0,1680,1080] and rgb==[1440,1080] and case['native_wh']==[1920,1080]
        K_native=np.array([[1,0,240],[0,1,0],[0,0,1.]])@K_rgb
    else:K_native=K_rgb.copy()
    return K_rgb,dict(S6_K=K.tolist(),S6_model_wh=wh,RGB_wh=rgb,K_RGB=K_rgb.tolist(),K_native=K_native.tolist(),S3db_K_max_difference_px=float(abs(K_rgb-kb).max()))

def body_pairs(vertices,joints,local,confidence,A,Krgb,objmask):
    """Approximate visible predicted-MHR vertex/pixel correspondence; no GT inputs."""
    h,w=local.shape[:2];v=np.asarray(vertices,float);km=A@Krgb
    uvh=v@km.T;uv=uvh[:,:2]/uvh[:,2,None];pix=np.rint(uv).astype(int)
    inside=np.isfinite(uv).all(1)&(v[:,2]>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    indices=np.flatnonzero(inside);xy=pix[indices];flat=xy[:,1]*w+xy[:,0]
    order=np.argsort(v[indices,2],kind='stable');_,unique=np.unique(flat[order],return_index=True)
    front=indices[order[unique]];foot=np.zeros((h,w),bool);foot[pix[front,1],pix[front,0]]=True
    foot=ndimage.binary_fill_holes(ndimage.binary_closing(foot,iterations=2))
    interior=ndimage.binary_erosion(foot,iterations=3)
    forbidden=ndimage.binary_dilation(objmask,iterations=3)
    # Region names are positions in the official22-joint list, not arbitrary MHR vertex IDs.
    nearest=cKDTree(joints).query(v,workers=1)[1]
    torso=np.isin(nearest,PROTOCOL['torso_joint_positions']);chosen=front[torso[front]]
    yy,xx=pix[chosen,1],pix[chosen,0]
    good=interior[yy,xx]&~forbidden[yy,xx]&np.isfinite(local[yy,xx]).all(1)&(local[yy,xx,2]>0)&np.isfinite(confidence[yy,xx])&(confidence[yy,xx]>0)
    chosen=chosen[good];yy,xx=pix[chosen,1],pix[chosen,0]
    if len(chosen):
        keep=confidence[yy,xx]>=np.median(confidence[yy,xx]);chosen=chosen[keep]
    chosen=chosen[sample_indices(len(chosen),PROTOCOL['body_max_pixels_per_frame'])] if len(chosen) else chosen
    yy,xx=pix[chosen,1],pix[chosen,0]
    return local[yy,xx].astype(float),v[chosen],np.stack([xx,yy],1),dict(body_pixels=len(chosen),visibility='approximate vertex Z-buffer, not triangle raster/observed human segmentation')

def foreground(local,confidence,mask,selected,score,sky=None):
    if not selected or score<.5:return np.empty((0,3)),dict(status='unsupported_mask_selection',count=0)
    for erosion in [3,1,0]:
        region=ndimage.binary_erosion(mask,iterations=erosion) if erosion else mask.copy()
        good=region&np.isfinite(local).all(2)&(local[:,:,2]>0)&np.isfinite(confidence)&(confidence>0)
        if sky is not None:good &= ~np.asarray(sky,bool)
        if good.any():good &= confidence>=np.median(confidence[good])
        flat=np.flatnonzero(good)
        if len(flat)>=PROTOCOL['object_minimum_pixels']:
            chosen=flat[sample_indices(len(flat),PROTOCOL['object_max_pixels'])]
            return local.reshape(-1,3)[chosen].astype(float),dict(status='supported',count=len(chosen),valid_pixels=len(flat),erosion=erosion,erosion_fallback=erosion!=3)
    return np.empty((0,3)),dict(status='unsupported_foreground_pixels',count=0,erosion_fallback=True)

def freeze():
    assert not OWN.exists(),'preserve previous frozen policy'
    j=json.loads(INPUTS.read_text());assert len(j['cases'])==15 and {c['episode'] for c in j['cases'] if c['dataset']!='challenge'}==set(range(900,912))
    pins={str(p):sha(p) for p in [INPUTS,FREEZE,Path(__file__).resolve(),R/'admin/test_geometry_pilot_eval.py',R/'run_depth_diagnostic.pbs',
        P/'models/mhr/mhr_model.pt',KIT/'v2dlb/mhr_metrics.py',BR/'code/labeled_object_scorer.py',REQ,*REFS.values(),
        W/'LABELED_EPISODES.json',W/'align_BEHAVE/TRANSFORMS.json',W/'align_InterCap/TRANSFORMS.json']}
    verify(j['input_sha256']);pins.update(j['input_sha256']);pins.update(json.loads(FREEZE.read_text())['sha256'])
    for c in j['cases']:
        if c['dataset']!='challenge':main_camera(c,pins)
    assert sha(FREEZE).startswith('099e739'),'review a new pilot source freeze explicitly'
    dump(OWN,dict(protocol=PROTOCOL,sha256=pins,GT_values_used_for_fit=False,pilot_source_freeze_sha256=sha(FREEZE)))
    print('FROZEN',len(pins),'sources; no native GT values fitted',flush=True)

def decode_main(model,data,ids,torch):
    vertices=[];joints=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];pose=np.concatenate([data['pose'][ix],np.repeat(data['scales'][None],len(ix),0)],1).astype(np.float32)
        shape=np.repeat(data['shape'].astype(np.float32).reshape(1,45),len(ix),0)
        with torch.inference_mode():v,j=model(torch.from_numpy(shape),torch.from_numpy(pose),torch.zeros(len(ix),72),True)
        vertices.append(v.double().numpy()/100*np.array([1.,-1.,-1.]));joints.append(j[...,:3].double().numpy()/100*np.array([1.,-1.,-1.]))
    return np.concatenate(vertices),np.concatenate(joints)

def predicted_stage(frozen,j):
    import torch
    sys.path.insert(0,str(KIT));from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    model=torch.jit.load(str(P/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
    records=[];pins=dict(frozen['sha256']);outputs={};metadata=json.loads((W/'LABELED_EPISODES.json').read_text());checked_models=set()
    for case in j['cases']:
        if case['dataset']=='challenge':continue
        ep=case['episode'];frames=case['frames'];ids=np.array([f['local_index'] for f in frames]);info=metadata[str(ep)]
        assert len(ids)==16 and [info['frame_ids'][i] for i in ids]==[f['native_frame_id'] for f in frames]
        assert np.allclose([info['times'][i] for i in ids],[f['time_seconds'] for f in frames],atol=1e-8,rtol=0)
        data=dict(np.load(case['fixed_MAIN_human_npz'],allow_pickle=False));verts,joints=decode_main(model,data,ids,torch)
        assert verts.shape==(16,18439,3) and np.isfinite(verts).all()
        joints=joints[:,np.array(MHR_TABLE3_BODY_JOINT_INDICES)];Krgb,camera=main_camera(case,pins)
        for spec in j['models']:
            name=spec['name'];base=Path(j['large_output_root'])/name/f'episode_{ep:06d}_normal';receipt=base/'RESULT.json';gp=base/'GEOMETRY.npz'
            rec=dict(episode=ep,dataset=case['dataset'],sequence=case['sequence'],model=name,metric_status=spec['metric_status'],
                     frames=[],calibration=None,calibration_metric_diagnostic_eligible=False,MAIN_camera=camera,fit_uses_only_predicted_human=True)
            if not receipt.exists() or not gp.exists():
                rec.update(status='model_window_missing',expected_geometry=str(gp),frames=[dict(local_index=f['local_index'],native_frame_id=f['native_frame_id'],time_seconds=f['time_seconds'],body={'body_pixels':0},foreground={'status':'model_window_missing','count':0}) for f in frames]);records.append(rec);continue
            rr=json.loads(receipt.read_text());assert rr['status']=='PASS' and rr['episode']==ep and rr['variant']=='normal'
            verify(rr['input_sha256']);verify(rr['output_sha256']);pins.update(rr['input_sha256']);pins.update(rr['output_sha256']);pins[str(receipt)]=sha(receipt)
            complete=base.parent/'COMPLETE.json';assert complete.exists();cc=json.loads(complete.read_text())
            assert cc['source_freeze_sha256']==frozen['pilot_source_freeze_sha256']
            if name not in checked_models:
                verify(cc['weight_sha256']);pins.update(cc['weight_sha256']);checked_models.add(name)
            pins[str(complete)]=sha(complete)
            with np.load(gp,allow_pickle=False) as g:
                local=g['local_camera_points'];confidence=g['confidence'];A=g['rgb_to_model'];depth=g['depth'];sky=g['predicted_sky'] if 'predicted_sky' in g else None
                assert local.shape[:3]==confidence.shape==depth.shape and local.shape[0]==16 and local.shape[-1]==3
                assert np.array_equal(g['frame_local_indices'],ids) and np.array_equal(g['native_frame_ids'],[f['native_frame_id'] for f in frames])
                assert np.allclose(g['time_seconds'],[f['time_seconds'] for f in frames],atol=1e-8,rtol=0)
                assert np.allclose(g['K_staged_rgb'],np.linalg.inv(A)@g['K_model'],atol=1e-5,rtol=1e-6)
                if case['native_to_staged_crop_xyxy'] is not None:assert np.allclose(g['K_native_rgb'][:,0,2]-g['K_staged_rgb'][:,0,2],240.)
                rec['model_camera_diagnostics']=dict(model_hw=list(depth.shape[1:]),K_staged_rgb=g['K_staged_rgb'].tolist(),
                    focal_ratio_to_MAIN_median=np.median(g['K_staged_rgb'][:,[0,1],[0,1]]/Krgb[[0,1],[0,1]],axis=0).tolist(),
                    principal_point_difference_to_MAIN_median=np.median(g['K_staged_rgb'][:,:2,2]-Krgb[:2,2],axis=0).tolist(),
                    model_extra=rr.get('extra'),pixel_convention=rr.get('pixel_convention'))
                paired=[];clouds=[]
                for i,f in enumerate(frames):
                    mask=warp_mask(f['mask'],A[i],depth.shape[1:]);x,y,pixel,qa=body_pairs(verts[i],joints[i],local[i],confidence[i],A[i],Krgb,mask)
                    paired.append((x,y));cloud,fg=foreground(local[i],confidence[i],mask,f['mask_selected'],f['mask_score'],None if sky is None else sky[i]);clouds.append(cloud)
                    rec['frames'].append(dict(local_index=f['local_index'],native_frame_id=f['native_frame_id'],time_seconds=f['time_seconds'],body=qa,foreground=fg,
                        raw_object_to_predicted_body_median_Z_ratio=None if not len(cloud) or not len(x) else float(np.median(cloud[:,2])/np.median(x[:,2]))))
                eligible=[i for i in range(8) if len(paired[i][0])>=PROTOCOL['minimum_body_pixels_per_frame']]
                if len(eligible)>=PROTOCOL['minimum_fit_frames'] and sum(len(paired[i][0]) for i in eligible)>=PROTOCOL['minimum_fit_points']:
                    x=np.concatenate([paired[i][0] for i in eligible]);y=np.concatenate([paired[i][1] for i in eligible]);weights=np.concatenate([np.full(len(paired[i][0]),1/len(paired[i][0])) for i in eligible])
                    ratiox=float(np.linalg.svd(x-x.mean(0),compute_uv=False)[-1]/np.linalg.svd(x-x.mean(0),compute_uv=False)[0]);ratioy=float(np.linalg.svd(y-y.mean(0),compute_uv=False)[-1]/np.linalg.svd(y-y.mean(0),compute_uv=False)[0])
                    if min(ratiox,ratioy)>=PROTOCOL['minimum_covariance_ratio']:
                        s,r,t=robust_fit(x,y,weights);rec['calibration']=dict(scale=s,R=r.tolist(),t=t.tolist(),mapping='X_MAIN=s*(X_model_local@R.T)+t',fit_frame_positions=eligible,source_span_ratio=ratiox,target_span_ratio=ratioy)
                for i,(x,y) in enumerate(paired):
                    rec['frames'][i]['raw_body_Z_ratio_MAIN_over_model']=None if not len(x) else float(np.median(y[:,2]/x[:,2]))
                    if rec['calibration'] is not None and len(x):
                        tr=rec['calibration'];rec['frames'][i]['calibrated_body_residual_cm']=stats(100*np.linalg.norm(apply(x,tr['scale'],tr['R'],tr['t'])-y,axis=1))
                for key,group in [('fit',range(8)),('heldout',range(8,16))]:
                    vals=[rec['frames'][i]['calibrated_body_residual_cm']['mean'] for i in group if 'calibrated_body_residual_cm' in rec['frames'][i]]
                    rec[key+'_mean_frame_body_residual_cm']=None if not vals else float(np.mean(vals));rec[key+'_supported_frames']=len(vals)
                    rec[key+'_sufficient_support_frames']=sum(len(paired[i][0])>=PROTOCOL['minimum_body_pixels_per_frame'] for i in group)
                rec['calibration_metric_diagnostic_eligible']=bool(rec['calibration'] is not None and
                    rec['heldout_sufficient_support_frames']>=PROTOCOL['minimum_heldout_sufficient_frames'] and
                    rec['fit_mean_frame_body_residual_cm'] is not None and rec['fit_mean_frame_body_residual_cm']<=PROTOCOL['maximum_fit_body_mean_cm'] and
                    rec['heldout_mean_frame_body_residual_cm'] is not None and rec['heldout_mean_frame_body_residual_cm']<=PROTOCOL['maximum_heldout_body_mean_cm'])
                rec['status']='calibrated_diagnostic' if rec['calibration'] is not None else 'unsupported_human_calibration'
                if rec['calibration'] is not None and not rec['calibration_metric_diagnostic_eligible']:rec['status']='human_calibration_failed_predicted_only_reliability_gate'
                cp=OUT/f'points_{name}_{ep:06d}.npz';np.savez_compressed(cp,**{f'frame_{i:02d}':p for i,p in enumerate(clouds)})
                outputs[str(cp)]=sha(cp);rec['foreground_cache']=str(cp);rec['foreground_cache_sha256']=outputs[str(cp)]
            records.append(rec);print('CALIBRATED',name,ep,rec['status'],'heldoutcm',rec['heldout_mean_frame_body_residual_cm'],flush=True)
    controls=control_diagnostics(j,pins)
    verify(pins);assert len(records)==36 and all(len(x['frames'])==16 for x in records)
    freeze_path=OUT/'CALIBRATION_AND_CLOUDS_FROZEN.json';dump(freeze_path,dict(protocol=PROTOCOL,records=records,controls=controls,source_sha256=pins,output_sha256=outputs,
        native_GT_values_read=False,GT_human_transform_values_read=False,GT_object_fit=False))
    return freeze_path

def control_diagnostics(j,pins):
    reports=[]
    for spec in j['models']:
        for case in j['cases']:
            if case['dataset']!='challenge':continue
            normal=None
            for variant in ['normal','duplicate_first','reverse']:
                base=Path(j['large_output_root'])/spec['name']/f"episode_{case['episode']:06d}_{variant}"
                rp,gp=base/'RESULT.json',base/'GEOMETRY.npz';rec=dict(model=spec['name'],episode=case['episode'],variant=variant)
                if not rp.exists() or not gp.exists():rec['status']='missing';reports.append(rec);continue
                result=json.loads(rp.read_text());verify(result['output_sha256']);pins.update(result['output_sha256']);pins[str(rp)]=sha(rp)
                with np.load(gp,allow_pickle=False) as g:
                    d=g['depth'].reshape(16,-1);ix=sample_indices(d.shape[1],4096);d=d[:,ix].astype(float)
                    positive=np.isfinite(d)&(d>0);rec.update(status='supported',positive_finite_fraction=positive.mean(1).tolist(),camera_motion=result.get('extra',{}).get('predicted_camera_motion'),K_focal_frame_std_px=np.std(g['K_staged_rgb'][:,[0,1],[0,1]],axis=0).tolist())
                    if variant=='normal':normal=d.copy()
                    elif variant=='duplicate_first':
                        valid=positive&positive[0];relative=np.where(valid,abs(d-d[0])/np.maximum(abs(d[0]),1e-12),np.nan)
                        rec['duplicate_relative_depth_abs_median']=None if not valid.any() else float(np.nanmedian(relative))
                    elif normal is not None:
                        target=normal[::-1];valid=positive&np.isfinite(target)&(target>0)
                        ratio=float(np.median(target[valid]/d[valid])) if valid.any() else None
                        rec['reverse_shared_Z_ratio_to_normal']=ratio
                        rec['reverse_ratio_aligned_relative_Z_error']=None if ratio is None else float(np.median(abs(ratio*d[valid]-target[valid])/target[valid]))
                reports.append(rec)
    return reports

def truth_stage(freeze_path,j):
    """Native GT and MAIN GT-human transforms are first decoded in this last stage."""
    frozen=json.loads(freeze_path.read_text());verify(frozen['source_sha256']);verify(frozen['output_sha256'])
    sys.path.insert(0,str(BR/'code'));import labeled_object_scorer as sc
    request=json.loads(REQ.read_text());refs={k:json.loads(v.read_text()) for k,v in REFS.items()}
    human={ds:{x['sequence']:x['primary_transform'] for x in json.loads((W/f'align_{ds}/TRANSFORMS.json').read_text())['records']} for ds in REFS}
    # Restrict InterCap annotation/image decode to exactly the pilot's16 native frames per take.
    pilot={c['sequence']:c for c in j['cases'] if c['dataset']=='InterCap'};rq=copy.deepcopy(request);icrefs=copy.deepcopy(refs['InterCap'])
    for source in rq['InterCap']:
        take=source['take'];old=list(source['native_frame_numbers']);wanted=[x['native_frame_id'] for x in pilot[take]['frames']];ix=[old.index(v) for v in wanted]
        for k,v in list(source.items()):
            if isinstance(v,list) and len(v)==len(old):source[k]=[v[i] for i in ix]
        ref=next(v for v in icrefs['development_samples'] if v['take']==take)
        for k,v in list(ref.items()):
            if isinstance(v,list) and len(v)==len(old):ref[k]=[v[i] for i in ix]
    normalized,ictruth,extra=sc.intercap_references(rq,icrefs,30.)
    gt_sources=dict(extra);gt_data={};behave_input={x['sequence']:x for x in request['BEHAVE']}
    for case in j['cases']:
        if case['dataset']=='challenge':continue
        seq=case['sequence'];dataset=case['dataset'];frames=[x['native_frame_id'] for x in case['frames']]
        if dataset=='BEHAVE':
            ref=next(x for x in refs['BEHAVE'] if x['sequence']==seq)
            for k,h in [('gt_npz','gt_sha256'),('calibration_json','calibration_sha256'),('object_mesh_original','object_mesh_original_sha256'),('object_mesh_simplified','object_mesh_simplified_sha256')]:
                assert sha(ref[k])==ref[h];gt_sources[ref[k]]=ref[h]
            gt=np.load(ref['gt_npz'],allow_pickle=False);assert int(gt['kid'])==1
            simple,_=sc.load_surface(ref['object_mesh_simplified']);offset=simple.mean(0)
            assert np.allclose(offset,ref['mesh_origin_subtracted_mean'],atol=1e-9,rtol=0)
            v,f=sc.load_surface(ref['object_mesh_original'],offset);positions=[behave_input[seq]['video_frame_indices'].index(i) for i in frames]
            rows=[ref['GT_row_indices'][i] for i in positions]
            assert all(str(gt['frames'][row])==ref['GT_frame_names'][i] for row,i in zip(rows,positions))
            poses=gt['pose_abs'][rows].astype(float)
        else:v,f=sc.load_surface(ictruth[seq]['mesh']);poses=ictruth[seq]['poses'].astype(float)
        assert poses.shape==(16,4,4) and np.isfinite(poses).all()
        for pose in poses:sc.validate_rotation(pose[:3,:3]);assert np.allclose(pose[3],[0,0,0,1])
        sampled,_=sc.surface_samples(v,f,4096,314159)
        gt_data[seq]=[sampled@pose[:3,:3].T+pose[:3,3] for pose in poses]
    records=[]
    for record in frozen['records']:
        rec=copy.deepcopy(record);seq=rec['sequence'];dataset=rec['dataset'];h=human[dataset][seq]
        if 'foreground_cache' not in rec:records.append(rec);continue
        arrays=np.load(rec['foreground_cache'],allow_pickle=False)
        for i,row in enumerate(rec['frames']):
            x=arrays[f'frame_{i:02d}'];row['one_sided_surface_diagnostic_cm']={}
            if not len(x):continue
            candidates={}
            if rec['model']!='vggt1b':candidates['raw_model_metric_prior']=x
            if rec['calibration'] is not None and rec['calibration_metric_diagnostic_eligible']:
                tr=rec['calibration'];candidates['predicted_body_calibrated']=apply(x,tr['scale'],tr['R'],tr['t'])
            tree=cKDTree(gt_data[seq][i])
            for label,cloud in candidates.items():
                for gauge,points in [('raw_camera',cloud),('fixed_MAIN_GT_human_surrogate',apply(cloud,h['scale'],h['R'],h['t']))]:
                    row['one_sided_surface_diagnostic_cm'][label+'/'+gauge]=stats(100*tree.query(points,workers=1)[0])
        rec['exact_hosted_CD_O']=False;rec['visible_foreground_only']=True;records.append(rec)
    verify(frozen['source_sha256']);verify(frozen['output_sha256']);verify(gt_sources)
    summaries=[]
    for model in PROTOCOL['models']:
        for dataset in REFS:
            cohort=[r for r in records if r['model']==model and r['dataset']==dataset]
            keys=sorted({k for r in cohort for f in r['frames'] for k in f.get('one_sided_surface_diagnostic_cm',{})})
            for key in keys:
                means=[];counts=[];split={}
                for r in cohort:
                    values=[f.get('one_sided_surface_diagnostic_cm',{}).get(key) for f in r['frames']]
                    supported=[v['mean'] for v in values if v is not None];counts.append(len(supported))
                    if supported:means.append(float(np.mean(supported)))
                for label,ix in [('fit_frames',range(8)),('heldout_frames',range(8,16))]:
                    values=[r['frames'][i].get('one_sided_surface_diagnostic_cm',{}).get(key) for r in cohort for i in ix]
                    supported=[v['mean'] for v in values if v is not None];split[label]=dict(supported_frames=len(supported),mean_frame_cm=None if not supported else float(np.mean(supported)))
                summaries.append(dict(model=model,dataset=dataset,diagnostic=key,expected_cases=len(cohort),supported_cases=len(means),
                    supported_frame_counts_per_case=counts,complete_frame_coverage=all(n==16 for n in counts),
                    mean_supported_sequence_cm=None if not means else float(np.mean(means)),partial_visibility_not_CD_O=True,split=split))
    dump(OUT/'COMPLETE.json',dict(status='COMPLETE_PARTIAL_SURFACE_DIAGNOSTIC',protocol=PROTOCOL,records=records,controls=frozen['controls'],summaries=summaries,
        expected_model_cases=36,model_cases_retained=len(records),expected_frames_per_model=192,
        fitted_model_cases=sum(x['calibration'] is not None for x in records),calibrated_metric_eligible_cases=sum(x['calibration_metric_diagnostic_eligible'] for x in records),
        calibration_frozen_before_GT=True,calibration_receipt=str(freeze_path),calibration_receipt_sha256=sha(freeze_path),
        GT_source_sha256=gt_sources,source_sha256=frozen['source_sha256'],exact_hosted_CD_O=False,reserve_used=False,
        native_InterCap_GT_annotations_decoded=128,challenge_GT_used=False,no_model_or_trajectory_export=True))

def run():
    assert os.environ.get('PBS_JOBID'),'CPU decode/scoring requires reserved scheduler job'
    assert not OUT.exists(),'preserve prior outputs'
    frozen=json.loads(OWN.read_text());verify(frozen['sha256']);assert frozen['protocol']==PROTOCOL
    j=json.loads(INPUTS.read_text());OUT.mkdir();dump(OUT/'PROTOCOL_BEFORE_FIT_AND_GT.json',PROTOCOL)
    p=predicted_stage(frozen,j);truth_stage(p,j)

def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    freeze() if a.freeze else run()

if __name__=='__main__':main()
