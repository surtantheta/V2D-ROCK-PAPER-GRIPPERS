#!/usr/bin/env python3
"""SCORER-ONLY GT decomposition. Never produces production initializers."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020');sys.path.insert(0,str(ROOT/'code'));import labeled_object_scorer as scorer
OUT=ROOT/'validation/intercap_failure_decomposition_v1';OUT.mkdir(exist_ok=False)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def surface_info(path):
 v,f=scorer.load_surface(path);q=v[f];a=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2;keep=a>1e-14;q,a=q[keep],a[keep];mass=a.sum();mu=(q.mean(1)*a[:,None]).sum(0)/mass;sumq=q.sum(1);moment=(np.einsum('nki,nkj->nij',q,q)+np.einsum('ni,nj->nij',sumq,sumq))/12;cov=(moment*a[:,None,None]).sum(0)/mass-np.outer(mu,mu);cov=(cov+cov.T)/2;eig,axes=np.linalg.eigh(cov);used=np.unique(f[keep]);ext=np.ptp((v[used]-mu)@axes[:,::-1],axis=0);radius=np.sqrt(np.trace(cov));points,_=scorer.surface_samples(v,f,4096,314159);return dict(vertices=v,faces=f,center=mu,rms_radius=float(radius),pca_extents=sorted(ext.tolist(),reverse=True),points=points)
def stats(a):
 a=np.asarray(a);return dict(first=float(a[0]),last=float(a[-1]),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max()),min=float(a.min()))

predpath=ROOT/'intercap_results_v1/human_aligned/foundationpose_PREDICTIONS.json';scorepath=ROOT/'intercap_results_v1/human_aligned/foundationpose_SCORES.json';requestpath=ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json';refpath=ROOT/'benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json';alignpath=ROOT/'intercap_human_alignment_sidecars_v1/TRANSFORMS.json';runtimepath=ROOT/'labeled_intercap_v1/FP_RUNTIME.json';humanpath=ROOT/'labeled_intercap_v1/human_scale/RESULT.json'
req=json.load(open(requestpath));refs=json.load(open(refpath));norm,truth,sources=scorer.intercap_references(req,refs,30.);pred=json.load(open(predpath));scores={s['sequence']:s for s in json.load(open(scorepath))['per_sequence']};align={s['sequence']:s for s in json.load(open(alignpath))['records']};runtime={s['take']:s for s in json.load(open(runtimepath))['records']};humans={s['sequence']:s for s in json.load(open(humanpath))['records']};firstnative=ROOT/'intercap_native_contract_v1/FIRST_FRAME_NATIVE_HUMANS.json';cameras={r['take']:r['camera'] for r in json.load(open(firstnative))['records']}
for p in [Path(__file__),predpath,scorepath,requestpath,refpath,alignpath,runtimepath,humanpath,firstnative]:sources[str(p)]=sha(p)
records=[]
for sequence in pred['sequences']:
 name=sequence['sequence'];item=runtime[name];disk=item['name'];g=surface_info(truth[name]['mesh']);p=surface_info(sequence['mesh_path']);scale=sequence['object_scale'];sources[sequence['mesh_path']]=sha(sequence['mesh_path']);gt=truth[name]['poses'];n=len(sequence['frames']);Rs=np.array([f['R_obj2cam'] for f in sequence['frames']]);ts=np.array([f['t_obj2cam_m'] for f in sequence['frames']]);pc=scale*p['center']@Rs.transpose(0,2,1)+ts;gc=g['center']@gt[:,:3,:3].transpose(0,2,1)+gt[:,:3,3];delta=pc-gc;radiusratio=scale*p['rms_radius']/g['rms_radius'];cd=[];centeroracle=[];radiusoracle=[]
 for i in range(n):
  pp=scale*p['points']@Rs[i].T+ts[i];gg=g['points']@gt[i,:3,:3].T+gt[i,:3,3];cd.append(scorer.surface_cd_sum_cm(pp,gg));centered=pp-pc[i]+gc[i];centeroracle.append(scorer.surface_cd_sum_cm(centered,gg));resized=(pp-pc[i])/radiusratio+gc[i];radiusoracle.append(scorer.surface_cd_sum_cm(resized,gg))
 assert np.max(abs(np.array(cd)-scores[name]['surface_cd_per_frame_cm']))<1e-10
 trackpath=ROOT/'labeled_intercap_v1/tracks'/disk/'TRACK.json';track=json.load(open(trackpath));posepath=ROOT/'labeled_intercap_v1/tracks'/disk/'poses.npz';raw=np.load(posepath,allow_pickle=False);maskpath=Path(item['masks'])/'MASKS.json';masks=json.load(open(maskpath))['rows'];depthrunpath=Path(item['depth'])/'RUN.json';depthrun=json.load(open(depthrunpath));kfpath=Path(item['directory'])/'keyframe.json';kf=json.load(open(kfpath))
 for path in [trackpath,posepath,maskpath,depthrunpath,kfpath]:sources[str(path)]=sha(path)
 confidence=np.array([r['score'] for r in masks]);selected=np.array([r['selected'] for r in masks]);iou=np.array(raw['iou']);validiou=iou[np.isfinite(iou)];humanfit=align[name];Kpred=np.array(depthrun['K_fixed_source']);Kgt=np.array([[cameras[name]['focal'][0],0,cameras[name]['princpt'][0]],[0,cameras[name]['focal'][1],cameras[name]['princpt'][1]],[0,0,1.]])
 r=dict(sequence=name,disk_name=disk,frames=n,expected_frames=len(item['native_frame_numbers']),all_frames_retained=True,mesh_units='metres',
  CD_cm=stats(cd),centroid_error_cm=stats(100*np.linalg.norm(delta,axis=1)),first_centroid_delta_xyz_cm=(100*delta[0]).tolist(),mean_centroid_delta_xyz_cm=(100*delta.mean(0)).tolist(),
  predicted_centroid_camera_m=pc.tolist(),GT_centroid_camera_m=gc.tolist(),centroid_delta_xyz_cm=(100*delta).tolist(),
  predicted_centroid_z_m=stats(pc[:,2]),GT_centroid_z_m=stats(gc[:,2]),centroid_depth_ratio=stats(pc[:,2]/gc[:,2]),
  predicted_metric_surface_RMS_radius_m=scale*p['rms_radius'],native_GT_surface_RMS_radius_m=g['rms_radius'],radius_ratio_pred_over_GT=radiusratio,
  predicted_metric_PCA_extent_lengths_m=(scale*np.array(p['pca_extents'])).tolist(),native_GT_PCA_extent_lengths_m=g['pca_extents'],PCA_extent_ratios=(scale*np.array(p['pca_extents'])/np.array(g['pca_extents'])).tolist(),
  object_scale_in_aligned_prediction=scale,predicted_human_alpha=humans[disk]['alpha'],human_alignment_scale=humanfit['primary_transform']['scale'],human_fit_mean_vertex_residual_cm=humanfit['primary_human_fit_mean_vertex_residual_cm'],human_fit_p95_vertex_residual_cm=humanfit['primary_human_fit_p95_vertex_residual_cm'],
  mask_confidence=stats(confidence),mask_selected_frames=int(selected.sum()),FP_mask_ok_frames=int(raw['mask_ok'].sum()),FP_image_IoU=stats(validiou),FP_events=track['events'],keyframe_local_index=kf['keyframe'],keyframe_native_frame=item['native_frame_numbers'][kf['keyframe']],
  predicted_K_source=Kpred.tolist(),native_K_for_scorer_diagnostic=Kgt.tolist(),focal_ratio_pred_over_native=[float(Kpred[0,0]/Kgt[0,0]),float(Kpred[1,1]/Kgt[1,1])],
  scorer_only_oracle_translation_to_GT_surface_centroid_CD_cm=stats(centeroracle),scorer_only_oracle_centroid_plus_shared_RMS_radius_normalization_CD_cm=stats(radiusoracle),
  diagnostic_CD_over50cm_frames=int(np.sum(np.array(cd)>50)),diagnostic_centroid_error_over25cm_frames=int(np.sum(np.linalg.norm(delta,axis=1)>.25)),
  raw_declared_score_matches=True,per_frame_CD_cm=cd,per_frame_centroid_oracle_CD_cm=centeroracle,per_frame_centroid_radius_oracle_CD_cm=radiusoracle)
 records.append(r);print(name,'CD',r['CD_cm']['mean'],'centroidcm',r['centroid_error_cm']['mean'],'radiusratio',radiusratio,'centerOracle',np.mean(centeroracle),'sizeOracle',np.mean(radiusoracle),'humanfitcm',r['human_fit_mean_vertex_residual_cm'],flush=True)
report=dict(schema='scorer-only-InterCap-object-failure-decomposition-v1',records=records,source_sha256=sources,all_takes=8,all_frames=sum(r['frames'] for r in records),reserve_takes_included=False,GT_used=True,GT_usage='scorer-only diagnostic; translation/radius oracle NEVER production initialization',core_or_protocol_modified=False,production_outputs_created=False,native_object_GT_registration_used_for_production=False,
 definitions=dict(surface_centroid='exact triangle-area centroid, transformed to native camera coordinates',surface_RMS_radius='exact area-weighted second moment about surface centroid',PCA_dimensions='extents of vertices used by nondegenerate triangles in area-covariance principal axes, sorted descending; descriptive dimensions not independent shape alignment',oracle_translation='per-frame GT area-centroid translation only, keep predicted scale/rotation/shape',oracle_radius='add one fixed GT/predicted RMS-radius ratio, about surface centroid, with same predicted rotations; no object rotation ICP',aligned_baseline='existing first-frame human-only Sim3 scorer output, unchanged'))
(OUT/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('wrote',OUT/'RESULT.json')
