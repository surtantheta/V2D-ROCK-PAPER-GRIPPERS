#!/usr/bin/env python3
"""Freeze RGB-only engineering requests separately from scorer labels and QA."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import trimesh

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
OUT=ROOT/'benchmarks/labeled_dev_v2'
SEQUENCES=['Date01_Sub01_stool_move','Date01_Sub01_monitor_hand','Date01_Sub01_chairblack_hand','Date01_Sub01_boxsmall_hand']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False))

def main():
    if OUT.exists():raise RuntimeError('immutable output already exists')
    OUT.mkdir(parents=True);(OUT/'scorer').mkdir();requests=[];scorers=[];qa=[]
    gate=Path('/rhome/tanuj/unicon3rpp_runs/sam3d_contract/gate3/gate3_manifest.json')
    refs=json.loads(gate.read_text())['samples']
    calib=Path('/rhome/tanuj/BEHAVE_raw/calibs/intrinsics/1/calibration.json')
    c=json.loads(calib.read_text())['color'];K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    dist=np.array([c[x] for x in ['k1','k2','p1','p2','k3','k4','k5','k6']])
    for seq in SEQUENCES:
        video=Path('/rhome/tanuj/BEHAVE_media')/(seq+'.1.color.mp4');timing=video.with_name(seq+'.1.time.json')
        stamps=np.asarray(json.loads(timing.read_text())['color'],float)/1e6
        start=int(np.abs(stamps-24.).argmin());indices=np.arange(start,start+96)
        dest=OUT/'rgb'/seq;dest.mkdir(parents=True)
        cap=cv2.VideoCapture(str(video));assert cap.isOpened();cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        paths=[];hashes=[];first=None
        for i in indices:
            ok,bgr=cap.read();assert ok,(seq,int(i))
            if first is None:first=bgr.copy()
            p=dest/f'{int(i):06d}.jpg';assert cv2.imwrite(str(p),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]);paths.append(str(p));hashes.append(sha(p))
        cap.release()
        requests.append({'sequence':seq,'camera':1,'rgb_paths':paths,'rgb_sha256':hashes,'video_frame_indices':indices.tolist(),'timestamps_seconds':stamps[indices].tolist(),'object_prompt':seq.split('_')[2],
                         'permitted_inference_inputs':['RGB','object text prompt'],'GT_fields_provided':False})
        gt=Path('/rhome/tanuj/BEHAVE_dev01/gt')/(seq+'.npz');data=np.load(gt,allow_pickle=True)
        names=data['frames'].astype(str);times=np.array([float(Path(x).name.lstrip('t')) for x in names])
        match=np.abs(times[:,None]-stamps[indices][None]).argmin(0);errors=np.abs(times[match]-stamps[indices])
        assert errors.max()<.02 and len(np.unique(match))==96,(seq,errors.max())
        row=next(x for x in refs if x['sequence']==seq)
        original=Path(row['template_original']);simple=Path(row['template_simplified'])
        m=trimesh.load(original,process=False);sm=trimesh.load(simple,process=False)
        center=np.asarray(sm.vertices).mean(0);verts=np.asarray(m.vertices)-center
        transform=data['pose_abs'][match[0]];points=verts@transform[:3,:3].T+transform[:3,3]
        assert points[:,2].min()>0
        projected=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,dist)[0][:,0]
        overlay=first.copy()
        for x,y in projected[::max(1,len(projected)//3000)]:
            if 0<=x<overlay.shape[1] and 0<=y<overlay.shape[0]:cv2.circle(overlay,(int(round(x)),int(round(y))),1,(0,220,255),-1)
        qpath=OUT/'scorer'/f'{seq}_GT_projection_QA.jpg';cv2.imwrite(str(qpath),overlay)
        scorers.append({'sequence':seq,'gt_npz':str(gt),'gt_sha256':sha(gt),'GT_row_indices':match.tolist(),'GT_frame_names':names[match].tolist(),'timestamp_errors_seconds':errors.tolist(),
                        'object_mesh_original':str(original),'object_mesh_simplified':str(simple),'mesh_origin_subtracted_mean':center.tolist(),'calibration_json':str(calib),'calibration_sha256':sha(calib),
                        'object_transform':'camera-frame metric R/t from prepared pose_abs, camera1 extrinsic identity','rational_distortion_applied_in_QA':True,'GT_overlay':str(qpath)})
        qa.append({'sequence':seq,'decoded_frames':96,'image_wh':list(first.shape[1::-1]),'max_time_join_error_seconds':float(errors.max()),'unique_GT_rows':len(np.unique(match)),
                   'source_frame_interval_seconds_minmax':[float(np.diff(stamps[indices]).min()),float(np.diff(stamps[indices]).max())],'scorer_projection_bbox_xyxy':[*projected.min(0).tolist(),*projected.max(0).tolist()]})
        print(seq,'decoded96','max timeerror',errors.max(),flush=True)
    plan=Path('/home/tanuj/contact_extension_gtcrop/INTERCAP_TRAINING_PLAN.json');p=json.loads(plan.read_text());takes=sorted({x['sequence_id'] for x in p['splits']['dev']},key=lambda x:hashlib.sha256(('v2d-opt-20261020:'+x).encode()).hexdigest())
    split={'scope':'optimizer engineering development on native training release; reserve only against new optimizer tuning, known shared-model training exposure; not ancestry-clean generalization',
           'dev_takes':takes[:8],'reporting_reserve_takes':takes[8:],'selection':'SHA256(v2d-opt-20261020:<take>), first8 dev','plan_sha256':sha(plan),'take_disjoint':True,'subject_disjoint':False,'reserve_labels_read_for_new_scores':False}
    inter=[]
    for take in takes[:8]:
        rows=[x for x in p['splits']['dev'] if x['sequence_id']==take and '/Frames_Cam1/' in x['file_name']]
        rows=sorted(rows,key=lambda x:x['file_name'])
        selected=rows[:min(32,len(rows))]
        inter.append({'take':take,'camera':1,'rgb_paths':[str(Path('/home/tanuj/contact_extension_gtcrop/intercap-train-media/sequences')/x['file_name']) for x in selected],
                      'native_frame_numbers':[int(Path(x['file_name']).stem) for x in selected],'object_prompt':'object '+take.split('/')[1],'GT_fields_provided':False,'source_frame_cadence':5,'exact_hosted_ACC_compatible':False})
        assert all(Path(x).is_file() for x in inter[-1]['rgb_paths'])
    write(OUT/'INFERENCE_REQUESTS.json',{'schema':'rgb-only-labeled-dev-v1','BEHAVE':requests,'InterCap':inter,'native_calibration_provided_to_inference':False,'released_masks_provided_to_inference':False})
    write(OUT/'scorer/BEHAVE_LABEL_REFERENCES.json',scorers);write(OUT/'INTERCAP_OPTIMIZER_SPLIT.json',split)
    write(OUT/'READINESS.json',{'code_sha256':sha(__file__),'QA':qa,'RGB_frames_decoded':384,'GT_overlays_for_scorer_QA_only':True,'GT_accuracy_of_any_model_measured':False,'BEHAVE_ancestry':'known Date01 shared-model training exposure, engineering dev only','graph_discovery_fallback':'MCP transport closed during code lookup; targeted source read used'})
    print('complete',OUT,flush=True)
if __name__=='__main__':main()
