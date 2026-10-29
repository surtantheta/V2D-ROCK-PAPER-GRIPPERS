#!/usr/bin/env python3
"""CPU adapter: fixed submitted MAIN humans/K, cached RGB-only geometry, no GT.

--freeze is metadata/input pinning only. --run requires a PBS allocation and a
reviewed input manifest; it decodes the exact submitted human for visibility.
"""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).parent))
from real_mesh_optimizer import Config,solve,raster,camera,proper,take,expand_corrections

P=Path('/home/tanuj/V2D_challenge');R=P/'experiments/identity_fusion_20261029'
PREV=P/'experiments/vggt_fusion_20261028';MAIN=P/'work_labeled'
CALIB=Path('/rhome/tanuj/v2d_fusion_20261028/diagnostics/body_surface_calibration_v3/CALIBRATION_AND_CLOUDS_FROZEN.json')
PROTOCOL=dict(schema='real-rendered-mesh-ICP-v1',model_policy='uniform requested provider; no GT clip selector',
    baseline='submitted MAIN repaired object poses/mesh and exact human arrays',geometry='authoritative provider local XYZ, predicted MAIN-human V3 proper Sim3 applied once',
    camera='fixed MAIN S6 K; old 512x384 integer pixel convention',resolution=[256,192],
    observations='up to8/16 cached frames per clip; cloud max256/object surface max384 each; support is image/calibration/finite only, never initial 3D-nearness',
    masks='explicit selected/state; unchanged-mask control keeps known identity errors; same source pixel IDs across providers only if separately requested',
    objective='count-normalized robust rendered-surface ICP XYZ + mask coverage + visible free-space/holes; alternating frozen visibility/correspondence surrogate',
    visibility='frontmost exact MAIN human and object triangles; object-positive wins body overlap; behind-human elsewhere unknown',
    temporal='actual time correction-centroid/SO3 acceleration; baseline-relative interpolation for unobserved frames',
    contacts=False,collision=False,reason_no_contact='no verified conditional contact/collision inputs supplied; disabled rather than forced hand attraction',
    shared_scale=True,local_mesh_unchanged=True,GT_read=False,reserve_read=False,
    previous_development_scores_known=True,claim='fullmesh experimental candidate requiring separate paired CD/ACC/PEN evaluation; not a model or SOTA result')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def jsonable(value):
    # v3 harness-only repair. Preserve numerical values and reject NaN/Inf at
    # json.dumps; do not change the frozen optimizer or its diagnostics.
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,dict):return {jsonable(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def dump(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(jsonable(j),indent=2,allow_nan=False)+'\n')


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise AssertionError(('source/input changed',p))


def mask_rows(path):
    """Explicit flat adapter schema; no guessing ambiguous nested manifests."""
    if path is None:return {}
    j=json.loads(Path(path).read_text());rows=j['rows'];result={}
    for row in rows:
        key=(int(row['episode']),int(row['local_index']))
        if key in result:raise ValueError('duplicate identity mask row')
        for k in ['maskpath','selected','state','sourceRGBwh']:assert k in row,k
        result[key]=row
    return result


def freeze(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    target=out/'INPUTS.json'
    if target.exists():raise FileExistsError('input freeze already exists; use new output version')
    pilot=json.loads((PREV/'PILOT_INPUTS.json').read_text());calib=json.loads(CALIB.read_text())
    if calib['native_GT_values_read'] or calib['GT_object_fit']:raise AssertionError('calibration provenance')
    masks=mask_rows(a.mask_manifest);pins={};cases=[]
    for p in [Path(__file__),Path(__file__).with_name('real_mesh_optimizer.py'),PREV/'PILOT_INPUTS.json',CALIB,MAIN/'LABELED_EPISODES.json',P/'models/mhr/mhr_model.pt',P/'models/mhr/mhr_faces_lod1.npy']:
        # The large model is already source-pinned by V3. Reuse its expected
        # hash during cheap preparation; execute verifies the actual bytes.
        if str(p) in calib['source_sha256']:pins[str(p)]=calib['source_sha256'][str(p)]
        else:pins[str(p)]=sha(p)
    if a.mask_manifest:pins[str(Path(a.mask_manifest).resolve())]=sha(a.mask_manifest)
    for case in pilot['cases']:
        ep=case['episode']
        if ep not in a.episodes:continue
        assert 900<=ep<=911,'only frozen engineering development cohort allowed'
        record=next(x for x in calib['records'] if x['episode']==ep and x['model']==a.model)
        modelroot=Path(pilot['large_output_root'])/a.model/f'episode_{ep:06d}_normal'
        # Pilot input contains the runs root; resolve only the declared schema.
        if not modelroot.exists():
            modelroot=Path('/rhome/tanuj/v2d_fusion_20261028/runs/pilot_v1')/a.model/f'episode_{ep:06d}_normal'
        result=json.loads((modelroot/'RESULT.json').read_text())
        assert result['status']=='PASS' and result['episode']==ep
        geometry=modelroot/'GEOMETRY.npz'
        pins[str(geometry)]=result['output_sha256'][str(geometry)]
        pins[str(modelroot/'RESULT.json')]=sha(modelroot/'RESULT.json')
        paths=dict(baseline=str(MAIN/'submission'/f'episode_{ep:06d}.npz'),mesh=str(MAIN/'submission'/f'episode_{ep:06d}_object.ply'),
                   poses=str(MAIN/'s6'/f'episode_{ep:06d}'/'poses.npz'),geometry=str(geometry))
        for p in paths.values():
            if p not in pins:pins[p]=sha(p)
        samples=[]
        for pos in take(len(case['frames']),a.frames):
            fr=case['frames'][int(pos)];key=(ep,fr['local_index']);row=masks.get(key)
            if a.mask_manifest and row is None:raise ValueError(('missing identity frame',key))
            mask=fr['mask'] if row is None else row['maskpath'];pins[mask]=sha(mask)
            state='visible' if row is None else row['state'];selected=fr['mask_selected'] if row is None else bool(row['selected'])
            if row is not None:assert list(row['sourceRGBwh'])==list(fr['rgb_wh'])
            samples.append(dict(position=int(pos),frame=fr,mask=mask,selected=selected,state=state))
        cases.append(dict(episode=ep,dataset=case['dataset'],sequence=case['sequence'],total_frames=case['total_frames'],samples=samples,paths=paths,calibration=record))
    assert sorted(x['episode'] for x in cases)==sorted(a.episodes)
    config=Config();config_dict=dict(config.__dict__)
    j=dict(protocol=PROTOCOL,model=a.model,mask_treatment='old_mask_control' if not a.mask_manifest else 'new_identity',
           cases=cases,config=config_dict,source_sha256=pins,output=str(out.resolve()),frames_requested=a.frames,
           masks_manifest=None if not a.mask_manifest else str(Path(a.mask_manifest).resolve()),
           created_before_new_fullmesh_GT_scoring=True)
    dump(target,j);print('FROZEN',target,len(pins),'pins',len(cases),'cases',flush=True)


def decode_human(model,data,ids,torch):
    """Exact submitted MAIN MHR units, proven previous decoder; no GT alignment."""
    result=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8]
        pose=np.c_[data['pose'][ix],np.repeat(data['scales'][None],len(ix),0)].astype(np.float32)
        shape=np.repeat(data['shape'].reshape(1,45),len(ix),0).astype(np.float32)
        with torch.inference_mode():v,_=model(torch.from_numpy(shape),torch.from_numpy(pose),torch.zeros(len(ix),72),True)
        result.append(v.double().numpy()/100*np.array([1.,-1.,-1.]))
    return np.concatenate(result)


def sample_cloud(local,A,mask_path,rgb_wh,calibration,budget=256):
    """Mask pixels→staged RGB→provider A→bilinear direct local XYZ, once to MAIN."""
    mask=np.asarray(Image.open(mask_path).convert('L'))>127
    if (mask.shape[1],mask.shape[0])!=tuple(rgb_wh):raise ValueError('identity mask is not staged RGB size')
    ref=np.asarray(Image.fromarray(mask).resize((512,384),Image.Resampling.NEAREST),bool)
    eroded=ndimage.binary_erosion(ref,iterations=3);fallback=eroded.sum()<16
    keep=ref if fallback else eroded;yy,xx=np.nonzero(keep);ix=take(len(xx),budget);yy,xx=yy[ix],xx[ix]
    pix=np.c_[(xx+.5)*rgb_wh[0]/512-.5,(yy+.5)*rgb_wh[1]/384-.5,np.ones(len(xx))]@A.T
    uv=pix[:,:2]/pix[:,2,None];H,W=local.shape[:2]
    valid=(uv[:,0]>=-.5)&(uv[:,0]<=W-.5)&(uv[:,1]>=-.5)&(uv[:,1]<=H-.5)
    cloud=np.stack([ndimage.map_coordinates(local[:,:,k],[uv[:,1],uv[:,0]],order=1,mode='nearest') for k in range(3)],1)
    valid&=np.isfinite(cloud).all(1)&(cloud[:,2]>.001)
    s=calibration['scale'];R=proper(calibration['R']);t=np.asarray(calibration['t'])
    cloud=s*(cloud[valid]@R.T)+t
    return cloud,dict(mask_pixels=int(ref.sum()),eroded_pixels=int(eroded.sum()),erosion_fallback=bool(fallback),finite_available_points=int(valid.sum()),source_mask_wh=list(rgb_wh))


def run(a):
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('real-data model decode/optimization requires reserved PBS CPU job')
    path=Path(a.inputs);j=json.loads(path.read_text());out=Path(j['output'])
    if (out/'COMPLETE.json').exists():raise FileExistsError('completed output is immutable')
    verify(j['source_sha256']);import torch,trimesh
    torch.set_num_threads(2)
    model=torch.jit.load(str(P/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
    human_faces=np.load(P/'models/mhr/mhr_faces_lod1.npy').astype(np.int64)
    assert human_faces.min()>=0 and human_faces.max()<18439
    metadata=json.loads((MAIN/'LABELED_EPISODES.json').read_text());sub=out/'submission';sub.mkdir(exist_ok=True)
    records=[];output_pins={};start=time.time()
    for case in j['cases']:
        began=time.time();ep=case['episode'];paths=case['paths'];npz=np.load(paths['baseline']);data={k:npz[k] for k in npz.files}
        n=len(data['pose']);assert n==case['total_frames']==len(data['object_rotation'])
        mesh=trimesh.load(paths['mesh'],force='mesh',process=False);V=np.asarray(mesh.vertices,float);F=np.asarray(mesh.faces,np.int64)
        R0=proper(data['object_rotation']);t0=np.asarray(data['object_translation'],float);s0=float(data['object_scale'])
        times=np.asarray(metadata[str(ep)]['times'],float);assert len(times)==n and np.all(np.diff(times)>0)
        ids=np.array([x['frame']['local_index'] for x in case['samples']],int)
        K=np.load(paths['poses'])['K_model'].astype(float);K=K[0] if K.ndim==3 else K
        assert case['calibration']['MAIN_camera']['S6_model_wh']==[512,384], 'MAIN raster coordinate contract'
        assert np.allclose(K,case['calibration']['MAIN_camera']['S6_K'],atol=1e-6)
        scale2d=np.diag([.5,.5,1.]);Kr=scale2d@K
        human=decode_human(model,data,ids,torch);geometry=np.load(paths['geometry'])
        obs=[];obs_records=[];eligible=case['calibration']['calibration_metric_diagnostic_eligible']
        for sample,verts in zip(case['samples'],human):
            p=sample['position'];fr=sample['frame'];assert int(geometry['frame_local_indices'][p])==fr['local_index']
            assert int(geometry['native_frame_ids'][p])==fr['native_frame_id']
            assert abs(float(geometry['time_seconds'][p])-fr['time_seconds'])<1e-6
            assert abs(times[fr['local_index']]-fr['time_seconds'])<1e-6
            hz,_,_=raster(verts,human_faces,Kr,192,256)
            image=Image.open(sample['mask']).convert('L');assert image.size==tuple(fr['rgb_wh'])
            mask=np.asarray(image.resize((256,192),Image.Resampling.NEAREST))>127
            framebody=case['calibration']['frames'][p]['calibrated_body_residual_cm']
            reliable=eligible and framebody is not None and framebody['mean']<=15 and case['calibration']['frames'][p]['body']['body_pixels']>=64
            enabled=bool(sample['selected']) and sample['state'] not in ['absent','uncertain','ambiguous'] and bool(mask.any())
            cloud,cloudinfo=sample_cloud(geometry['local_camera_points'][p],geometry['rgb_to_model'][p],sample['mask'],fr['rgb_wh'],case['calibration']['calibration'])
            # Mask-only optimization may remain enabled with unreliable provider
            # geometry; it cannot change scale (no metric anchor admitted).
            if not reliable:cloud=cloud[:0]
            obs.append(dict(mask=mask,cloud=cloud,human_z=hz,enabled=enabled))
            obs_records.append(dict(local_index=fr['local_index'],native_frame_id=fr['native_frame_id'],time_seconds=fr['time_seconds'],
                selected=sample['selected'],state=sample['state'],enabled=enabled,provider_metric_reliable=bool(reliable),
                admitted_cloud_points=len(cloud),**cloudinfo))
        cfg=Config(**{**j['config'],'scale_bounds':tuple(j['config']['scale_bounds'])})
        fit=solve(V,F,R0[ids],t0[ids],s0,Kr,times[ids],obs,cfg)
        R,t,s=expand_corrections(V,F,R0,t0,s0,times,ids,fit)
        minz=min(float(camera(V,R[f],t[f],s)[:,2].min()) for f in range(n))
        fallback=fit['fallback']
        if not np.isfinite(t).all() or not np.isfinite(s) or s<=0 or minz<cfg.minimum_z_m:
            fallback='invalid_expanded_fullmesh_geometry';R,t,s=R0.copy(),t0.copy(),s0
        output={**data,'object_rotation':R,'object_translation':t,'object_scale':np.asarray(s)}
        for k in ['pose','scales','shape']:assert np.array_equal(output[k],data[k]) and output[k].dtype==data[k].dtype
        target=sub/f'episode_{ep:06d}.npz';np.savez_compressed(target,**output)
        targetmesh=sub/f'episode_{ep:06d}_object.ply';shutil.copyfile(paths['mesh'],targetmesh)
        assert sha(targetmesh)==j['source_sha256'][paths['mesh']]
        diag=dict(episode=ep,frames=n,observed_frames=ids.tolist(),unobserved_frames_are_baseline_relative_interpolated_predictions=True,
            provider=j['model'],mask_treatment=j['mask_treatment'],baseline_scale=s0,final_scale=s,fullmesh_candidate_min_z_m=minz,
            fallback=fallback,observations=obs_records,fit=fit['diagnostics'],runtime_seconds=time.time()-began,
            human_bitexact=True,mesh_byteexact=True,GT_read=False)
        diagpath=out/f'episode_{ep:06d}_FIT.json';dump(diagpath,diag)
        for p in [target,targetmesh,diagpath]:output_pins[str(p)]=sha(p)
        records.append(dict(episode=ep,frames=n,fallback=fallback,diagnostics=str(diagpath)))
        print('CASE_COMPLETE',ep,n,'fallback',fallback,'seconds',time.time()-began,flush=True)
    verify(j['source_sha256'])
    total=sum(x['frames'] for x in records)
    if len(records)==12:assert total==630
    complete=dict(schema='real-mesh-fit-complete-v1',cases=len(records),frames=total,records=records,submission=str(sub),
        input_manifest=str(path),input_manifest_sha256=sha(path),source_sha256=j['source_sha256'],output_sha256=output_pins,
        human_bitexact=True,mesh_byteexact=True,fullmesh_front_checked=True,GT_read=False,GT_used_for_inference=False,reserve_read=False,
        runtime_seconds=time.time()-start,protocol=j['protocol'])
    dump(out/'COMPLETE.json',complete);print('REAL_MESH_FIT_COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--output')
    p.add_argument('--inputs');p.add_argument('--model',choices=['pi3x','vggt1b'],default='pi3x');p.add_argument('--mask-manifest')
    p.add_argument('--frames',type=int,default=8);p.add_argument('--episodes',type=int,nargs='+',default=list(range(900,912)))
    a=p.parse_args()
    if a.freeze:
        assert a.output and 2<=a.frames<=16;freeze(a)
    elif a.run:
        assert a.inputs;run(a)
    else:p.error('choose --freeze or --run')


if __name__=='__main__':main()
