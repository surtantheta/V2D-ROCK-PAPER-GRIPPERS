"""Bounded object translation smoothing; no human, mesh, rotation or scale edits."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT=Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/physical_refinement_20261031')
LARGE=Path('/rhome/tanuj/v2d_physical_20261031')
OLD=Path('/home/tanuj/V2D_challenge')
PRIOR=OLD/'experiments/identity_fusion_20261029'
CHALLENGE=Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')
LOCAL=Path('/rhome/tanuj/v2d_identity_20261029/mesh_fit/old_pi3x_v3/submission')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def dump(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def verify(pins):
    for p,d in pins.items():assert sha(p)==d,('input changed',p)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def smooth_translation(t,frame_ids,strength,max_change=.02):
    """Frame IDs are original 30fps IDs; gaps split instead of being bridged."""
    t=np.asarray(t,np.float64); ids=np.asarray(frame_ids,np.int64)
    assert t.shape==(len(ids),3) and np.isfinite(t).all() and (np.diff(ids)>0).all()
    delta=np.zeros_like(t);split=np.flatnonzero(np.diff(ids)!=np.median(np.diff(ids)))+1
    groups=np.split(np.arange(len(ids)),split)
    for g in groups:
        if len(g)<3:continue
        stride=int(ids[g[1]]-ids[g[0]]);assert stride>0 and (np.diff(ids[g])==stride).all()
        n=len(g);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
        a=sparse.eye(n,format='csc')+(float(strength)/(stride**4))*(d.T@d)
        delta[g]=spsolve(a,t[g])-t[g]
    proposed=float(np.linalg.norm(delta,axis=1).max(initial=0))
    factor=min(1.,max_change/max(proposed,1e-30))
    out=t+factor*delta
    assert np.linalg.norm(out-t,axis=1).max(initial=0)<=max_change+1e-12
    return out,dict(proposed_max_m=proposed,global_clip_factor=factor,max_change_m=float(np.linalg.norm(out-t,axis=1).max(initial=0)))

def own_motion(t):
    t=np.asarray(t,float);a=np.linalg.norm(np.diff(t,n=2,axis=0),axis=1)*100
    return dict(own_acc_cm_per_native_sample_squared_mean=float(a.mean()),own_acc_max=float(a.max()),is_reference_acceleration_error=False)

def selftest():
    ids=np.arange(80);linear=np.c_[ids*.001,ids*.002,np.ones(80)*3]
    s,_=smooth_translation(linear,ids,64);assert np.allclose(s,linear,atol=1e-11,rtol=0)
    noisy=linear.copy();noisy[:,2]+=.01*(-1.)**ids
    s,diag=smooth_translation(noisy,ids,64)
    assert np.linalg.norm(np.diff(s,n=2,axis=0))<np.linalg.norm(np.diff(noisy,n=2,axis=0))
    large=noisy.copy();large[40,2]+=1
    b,diag=smooth_translation(large,ids,64);assert abs(diag['max_change_m']-.02)<1e-12
    # Fixed physical regularization: stride5 lambda64 equals stride1 lambda64/625.
    q=np.c_[np.sin(ids*.1),ids*.001,np.ones(80)*3]
    a,_=smooth_translation(q,ids*5,64,max_change=1)
    b,_=smooth_translation(q,ids,64/625,max_change=1)
    assert np.allclose(a,b,atol=1e-12,rtol=0)
    return dict(linear_motion_preserved=True,alternating_noise_reduced=True,bound_enforced=True,stride_scaling_exact=True)

def generate(manifest):
    assert os.environ.get('PBS_JOBID');verify(manifest['sha256'])
    dump(ROOT/'TEMPORAL_SYNTHETICS_V2.json',selftest())
    metadata=json.loads((OLD/'work_labeled/LABELED_EPISODES.json').read_text())
    allrecords=[]
    for cohort,source,eps in [('challenge30',CHALLENGE,range(30)),('local12',LOCAL,range(900,912))]:
        for strength in [16,64]:
            out=LARGE/f'temporal_v2/{cohort}/lambda_{strength}/submission';assert not out.exists();out.mkdir(parents=True)
            rows=[];pins={}
            for ep in eps:
                name=f'episode_{ep:06d}';npz=source/f'{name}.npz';mesh=source/f'{name}_object.ply'
                data=dict(np.load(npz,allow_pickle=False));t=data['object_translation'];n=len(t)
                ids=np.arange(n) if cohort=='challenge30' else np.array(metadata[str(ep)]['frame_ids'])
                updated,diag=smooth_translation(t,ids,strength)
                dest=dict(data);dest['object_translation']=updated
                for k in data:
                    if k!='object_translation':assert dest[k].tobytes()==data[k].tobytes() and dest[k].dtype==data[k].dtype
                np.savez_compressed(out/f'{name}.npz',**dest);shutil.copyfile(mesh,out/f'{name}_object.ply')
                assert sha(mesh)==sha(out/f'{name}_object.ply')
                for p in [out/f'{name}.npz',out/f'{name}_object.ply']:pins[str(p)]=sha(p)
                rows.append(dict(episode=ep,frames=n,diag=diag,before=own_motion(t),after=own_motion(updated),source_npz=str(npz)))
                print('TEMPORAL_CASE',cohort,strength,ep,diag,flush=True)
            verify(manifest['sha256'])
            receipt=out.parent/'COMPLETE.json'
            result=dict(status='BOUNDED_TEMPORAL_CANDIDATE_COMPLETE',cohort=cohort,strength=strength,cases=len(rows),frames=sum(r['frames'] for r in rows),records=rows,output_sha256=pins,GT_used_for_inference=False,MAIN_human_bitexact=True,mesh_bitexact=True,rotation_scale_bitexact=True,maximum_translation_change_m=.02,protocol_sha256=sha(ROOT/'INPUTS_V2.json'),CPU_job=os.environ['PBS_JOBID'])
            dump(receipt,result);allrecords.append(dict(cohort=cohort,strength=strength,receipt=str(receipt),sha256=sha(receipt)))
    dump(ROOT/'TEMPORAL_GENERATION_COMPLETE_V2.json',dict(records=allrecords,CPU_job=os.environ['PBS_JOBID'],GT_used_for_inference=False))

def evaluate(strength):
    assert os.environ.get('PBS_JOBID')
    base=LARGE/f'temporal_v2/local12/lambda_{strength}'
    candidate=base/'submission';receipt=base/'COMPLETE.json'
    # Exact existing evaluator functions; only permitted output root is redirected.
    import argparse
    ev=module(PRIOR/'code/evaluate_candidate.py',f'physical_cd_{strength}');ev.H=ROOT
    args=argparse.Namespace(candidate=candidate,completion=receipt,out=ROOT/f'local_lambda_{strength}_scores_v2',baseline_parity=False,label=f'Pi3X old-mask + bounded temporal lambda{strength}')
    ev.run(args)
    # The immutable PEN runner owns torch/numba thread setup once in this process.
    pe=module(PRIOR/'code/evaluate_penetration.py',f'physical_pen_{strength}');pe.H=ROOT
    pe.run(argparse.Namespace(candidate=candidate,completion=receipt,out=ROOT/f'local_lambda_{strength}_pen_v2'))

def audit30():
    assert os.environ.get('PBS_JOBID')
    import torch,numba
    torch.set_num_threads(2);torch.set_num_interop_threads(1);numba.set_num_threads(2)
    pe=module(PRIOR/'code/evaluate_penetration.py','physical_pen_audit');pen=pe.module(pe.PEN,'physical_official_pen')
    decode=module(PRIOR/'code/run_real_mesh_fit_v3.py','physical_decode')
    sys.path.insert(0,str(pe.KIT));from v2dlb.mesh_budget import budget_mesh
    hand=np.load(pe.HAND);assert len(hand)==5840
    model=torch.jit.load(str(OLD/'models/mhr/mhr_model.pt'),map_location='cpu').eval()
    import pyarrow.parquet as pq
    sample=pq.read_table(pe.KIT/'data/track_1_sample_submission.parquet',columns=['row_id']).column(0).to_pylist()
    scored={e:set() for e in range(30)}
    for row in sample:
        a=row.split('_');ep=int(a[1]);frame=int(a[2])
        if frame!=999999:scored[ep].add(frame)
    out=LARGE/'audit30_v2';assert not out.exists();out.mkdir(parents=True)
    rows=[]
    for ep in range(30):
        name=f'episode_{ep:06d}';data=dict(np.load(CHALLENGE/f'{name}.npz',allow_pickle=False));n=len(data['pose'])
        allowed=np.array(sorted(scored[ep]),int);ids=allowed[np.unique(np.rint(np.linspace(0,len(allowed)-1,min(16,len(allowed)))).astype(int))]
        human=decode.decode_human(model,data,ids,torch);hands=human[:,hand]
        v,f=budget_mesh(CHALLENGE/f'{name}_object.ply',4096,4096)
        arms={}
        for label,folder in [('Pi3X',CHALLENGE),('temporal16',LARGE/'temporal_v2/challenge30/lambda_16/submission'),('temporal64',LARGE/'temporal_v2/challenge30/lambda_64/submission')]:
            a=dict(np.load(folder/f'{name}.npz',allow_pickle=False));rot=a['object_rotation'][ids];t=a['object_translation'][ids];s=float(a['object_scale'])
            local=np.einsum('tpi,tij->tpj',hands-t[:,None],rot)/s
            depth=pen._penetration_depth(local.reshape(-1,3),v,f).reshape(len(ids),len(hand))*s
            assert np.isfinite(depth).all() and (depth>=0).all()
            arms[label]=dict(mean_camera_gauge_pen_proxy_cm=float(depth.mean()*100),maximum_point_depth_cm=float(depth.max()*100),positive_point_fraction=float((depth>0).mean()),frames_with_penetration=int((depth>0).any(axis=1).sum()),own_full_trajectory_motion=own_motion(a['object_translation']))
            np.savez_compressed(out/f'{name}_{label}.npz',frame_ids=ids,hand_vertex_ids=hand,depth_scene_m=depth)
        np.savez_compressed(out/f'{name}_HUMAN_SAMPLE.npz',frame_ids=ids,hand_vertex_ids=hand,hands=hands)
        r=dict(episode=ep,full_frames=n,scored_frames=len(allowed),sample_frame_ids=ids.tolist(),arms=arms)
        rows.append(r);dump(out/f'CASE_{ep}.json',r);print('AUDIT30_CASE',ep,arms,flush=True)
    dump(out/'COMPLETE.json',dict(status='ALL30_SAMPLED_PEN_FULL_TRAJECTORY_AUDIT_COMPLETE',records=rows,cases=30,full_trajectory_frames=sum(r['full_frames'] for r in rows),penetration_sample_frames=sum(len(r['sample_frame_ids']) for r in rows),hosted_hand_roles=False,hand_proxy_points=5840,host_alignment_not_available=True,GT_used=False,CPU_job=os.environ['PBS_JOBID']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate','audit30','evaluate','test']);ap.add_argument('--strength',type=int);args=ap.parse_args()
    if args.stage=='test':print(json.dumps(selftest()))
    else:
        manifest=json.loads((ROOT/'INPUTS_V2.json').read_text());verify(manifest['sha256'])
        if args.stage=='generate':generate(manifest)
        elif args.stage=='audit30':audit30()
        else:evaluate(args.strength)
