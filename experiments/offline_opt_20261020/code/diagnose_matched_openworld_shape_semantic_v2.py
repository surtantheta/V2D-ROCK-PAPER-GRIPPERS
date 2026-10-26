#!/usr/bin/env python3
"""SCORER ONLY: semantic-filtered v2 GT Sim3-ICP shape diagnostic; v1 preserved.

Same image/mask, same kit4096 mesh budget, same fit/evaluation sample seeds,
24 proper PCA-frame starts and identical ICP limits. Alignment uses GT and must
never initialize production inference. This is NOT raw-camera CD or hosted score.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deps/hunyuan21'),str(ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit')]
from v2dlb.mesh_budget import budget_mesh
from labeled_object_scorer import sha256,load_surface,surface_samples

FIT_N=2048
EVAL_N=8192
FIT_SEED=314159
EVAL_SEED=271828
ITERATIONS=60


def proper_pca_starts(x,y):
    def basis(p):
        _,v=np.linalg.eigh(np.cov(p.T));v=v[:,::-1]
        if np.linalg.det(v)<0:v[:,-1]*=-1
        return v
    px,py=basis(x),basis(y)
    out=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.,1.],repeat=3):
            a=np.eye(3)[:,perm]@np.diag(signs)
            if np.linalg.det(a)>.99:out.append(py@a@px.T)
    assert len(out)==24
    return out


def fit_sim3(x,y):
    mx,my=x.mean(0),y.mean(0);a=x-mx;b=y-my
    u,d,vt=np.linalg.svd(a.T@b)
    signs=np.ones(3);signs[-1]=1 if np.linalg.det(vt.T@u.T)>0 else -1
    r=vt.T@np.diag(signs)@u.T
    scale=float(np.clip(np.dot(d,signs)/np.square(a).sum(),.25,4.))
    t=my-scale*mx@r.T
    return scale,r,t


def fit_shape(pred,gt):
    cp,cg=pred.mean(0),gt.mean(0)
    rp=float(np.sqrt(np.square(pred-cp).sum(1).mean()));rg=float(np.sqrt(np.square(gt-cg).sum(1).mean()))
    if min(rp,rg)<1e-8:raise ValueError('collapsed shape')
    x,y=(pred-cp)/rp,(gt-cg)/rg
    target_tree=cKDTree(y);solutions=[]
    for index,r in enumerate(proper_pca_starts(x,y)):
        scale,t=1.,np.zeros(3);previous=float('inf');best=None
        for step in range(ITERATIONS):
            posed=scale*x@r.T+t
            da,ia=target_tree.query(posed,workers=1)
            db,ib=cKDTree(posed).query(y,workers=1)
            loss=float(np.square(da).mean()+np.square(db).mean())
            if best is None or loss<best[0]:best=(loss,scale,r.copy(),t.copy(),step)
            if abs(previous-loss)<1e-10:break
            previous=loss
            scale,r,t=fit_sim3(np.concatenate([x,x[ib]]),np.concatenate([y[ia],y]))
        loss,scale,r,t,last=best
        # Convert normalized-space map back to original object coordinates.
        s=scale*rg/rp;translation=cg+rg*t-s*cp@r.T
        solutions.append({'start':index,'squared_normalized_fit_objective':loss,'scale':s,'R':r.tolist(),'t':translation.tolist(),
                          'normalized_scale':scale,'iterations':last+1,'scale_bound_active':abs(scale-.25)<1e-8 or abs(scale-4)<1e-8})
    selected=min(solutions,key=lambda a:a['squared_normalized_fit_objective'])
    return selected,solutions,rg


def distances(x,y):
    a=cKDTree(y).query(x,workers=1)[0];b=cKDTree(x).query(y,workers=1)[0]
    return float(a.mean()+b.mean()),a,b


def self_test():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(1500,3))*[.4,.15,.07];x[:200]+=[.6,.1,0]
    r=Rotation.from_rotvec([.8,-.6,1.2]).as_matrix();y=2.3*x@r.T+[1,-.2,.4]
    fit,_,_=fit_shape(x,y);posed=fit['scale']*x@np.array(fit['R']).T+fit['t']
    assert np.max(abs(posed-y))<1e-7 and np.linalg.det(fit['R'])>.999999
    assert abs(fit['scale']-2.3)<1e-7
    return {'known_Sim3_recovered':True,'proper_rotation_only':True,'max_point_error':float(abs(posed-y).max())}


def show_four(points,title,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,3))
    center=points.mean(0);extent=np.ptp(points,axis=0).max()/2*1.05
    for k,(elev,azim) in enumerate([(15,0),(15,90),(15,180),(75,225)]):
        ax=fig.add_subplot(1,4,k+1,projection='3d');ax.scatter(*points.T,s=.7,c=points[:,2],cmap='viridis',alpha=.7,rasterized=True)
        ax.set_xlim(center[0]-extent,center[0]+extent);ax.set_ylim(center[1]-extent,center[1]+extent);ax.set_zlim(center[2]-extent,center[2]+extent)
        ax.set_box_aspect([1,1,1]);ax.view_init(elev,azim);ax.set_axis_off()
    fig.suptitle(title);fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    if args.out.exists():raise ValueError('new output directory required')
    args.out.mkdir(parents=True);begin=time.monotonic()
    plan_path=ROOT/'labeled_predictions_v2/hunyuan21_shapes_v2/PLAN.json'
    audit_path=ROOT/'labeled_predictions_v2/SEMANTIC_MASK_AUDIT.json'
    runtime_path=ROOT/'labeled_predictions_v2/RUN_INPUTS.json'
    audit=json.loads(audit_path.read_text());changed={x['sequence'] for x in audit['records'] if x['generation_candidates_changed']}
    refs_path=ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'
    refs={r['sequence']:r for r in json.loads(refs_path.read_text())};plan=json.loads(plan_path.read_text())
    sources={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('labeled_object_scorer.py'),plan_path,refs_path,audit_path,runtime_path,ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py']}
    if len(plan['records'])!=4 or len({x['sequence'] for x in plan['records']})!=4:raise ValueError('four unique frozen sequences required')
    records=[]
    for item in plan['records']:
        seq=item['sequence'];rec={'sequence':seq,'local_frame_index':item.get('local_frame_index'),'video_frame_index':item.get('video_frame_index'),'arms':{}};records.append(rec)
        if item['status']!='planned':rec['status']='no_predicted_input';continue
        ref=refs[seq]
        for key in ['original','simplified']:
            p=Path(ref['object_mesh_'+key]);h=ref['object_mesh_'+key+'_sha256']
            if sha256(p)!=h:raise ValueError('pinned GT CAD changed')
            sources[str(p)]=h
        sv,_=load_surface(ref['object_mesh_simplified']);gv,gf=load_surface(ref['object_mesh_original'],sv.mean(0))
        gt_fit,_=surface_samples(gv,gf,FIT_N,FIT_SEED);gt_eval,_=surface_samples(gv,gf,EVAL_N,EVAL_SEED)
        gt_rms=float(np.sqrt(np.square(gt_eval-gt_eval.mean(0)).sum(1).mean()));gt_diag=float(np.linalg.norm(np.ptp(gv,axis=0)))
        sam_version='labeled_predictions_v2' if seq in changed else 'labeled_predictions_v1'
        sam_json=ROOT/sam_version/'shapes'/seq/f"proposal_{item['local_frame_index']:06d}.json"
        hy_json=Path(item['output'])/'PROPOSAL.json'
        outdir=args.out/seq;outdir.mkdir()
        show_four(gt_eval,'GT canonical — scorer only',outdir/'GT_fourview.jpg')
        for model,p in [('Hunyuan21',hy_json),('SAM3D',sam_json)]:
            arm={'status':'pending_or_missing_proposal','proposal_path':str(p)};rec['arms'][model]=arm
            if not p.exists():continue
            try:
                proposal=json.loads(p.read_text());sources[str(p)]=sha256(p)
                if model=='Hunyuan21':
                    if proposal['input_rgba_sha256']!=item['rgba_sha256'] or sha256(item['rgba'])!=item['rgba_sha256']:raise ValueError('Hunyuan RGBA mismatch')
                    for path,h in proposal['output_sha256'].items():
                        if sha256(path)!=h:raise ValueError('Hunyuan output artifact hash mismatch')
                        sources[path]=h
                    if proposal['coordinate_units']!='model-normalized arbitrary units; NOT metres':raise ValueError('unexpected coordinate declaration')
                    stored=np.load(Path(item['output'])/'shape_normalized_4096.npz')
                    stored_v,stored_f=stored['vertices'],stored['faces']
                    if stored_v.shape!=(4096,3) or stored_f.shape!=(4096,3) or not np.isfinite(stored_v).all():raise ValueError('stored Hunyuan budget invalid')
                    stored_tri=stored_v[stored_f]
                    stored_area=np.linalg.norm(np.cross(stored_tri[:,1]-stored_tri[:,0],stored_tri[:,2]-stored_tri[:,0]),axis=1)/2
                    if int((stored_area>1e-14).sum())!=proposal['nondegenerate_faces'] or not (stored_area>1e-14).any():raise ValueError('stored Hunyuan positive-area metadata mismatch')
                    raw=Path(item['output'])/'shape_normalized_raw.ply'
                    if str(raw) not in proposal['output_sha256']:raise ValueError('raw Hunyuan mesh must be explicitly hash-bound')
                    arm['runtime']={k:proposal[k] for k in ['cuda_peak_reserved_bytes','cuda_peak_allocated_bytes','host_peak_rss_KiB','seconds_model_loading','seconds_generation','cuda_device','cuda_capability']}
                    # Confirm stored model RGBA bytes match shared source RGB+mask.
                    from PIL import Image
                    rgb=np.asarray(Image.open(item['source_rgb']).convert('RGB'));mask=np.asarray(Image.open(item['source_mask']))>127
                    rgba=np.concatenate([rgb,(mask.astype(np.uint8)*255)[...,None]],-1)
                    if not np.array_equal(np.asarray(Image.open(item['rgba'])),rgba):raise ValueError('RGBA pixels differ from paired source construction')
                else:
                    for a,b in [('local_frame_index','local_frame_index'),('video_frame_index','video_frame_index'),('source_rgb_sha256','source_rgb_sha256'),('source_predicted_mask_sha256','source_mask_sha256')]:
                        if proposal[a]!=item[b]:raise ValueError('SAM3D matched conditioning/frame mismatch')
                    raw=Path(proposal['mesh_path'])
                    if sha256(raw)!=proposal['mesh_sha256']:raise ValueError('SAM3D mesh hash mismatch')
                    sources[str(raw)]=proposal['mesh_sha256']
                for key,hkey in [('source_rgb','source_rgb_sha256'),('source_mask','source_mask_sha256')]:
                    if sha256(item[key])!=item[hkey]:raise ValueError('frozen source input changed')
                    sources[item[key]]=item[hkey]
                v,f=budget_mesh(raw,faces=4096,vertices=4096)
                tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
                if v.shape!=(4096,3) or f.shape!=(4096,3) or not np.isfinite(v).all() or not (area>1e-14).any():raise ValueError('mesh budget finite/positive-area gate failed')
                budget=outdir/(model+'_budget4096.npz');np.savez_compressed(budget,vertices=v,faces=f);sources[str(budget)]=sha256(budget)
                fit_points,_=surface_samples(v,f,FIT_N,FIT_SEED);eval_points,_=surface_samples(v,f,EVAL_N,EVAL_SEED)
                fit,starts,_=fit_shape(fit_points,gt_fit)
                posed=fit['scale']*eval_points@np.array(fit['R']).T+fit['t'];cd,a,b=distances(posed,gt_eval)
                arm.update(status='scored_scoring_only_GT_alignment',mesh_source=str(raw),budget_vertices=len(v),budget_faces=len(f),nondegenerate_faces=int((area>1e-14).sum()),
                           fit=fit,all_start_results=starts,oracle_Sim3_aligned_CD_sum_cm=cd*100,
                           oracle_Sim3_aligned_CD_sum_div_GT_surface_RMS=cd/gt_rms,oracle_Sim3_aligned_CD_sum_div_GT_bbox_diagonal=cd/gt_diag,
                           directed_mean_cm=[float(a.mean()*100),float(b.mean()*100)],directed_p95_cm=[float(np.quantile(a,.95)*100),float(np.quantile(b,.95)*100)],
                           GT_surface_RMS_m=gt_rms,GT_bbox_diagonal_m=gt_diag,
                           source_axes_extents=np.ptp(v,axis=0).tolist(),source_coordinate_units='arbitrary normalized' if model=='Hunyuan21' else 'predicted MoGe-scaled units, metric calibration unverified')
                np.savez_compressed(outdir/(model+'_scorer_only_aligned_points.npz'),points=posed,GT_points=gt_eval)
                show_four(eval_points,model+' raw object coordinates — arbitrary display units',outdir/(model+'_raw_fourview.jpg'))
                show_four(posed,model+' SCORER-ONLY Sim3 alignment to GT',outdir/(model+'_GTaligned_fourview.jpg'))
                print(seq,model,'oracle_shape_CDsum_cm',cd*100,flush=True)
            except Exception as error:
                arm.update(status='diagnostic_failed',error=repr(error));print(seq,model,'FAILED',repr(error),flush=True)
        rec['complete_pair']=all(a['status']=='scored_scoring_only_GT_alignment' for a in rec['arms'].values())
    complete=all(r.get('complete_pair',False) for r in records)
    report={'schema':'scorer-only-matched-open-world-semantic-repair-shape-v2','WARNING':'GT-dependent alignment/scale used ONLY inside scorer. NOT raw camera CD, hosted metric, production pose/scale initialization or evidence of benchmark training independence.',
            'cohort':'same four BEHAVE clips/384 RGB frames; image-only semantic mask rejection before new first candidates; first3 reused, box8 regenerated; original v1 diagnostics preserved',
            'conditioning_difference':'Both same predicted RGB+mask. SAM3D additionally uses RGB-predicted MoGe and fixed predicted sequence-median K; Hunyuan shape uses RGBA only and its native crop/resize.',
            'mesh_budget':{'vertices':4096,'faces':4096,'source':'actual submission kit budget_mesh, both arms rebudgeted'},
            'fit':{'samples':FIT_N,'seed':FIT_SEED,'starts':24,'initializations':'proper PCA signed permutations','max_iterations_per_start':ITERATIONS,'loss':'symmetric mean squared nearest neighbor; no reflection; positive normalized scale clipped[.25,4]'},
            'evaluation':{'samples':EVAL_N,'seed':EVAL_SEED,'metric':'sum of directed mean Euclidean NN; independent deterministic area samples, cm after scoring-onlyGTscale'},
            'complete_four_pairs':complete,'records':records,'source_sha256':sources,'self_test':self_test(),'seconds':time.monotonic()-begin}
    (args.out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'complete_four_pairs':complete,'seconds':report['seconds']},indent=2))


if __name__=='__main__':main()
