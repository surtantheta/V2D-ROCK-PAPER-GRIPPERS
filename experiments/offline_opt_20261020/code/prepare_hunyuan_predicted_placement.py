#!/usr/bin/env python3
"""Place normalized Hunyuan meshes using SAME-INPUT generated SAM3D geometry only.

This CPU adapter never loads native CAD, benchmark labels or scoring transforms.
It bakes positive proper Sim3 into vertices, then carries SAM3D's predicted R/t/K.
No extra mesh recentering is performed. Nominal scale remains RGB-MoGe predicted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from predicted_geometry_sim3 import align_clouds

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deps/hunyuan21'),str(ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit')]
from v2dlb.mesh_budget import budget_mesh
POSE_CONVENTION='X_cam = object_scale * (V_local @ R_obj2cam.T) + t_obj2cam_m'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def generated_path(path, permitted):
    p=Path(path).resolve()
    if not any(p.is_relative_to(root.resolve()) for root in permitted):
        raise ValueError('geometry path must be within declared generator runtime outputs')
    return p


def surface_points(v,f,count,seed):
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    keep=area>1e-14;tri,area=tri[keep],area[keep]
    if not len(area) or not np.isfinite(area).all():raise ValueError('no finite positive-area surface')
    rng=np.random.default_rng(seed);chosen=rng.choice(len(tri),count,p=area/area.sum());uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    t=tri[chosen];return t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])


def valid_mesh(v,f):
    if v.shape!=(4096,3) or f.shape!=(4096,3) or not np.isfinite(v).all() or f.min()<0 or f.max()>=4096:
        raise ValueError('require finite valid exact4096 mesh rows')
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if not (area>1e-14).any():raise ValueError('no nondegenerate budget surface')
    return int((area>1e-14).sum())


def rotation(value):
    r=np.asarray(value,np.float64)
    if r.shape!=(3,3) or not np.isfinite(r).all() or abs(np.linalg.det(r)-1)>1e-6 or np.max(abs(r@r.T-np.eye(3)))>1e-6:
        raise ValueError('proper finite rotation required')
    return r


def sampled_sum(x,y):
    return float(cKDTree(y).query(x,workers=1)[0].mean()+cKDTree(x).query(y,workers=1)[0].mean())


def silhouette(v,f,r,t,k,hw):
    import cv2
    cam=v@r.T+t;zmin=float(cam[:,2].min())
    if zmin<=1e-4:raise ValueError('placed mesh crosses/behind camera; no silent clipping')
    projected=cam@k.T;xy=projected[:,:2]/projected[:,2:]
    if not np.isfinite(xy).all() or abs(xy).max()>1e6:raise ValueError('projection outside safe raster range')
    out=np.zeros(hw,np.uint8)
    for face in f:
        cv2.fillConvexPoly(out,np.rint(xy[face]).astype(np.int32),1)
    return out.astype(bool),zmin


def iou(a,b):return float((a&b).sum()/max((a|b).sum(),1))


def self_test():
    rng=np.random.default_rng(20261020);x=rng.normal(size=(1000,3))*[.4,.15,.06];x[:100]+=[.7,.2,0]
    ra=Rotation.from_rotvec([.7,-.4,1.1]).as_matrix();ta=np.array([.2,-.1,.07]);s=.35;y=s*x@ra.T+ta
    fit,_,_=align_clouds(x,y);actual=fit['scale']*x@np.array(fit['R']).T+fit['t'];error=float(abs(actual-y).max())
    if error>1e-8:raise AssertionError('synthetic generated-to-generated Sim3 failed')
    rc=Rotation.from_rotvec([-.2,.5,.3]).as_matrix();tc=np.array([.1,.4,2.])
    sequential=(s*x@ra.T+ta)@rc.T+tc
    composed=s*x@(rc@ra).T+rc@ta+tc
    composition=float(abs(sequential-composed).max())
    if composition>1e-12:raise AssertionError('baked-local-to-camera composition failed')
    return {'synthetic_Sim3_max_point_error':error,'baked_camera_composition_max_error':composition,'proper_rotation_only':True}


def keyframe_record(sam,mesh,mesh_hash,alignment=None):
    rec={k:sam[k] for k in ['local_frame_index','video_frame_index','R_obj2cam','t_obj2cam_m','K_predicted','K_policy','mask_score','source_rgb','source_rgb_sha256','source_predicted_mask','source_predicted_mask_sha256','mask_confidence_fallback','mask_confidence_policy']}
    rec.update(keyframe=sam['local_frame_index'],focal=sam['K_predicted'][0][0],t_obj2cam=sam['t_obj2cam_m'],
               mesh_path=str(mesh),mesh_sha256=mesh_hash,budgeted_mesh_sha256=mesh_hash,budget_faces=4096,budget_vertices=4096,
               mesh_units='metres',metric_scale_recovered=False,camera_pose_source='same-input RGB-generated SAM3D proposal',
               scale_source='same-input RGB-generated SAM3D mesh; RGB-predicted MoGe nominal metric scale; uncalibrated',
               origin_convention='SAM3D predicted local coordinate frame, no subsequent recenter; arbitrary generated origin',
               origin_correspondence_to_gt='unverified',selection_uses_GT=False)
    if alignment is not None:rec['Hunyuan_to_SAM3D_local_Sim3_baked_into_vertices']=alignment
    return rec


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'labeled_hunyuan_placement_v1');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    gates=self_test()
    if args.self_test:print(json.dumps(gates,indent=2));return
    if args.out.exists():raise ValueError('output must be new')
    manifest_path=ROOT/'labeled_predictions_v2/RUN_INPUTS.json';plan_path=ROOT/'labeled_predictions_v2/hunyuan21_shapes_v2/PLAN.json';audit_path=ROOT/'labeled_predictions_v2/SEMANTIC_MASK_AUDIT.json'
    manifest=json.loads(manifest_path.read_text());plan=json.loads(plan_path.read_text());audit=json.loads(audit_path.read_text())
    if manifest['GT_fields_included'] is not False or plan['GT_inputs_provided'] is not False or plan['native_CAD_input'] is not False or audit['GT_used'] is not False:
        raise ValueError('require explicitly predicted-only source manifests')
    inputs={x['sequence']:x for x in manifest['items'] if x['dataset']=='BEHAVE'};changed={x['sequence'] for x in audit['records'] if x['generation_candidates_changed']}
    if len(plan['records'])!=4 or {x['sequence'] for x in plan['records']}!=set(inputs):raise ValueError('require exactly the four frozen source clips')
    sources={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('predicted_geometry_sim3.py'),manifest_path,plan_path,audit_path,ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py']}
    args.out.mkdir(parents=True);records=[];predictions=[];sam_records=[]
    for candidate in plan['records']:
        seq=candidate['sequence'];item=inputs[seq];index=candidate['local_frame_index'];mask_manifest=Path(item['masks'])/'MASKS.json';rows=json.loads(mask_manifest.read_text())['rows'];row=rows[index]
        if not row['selected'] or row['person_duplicate_gate']['rejected']:raise ValueError('candidate rejected by current image-only gate')
        source_rgb=Path(item['source_paths'][index]);source_mask=Path(row['mask'])
        if sha(source_rgb)!=candidate['source_rgb_sha256'] or sha(source_mask)!=candidate['source_mask_sha256']:raise ValueError('candidate input hash mismatch')
        version='labeled_predictions_v2' if seq in changed else 'labeled_predictions_v1';sam_path=ROOT/version/'shapes'/seq/f'proposal_{index:06d}.json';sam=json.loads(sam_path.read_text())
        if sam['local_frame_index']!=index or sam['video_frame_index']!=candidate['video_frame_index'] or sam['source_rgb_sha256']!=candidate['source_rgb_sha256'] or sam['source_predicted_mask_sha256']!=candidate['source_mask_sha256']:
            raise ValueError('SAM target must be exact SAME INPUT FRAME; no best-proposal substitution')
        sam_mesh=generated_path(sam['mesh_path'],[ROOT/'labeled_predictions_v1/shapes',ROOT/'labeled_predictions_v2/shapes'])
        if sha(sam_mesh)!=sam['mesh_sha256']:raise ValueError('SAM generated mesh hash mismatch')
        hy_path=Path(candidate['output'])/'PROPOSAL.json';hy=json.loads(hy_path.read_text());hy_mesh=generated_path(Path(candidate['output'])/'shape_normalized_4096.npz',[ROOT/'labeled_predictions_v1/hunyuan21_shapes_v1',ROOT/'labeled_predictions_v2/hunyuan21_shapes_v2'])
        if str(hy_mesh) not in hy['output_sha256'] or sha(hy_mesh)!=hy['output_sha256'][str(hy_mesh)] or hy['input_rgba_sha256']!=candidate['rgba_sha256']:
            raise ValueError('H normalized generated geometry provenance mismatch')
        if hy['coordinate_units']!='model-normalized arbitrary units; NOT metres':raise ValueError('unexpected H coordinate declaration')
        if sha(candidate['rgba'])!=candidate['rgba_sha256']:raise ValueError('H source RGBA changed')
        for p in [mask_manifest,source_rgb,source_mask,Path(candidate['rgba']),sam_path,sam_mesh,hy_path,hy_mesh]:sources[str(p.resolve())]=sha(p)
        hv=np.load(hy_mesh);v=np.asarray(hv['vertices'],np.float64);f=np.asarray(hv['faces'],np.int64);valid_mesh(v,f)
        tv,tf=budget_mesh(sam_mesh,faces=4096,vertices=4096);valid_mesh(tv,tf)
        hf=surface_points(v,f,2048,314159);sf=surface_points(tv,tf,2048,314159);fit,starts,_=align_clouds(hf,sf)
        a=rotation(fit['R']);shift=np.asarray(fit['t'],np.float64);scale=float(fit['scale'])
        if not np.isfinite(scale) or scale<=0 or fit['scale_bound_active']:raise ValueError('invalid or bound-limited generated-to-generated fit')
        placed=scale*v@a.T+shift;positive=valid_mesh(placed,f)
        r=rotation(sam['R_obj2cam']);t=np.asarray(sam['t_obj2cam_m'],np.float64);k=np.asarray(sam['K_predicted'],np.float64)
        if t.shape!=(3,) or k.shape!=(3,3) or not np.isfinite(t).all() or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('invalid predicted camera')
        depth_run=ROOT/'labeled_predictions_v1/depth'/seq/'RUN.json';depth=json.loads(depth_run.read_text())
        if not np.allclose(depth['K_fixed_source'],k,rtol=0,atol=1e-5):raise ValueError('fixed predicted K/depth convention mismatch')
        sources[str(depth_run.resolve())]=sha(depth_run)
        for label,verts,faces in [('objects',placed,f),('sam3d_matched',tv,tf)]:
            out=args.out/label/seq;out.mkdir(parents=True);mesh=out/'object_mesh.ply';trimesh.Trimesh(verts,faces,process=False).export(mesh)
            decoded=trimesh.load(mesh,force='mesh',process=False)
            if len(decoded.vertices)!=4096 or len(decoded.faces)!=4096 or np.max(abs(np.asarray(decoded.vertices)-verts))>1e-6:raise ValueError('mesh file budget/coordinate roundtrip changed')
            record=keyframe_record(sam,mesh,sha(mesh),fit if label=='objects' else None);(out/'keyframe.json').write_text(json.dumps(record,indent=2,allow_nan=False))
            entry={'sequence':seq,'directory':str(out),'mesh_path':str(mesh),'mesh_sha256':sha(mesh)}
            if label=='objects':asset=entry
            else:sam_records.append(entry)
        evaluation=surface_points(placed,f,8192,271828);target_evaluation=surface_points(tv,tf,8192,271828)
        residual=sampled_sum(evaluation,target_evaluation)
        mask=np.asarray(Image.open(source_mask))>127;hr,zmin=silhouette(placed,f,r,t,k,mask.shape);sr,sam_zmin=silhouette(tv,tf,r,t,k,mask.shape)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,4));rgb=np.asarray(Image.open(source_rgb).convert('RGB'))
        for ax,title,binary in zip(axes,['predicted object mask','H placement silhouette','same-input SAM silhouette'],[mask,hr,sr]):
            ax.imshow(rgb);ax.contour(binary.astype(float),levels=[.5],colors=['red'],linewidths=.8);ax.set_title(title);ax.axis('off')
        fig.suptitle(seq+' — predicted inputs only');fig.tight_layout();fig.savefig(Path(asset['directory'])/'KEYFRAME_QA.jpg',dpi=140);plt.close(fig)
        composition_error=float(abs((scale*v@a.T+shift)@r.T+t-(scale*v@(r@a).T+r@shift+t)).max())
        if composition_error>1e-10:raise ValueError('baked local-to-camera composition failed')
        rec=dict(asset,keyframe=index,video_frame_index=candidate['video_frame_index'],source_H_proposal=str(hy_path),source_H_proposal_sha256=sha(hy_path),source_SAM_proposal=str(sam_path),source_SAM_proposal_sha256=sha(sam_path),
                 generated_to_generated_fit=fit,all_start_results=starts,nondegenerate_faces=positive,source_H_normalized_to_target_nominal_scale=scale,
                 target_SAM_origin_preserved=True,extra_recenter_after_alignment=False,source_RGB= str(source_rgb),source_mask=str(source_mask),masks=str(mask_manifest.parent),frames=item['frames'],depth=str(depth_run.parent),
                 H_to_SAM_independent_surface_residual_sum_nominal_cm=residual*100,H_keyframe_predicted_mask_IoU=iou(hr,mask),SAM_keyframe_predicted_mask_IoU=iou(sr,mask),
                 min_camera_z_nominal_m=zmin,SAM_min_camera_z_nominal_m=sam_zmin,camera_composition_max_error=composition_error)
        records.append(rec)
        predictions.append({'sequence':seq,'camera':1,'mesh_path':asset['mesh_path'],'mesh_sha256':asset['mesh_sha256'],'mesh_units':'metres','object_scale':1.,'pose_convention':POSE_CONVENTION,
                            'origin_convention':'H geometry baked into SAM3D predicted local frame; no further recenter; native-origin correspondence unverified',
                            'origin_correspondence_to_gt':'unverified','source':{'method':'predicted-only H->same-input SAM geometry Sim3; inherited SAM R/t/K','metric_scale_recovered':False,'source_record':str(Path(asset['directory'])/'keyframe.json')},
                            'frames':[{'video_frame_index':candidate['video_frame_index'],'R_obj2cam':r.tolist(),'t_obj2cam_m':t.tolist()}]})
        print(seq,'prepared',index,'scale',scale,'H maskIoU',iou(hr,mask),'SAM maskIoU',iou(sr,mask),flush=True)
    run={'schema':'predicted-only-Hunyuan-placement-v1','GT_inputs_provided':False,'native_CAD_input':False,'scorer_transform_used':False,
         'source_scope':'only generated runtime meshes/poses, predicted masks/RGB and predicted depth/K metadata; no native object models or label references',
         'placement_policy':'bake positive proper Sim3 from H normalized geometry to SAME INPUT SAM local geometry; keep SAM R/t/fixed K; no subsequent recenter',
         'metric_scale_recovered':False,'scale_and_pose_are_independent_Hunyuan_estimates':False,'scale_policy':'inherits RGB-MoGe/SAM3D uncalibrated nominal metres',
         'source_sha256':sources,'alignment':{'fit_samples':2048,'fit_seed':314159,'starts':24,'max_steps':60,'selection':'minimum generated-to-generated symmetric squared surface residual only','evaluation_samples':8192,'evaluation_seed':271828},
         'records':records,'self_test':gates}
    (args.out/'RUNTIME.json').write_text(json.dumps(run,indent=2,allow_nan=False))
    (args.out/'BUDGETED.json').write_text(json.dumps({'records':[{k:r[k] for k in ['sequence','directory','mesh_path','mesh_sha256']} for r in records],'source_runtime_sha256':sha(args.out/'RUNTIME.json')},indent=2))
    (args.out/'SAM3D_MATCHED_BUDGETED.json').write_text(json.dumps({'records':sam_records,'source_runtime_sha256':sha(args.out/'RUNTIME.json')},indent=2))
    (args.out/'KEYFRAME_PREDICTIONS.json').write_text(json.dumps({'schema':'opencv-metric-object-predictions-v1','sequences':predictions},indent=2))
    print(json.dumps({'prepared':len(records),'runtime':str(args.out/'RUNTIME.json'),'GT_inputs_provided':False},indent=2))


if __name__=='__main__':main()
