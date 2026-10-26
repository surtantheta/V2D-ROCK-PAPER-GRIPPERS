#!/usr/bin/env python3
"""CPU metadata/packed-schema audit of main ep0 and remaining30episode critical path."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd,trimesh
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1];KIT=PROJECT/'data/assets/submission_kit/v2d_submission_kit';sys.path.insert(0,str(KIT));from v2dlb import mhr_submission as ms

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def headers(p):
 out={}
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   with z.open(n) as f:
    version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);out[n[:-4]]=dict(shape=list(shape),dtype=str(dtype))
 return out

def main():
 out=ROOT/'main_pilot_readiness_v1';out.mkdir(exist_ok=True);src=PROJECT/'work/submission_v0/episode_000000.npz';data=np.load(src,allow_pickle=False);sample_path=KIT/'data/track_1_sample_submission.parquet';sample=pd.read_parquet(sample_path);subset_path=PROJECT/'work/pack_test/track1_subset.parquet';subset=pd.read_parquet(subset_path);subsample=pd.read_parquet(PROJECT/'work/pack_test/sample_subset.parquet');parts=subset.row_id.str.split('_',expand=True);block=subset.copy();block['episode_index']=parts[1].astype(int);block['frame_index']=parts[2].astype(int);block['array_name']=parts[3].astype(int).map(dict(enumerate(ms.array_order())));block['point_index']=parts[4].astype(int);raw=ms._to_arrays(block,['x','y','z']);ids=np.sort(block.loc[block.frame_index!=999999,'frame_index'].unique());assert len(ids)==169 and ids[0]==90 and ids[-1]==699
 assert subset.row_id.is_unique and subset.row_id.equals(subsample.row_id);assert set(subset.row_id)==set(sample.loc[sample.row_id.str.startswith('t1_000000_'),'row_id']);assert np.isfinite(subset[['x','y','z']]).all().all()
 errors={}
 for key,n in [('pose',136),('scales',68),('shape',45)]:
  val=ms._unpack(raw['mhr_'+key],n,key);expected=data[key][ids] if key=='pose' else data[key];errors[key]=float(abs(val-expected).max());assert errors[key]==0
 errors['R']=float(abs(raw['object_rotation']-data['object_rotation'][ids]).max());errors['t']=float(abs(raw['object_translation'][:,0]-data['object_translation'][ids]).max());assert max(errors.values())==0;ms._rotations(data['object_rotation']);assert ms._object_scale(raw['object_scale'])==float(data['object_scale'])==1.
 mesh=trimesh.Trimesh(raw['object_mesh_vertices'],raw['object_mesh_faces'].astype(int),process=False);assert len(mesh.vertices)==len(mesh.faces)==4096 and np.isfinite(mesh.vertices).all() and mesh.faces.min()>=0 and mesh.faces.max()<4096;positive=mesh.area_faces>1e-14;clean=trimesh.Trimesh(mesh.vertices,mesh.faces[positive],process=True);clean.remove_unreferenced_vertices();meshq=dict(vertices=4096,faces=4096,positive_faces=int(positive.sum()),physical_vertices=len(clean.vertices),watertight=bool(clean.is_watertight),winding_consistent=bool(clean.is_winding_consistent),components=len(clean.split(only_watertight=False)),extent_m=clean.extents.tolist(),surface_area_m2=float(clean.area))
 source_mesh=trimesh.load(PROJECT/'work/submission_v0/episode_000000_object.ply',process=False);T=len(data['pose']);npzmeta={k:dict(shape=list(data[k].shape),dtype=str(data[k].dtype),finite=bool(np.isfinite(data[k]).all())) for k in data.files};assert npzmeta['pose']['shape']==[790,136] and npzmeta['scales']['shape']==[68] and npzmeta['shape']['shape']==[45] and all(x['finite'] for x in npzmeta.values());assert all(len(data[k])==T for k in ['object_rotation','object_translation'])
 verify=json.loads((PROJECT/'logs/track1_verify_report.json').read_text());inventory=[]
 for ep in verify['episodes']:
  e=f"episode_{ep['episode']:06d}";w=PROJECT/'work';n=ep['decoded_frames'];maskfile=w/'masks'/e/'MASKS.json';rows=json.loads(maskfile.read_text())['rows'];human=json.loads((w/'s3_human'/e/'RUN.json').read_text());body=json.loads((w/'s3db'/e/'BBOX.json').read_text());head=headers(w/'s3db'/e/'sam3d_body.npz');assert len(rows)==human['frames']==body['frames']==n;assert head['vertices_cam']['shape']==[n,18439,3] and head['mhr_model_params']['shape']==[n,204]
  confident=[(float(r['area']),i,r) for i,r in enumerate(rows) if r['selected'] and r['score']>=.8];sampled=[]
  for _,i,row in sorted(confident,reverse=True)[:12]:
   m=np.asarray(Image.open(row['mask']))>127;y,x=np.nonzero(m)
   if len(x) and x.min()>2 and y.min()>2 and x.max()<1533 and y.max()<1149:sampled.append(i)
  inventory.append(dict(episode=ep['episode'],frames=n,object=ep['object'],mask_selected=sum(bool(r['selected']) for r in rows),confident_masks=len(confident),nonzero_masks=sum(r['area']>0 for r in rows),top12_checked_nonborder_candidate_indices=sampled,S3_nohuman_frames=len(human.get('frames_without_human',[])),S3db_filled_frames=len(body.get('metav3rse_filled_frames',[])),S3db_inference_seconds=body['seconds'],S4b=(w/'s4b'/e/'mhr_params.npz').exists(),depth_scale=(w/'depth_scale'/f'{e}.npz').exists(),S5=(w/'s5'/e/'keyframe.json').exists(),S6=(w/'s6'/e/'poses.npz').exists(),assembled=(w/'submission_v0'/f'{e}.npz').exists()))
 overlay=json.loads((PROJECT/'work/qa/overlay_episode_000000.json').read_text());scored_overlay=[x['frame'] for x in overlay['rows'] if x['frame'] in set(ids.tolist())];assert scored_overlay==[143,215]
 paths=[Path(__file__),src,subset_path,sample_path,PROJECT/'work/submission_v0/episode_000000_object.ply',PROJECT/'code/assemble_episode.py',PROJECT/'code/mhr_single_identity.py',PROJECT/'code/object_mesh_sam3d.py',PROJECT/'code/object_track_fp.py',PROJECT/'code/qa_overlay.py',KIT/'v2dlb/mhr_submission.py',KIT/'metric_code/track_1/CD-H.py']
 report=dict(schema='main-pilot-metadata-readiness-v1',npz=npzmeta,frame_count=T,scored_frames=ids.tolist(),scored_stretches=[[int(ids[s.start]),int(ids[s.stop-1]),s.stop-s.start] for s in ms._stretches(ids)],pack_rows=len(subset),full_pack_rows=len(sample),full_scored_frames=9877,pack_row_roundtrip_max_error=errors,pack_finite=True,pack_ep0_required_row_keys_exact=True,code_commit_urls=subset.code_commit_url.unique().tolist(),packed_mesh=meshq,source_mesh_vertices=len(source_mesh.vertices),source_mesh_faces=len(source_mesh.faces),overlay_frames=[x['frame'] for x in overlay['rows']],overlay_scored_frames=scored_overlay,inventory=inventory,total_video_frames=sum(x['frames'] for x in inventory),remaining_frames=sum(x['frames'] for x in inventory if not x['S4b']),metadata_only_upstream_vertex_finiteness_not_tested=True,main_files_modified=False,source_sha256={str(p):sha(p) for p in paths})
 (out/'METADATA_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['inventory','source_sha256','scored_frames']},indent=2));print('inventory');print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
