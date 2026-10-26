#!/usr/bin/env python3
"""CPU numeric and image-reference comparison; no GT or MAIN writes."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def silhouette(vertices,faces,K,shape):
 z=vertices[:,2];assert (z>.001).all();uv=vertices[:,:2]/z[:,None]*[K[0,0],K[1,1]]+K[:2,2];s=np.zeros(shape,np.uint8)
 for t in uv[faces].round().astype(np.int32):cv2.fillConvexPoly(s,t,1)
 return s.astype(bool)
def main():
 olddir=PROJECT/'work/s5/episode_000000';newdir=ROOT/'main_s5_corrected_v1/episode_000000';old=json.loads((olddir/'keyframe.json').read_text());new=json.loads((newdir/'keyframe.json').read_text());receipt=json.loads((newdir/'RESULT.json').read_text());cpu=json.loads((ROOT/'main_pilot_readiness_v1/HOST_DECODE.json').read_text())
 for record in [receipt,cpu]:
  for p,h in record['source_sha256'].items():assert sha(p)==h,p
 assert sha(newdir/'object_mesh.ply')==receipt['output_mesh_sha256'] and sha(newdir/'keyframe.json')==receipt['keyframe_sha256'];assert receipt['candidate_frames']==[743,711,785];assert old['keyframe']==new['keyframe']==785
 rows=[]
 for a,b in zip(old['candidates'],new['candidates']):
  assert a['frame']==b['frame'] and a['scale']==b['scale'];assert b['layout_requested'] is False and b['layout_post_ran'] is False
  K=np.diag([1536,1152,1.])@np.asarray(b['K_used_normalized']);assert abs(K-np.asarray(receipt['K_source'])).max()<1e-4
  rows.append({'frame':a['frame'],'old_iou':a['iou'],'corrected_iou':b['iou'],'delta_iou':b['iou']-a['iou'],'old_scale':a['scale'],'corrected_scale':b['scale'],'mesh_median_z_delta_m':b['mesh_z_median']-a['mesh_z_median'],'corrected_seconds':b['seconds']})
 o=trimesh.load(olddir/'object_mesh.ply',process=False);n=trimesh.load(newdir/'object_mesh.ply',process=False);R0=np.asarray(old['R_obj2cam']);R1=np.asarray(new['R_obj2cam']);t0=np.asarray(old['t_obj2cam']);t1=np.asarray(new['t_obj2cam']);v0=np.asarray(o.vertices)@R0.T+t0;v1=np.asarray(n.vertices)@R1.T+t1;dist=np.r_[cKDTree(v0).query(v1)[0],cKDTree(v1).query(v0)[0]]
 K=np.asarray(receipt['K_source']);maskrows=json.loads((PROJECT/'work/masks/episode_000000/MASKS.json').read_text())['rows'];m=np.asarray(Image.open(maskrows[785]['mask']))>127;s0=silhouette(v0,np.asarray(o.faces),K,m.shape);s1=silhouette(v1,np.asarray(n.faces),K,m.shape);iou=lambda x,y:float((x&y).sum()/max((x|y).sum(),1))
 recomputed=[iou(s0,m),iou(s1,m)];assert abs(recomputed[0]-old['iou'])<1e-10 and abs(recomputed[1]-new['iou'])<1e-10
 # Scientific projection QA: identical source pixels, camera and mask crop on all comparison panels.
 tiles=[]
 def panel(frame,title,sil=None,color=None):
  im=np.asarray(Image.open(PROJECT/f'work/frames/episode_000000/{frame:06d}.jpg').convert('RGB')).copy();target=np.asarray(Image.open(maskrows[frame]['mask']))>127;y,x=np.where(target);crop=(max(0,int(x.min())-90),max(0,int(y.min())-90),min(im.shape[1],int(x.max())+90),min(im.shape[0],int(y.max())+90));outline=target&~cv2.erode(target.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);im[outline]=[255,30,210]
  if sil is not None:
   edge=sil&~cv2.erode(sil.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);im[edge]=color
  c=Image.fromarray(im).crop(crop);c.thumbnail((500,390));tile=Image.new('RGB',(512,430),'white');tile.paste(c,((512-c.width)//2,20));ImageDraw.Draw(tile).text((7,407),title,fill='black');return tile
 for r in rows:tiles.append(panel(r['frame'],f"RGB frame{r['frame']}; magenta=predicted mask"))
 tiles.append(panel(785,f"Original S5 green; maskIoU={recomputed[0]:.5f}",s0,[0,255,60]));tiles.append(panel(785,f"Known K/layout off cyan; IoU={recomputed[1]:.5f}",s1,[0,230,255]));diff=Image.new('RGB',(512,430),'white');dr=ImageDraw.Draw(diff);dr.text((12,30),'Same3inputs, focal894.6, seed42',fill='black');dr.text((12,60),'Scales exactly unchanged',fill='black');dr.text((12,90),'Best frame remains785',fill='black');dr.text((12,120),f'Max absolute IoU change {max(abs(x["delta_iou"]) for x in rows):.6f}',fill='black');dr.text((12,150),'No measured reconstruction gain',fill='black');dr.text((12,180),'No single-factor causal attribution',fill='black');tiles.append(diff)
 sheet=Image.new('RGB',(1536,860),'white')
 for i,t in enumerate(tiles):sheet.paste(t,((i%3)*512,(i//3)*430))
 image=ROOT/'main_s5_corrected_v1/CANDIDATE_REFERENCE_QA.jpg';sheet.save(image)
 report={'schema':'main-S5-corrected-paired-CPU-comparison-v1','candidates':rows,'best_keyframe':785,'best_translation_delta_m':float(np.linalg.norm(t1-t0)),'best_rotation_delta_degrees':float(np.degrees(Rotation.from_matrix(R1@R0.T).magnitude())),'world_vertex_nearest_neighbor_mean_mm':float(dist.mean()*1000),'world_vertex_nearest_neighbor_p95_mm':float(np.percentile(dist,95)*1000),'paired_mesh_silhouette_iou':iou(s0,s1),'best_recomputed_mask_iou':recomputed,'GT_accuracy_inferred':False,'single_factor_causal_attribution':False,'interpretation':'Near-parity of original and corrected stage1 proposals; original optional layout already failed and fell back. Confirms corrected camera API execution without demonstrating shape/placement improvement.','image':str(image),'image_sha256':sha(image),'source_sha256':{str(p):sha(p) for p in [Path(__file__),olddir/'keyframe.json',olddir/'object_mesh.ply',newdir/'keyframe.json',newdir/'object_mesh.ply',newdir/'RESULT.json',ROOT/'main_pilot_readiness_v1/HOST_DECODE.json']}}
 (ROOT/'main_s5_corrected_v1/COMPARISON_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
