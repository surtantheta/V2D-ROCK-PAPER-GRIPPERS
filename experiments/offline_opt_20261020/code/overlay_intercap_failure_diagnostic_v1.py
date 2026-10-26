#!/usr/bin/env python3
"""SCORER-ONLY RGB overlays; GT geometry is never used for production."""
from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020');sys.path.insert(0,str(P/'code'));import labeled_object_scorer as s
O=P/'validation/intercap_failure_decomposition_v1';diag=json.load(open(O/'RESULT.json'));lookup={r['sequence']:r for r in diag['records']};runtime={r['take']:r for r in json.load(open(P/'labeled_intercap_v1/FP_RUNTIME.json'))['records']};raw={r['sequence']:r for r in json.load(open(P/'constrained_intercap_v1/foundationpose_PREDICTIONS.json'))['sequences']};aligned={r['sequence']:r for r in json.load(open(P/'intercap_results_v1/human_aligned/foundationpose_PREDICTIONS.json'))['sequences']};req=json.load(open(P/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'));refs=json.load(open(P/'benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json'));_,truth,_=s.intercap_references(req,refs,30.)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)

def render(v,f,R,t,scale,K):
 points=scale*v@R.T+t;z=points[:,2];uv=points[:,:2]/np.maximum(z[:,None],1e-9);uv=uv*np.array([K[0,0],K[1,1]])+K[:2,2];output=np.zeros((360,640),np.uint8);uv=np.round(uv).astype(np.int32)
 valid=np.isfinite(points).all(1)&(z>.01)
 for tri in f:
  if valid[tri].all():cv2.fillConvexPoly(output,uv[tri],1)
 return output

def outline(rgb,mask,color):
 edges=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0];cv2.drawContours(rgb,edges,-1,color,2)
 return rgb

reports=[]
for name,indices in [('01/06/Seg_0',[0,17,31]),('08/06/Seg_1',[0,15,31]),('07/02/Seg_0',[0,16,31])]:
 item=runtime[name];disk=item['name'];r=lookup[name];v,f=s.load_surface(raw[name]['mesh_path']);gv,gf=s.load_surface(truth[name]['mesh']);Kp=np.array(r['predicted_K_source']);Kg=np.array(r['native_K_for_scorer_diagnostic']);rows=json.load(open(Path(item['masks'])/'MASKS.json'))['rows'];canvas=Image.new('RGB',(1920,3*420),'white')
 for jj,i in enumerate(indices):
  im=Image.open(item['source_paths'][i]).convert('RGB');W,H=im.size;scaleK=np.diag([640/W,360/H,1.]);kp=scaleK@Kp;kg=scaleK@Kg;rgb=np.array(im.resize((640,360),Image.Resampling.BILINEAR));mask=np.array(Image.open(rows[i]['mask']).resize((640,360),Image.Resampling.NEAREST))>127
  gt=truth[name]['poses'][i];gtmask=render(gv,gf,gt[:3,:3],gt[:3,3],1.,kg);a=raw[name]['frames'][i];b=aligned[name]['frames'][i];pm=render(v,f,np.array(a['R_obj2cam']),np.array(a['t_obj2cam_m']),raw[name]['object_scale'],kp);am=render(v,f,np.array(b['R_obj2cam']),np.array(b['t_obj2cam_m']),aligned[name]['object_scale'],kg)
  left=rgb.copy();left[mask]=(left[mask]*.55+np.array([255,0,0])*.45).astype('uint8');center=outline(outline(rgb.copy(),gtmask,(0,255,0)),pm,(0,210,255));right=outline(outline(rgb.copy(),gtmask,(0,255,0)),am,(255,150,0))
  labels=['Predicted object mask (red)','GT green / production mesh cyan, own predicted K','GT green / human-aligned prediction orange, native K']
  for col,panel in enumerate([left,center,right]):
   canvas.paste(Image.fromarray(panel),(col*640,jj*420));draw=ImageDraw.Draw(canvas);draw.text((col*640+6,jj*420+362),labels[col],fill='black',font=font)
  draw.text((6,jj*420+387),f'{name}: local{i}/native{item["native_frame_numbers"][i]}, CD={r["per_frame_CD_cm"][i]:.1f}cm, center dZ={r["centroid_delta_xyz_cm"][i][2]:+.1f}cm, score={rows[i]["score"]:.3f}',fill='black',font=font)
 dest=O/(disk+'_OVERLAY.jpg');canvas.save(dest,quality=94);reports.append({'sequence':name,'local_frames':indices,'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'GT_usage':'scorer-only RGB visualization, never initializer'})
(O/'OVERLAY_RECEIPT.json').write_text(json.dumps({'records':reports,'columns':['predicted segmentation','production projection with predicted K vs native GT','human-aligned prediction with native K vs native GT'],'GT_occlusion_warning':'GT silhouette is complete mesh projection, not visible GT mask; expected to pass behind hands/person','production_outputs_modified':False},indent=2));print(json.dumps(reports,indent=2))
