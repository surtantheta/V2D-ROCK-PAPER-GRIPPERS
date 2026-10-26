#!/usr/bin/env python3
"""Fresh challenge RGB MoGe depth, same official infer path as SAM3D wrapper."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
CHECKPOINT=Path('/home/tanuj/.cache/huggingface/hub/models--Ruicheng--moge-vitl/snapshots/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/model.pt')
EXPECTED='da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--episode',type=int,required=True);args=ap.parse_args();assert os.environ.get('PBS_JOBID')
 spec=json.loads((R/'INPUTS.json').read_text());case=spec['cases'][args.episode];e=f'episode_{args.episode:06d}';out=R/'depth'/e;out.mkdir(parents=True,exist_ok=True);producer=sha(Path(__file__));assert sha(CHECKPOINT)==EXPECTED
 import torch
 from moge.model.v1 import MoGeModel
 model=MoGeModel.from_pretrained(str(CHECKPOINT)).to('cuda').eval();torch.set_num_threads(2);intrinsics=[];outputs={};start=time.monotonic();wh=case['source_wh']
 for i in range(case['frames']):
  src=R/'work/frames'/e/f'{i:06d}.jpg';rgbsha=sha(src);assert rgbsha==case['frame_rgb_sha256'][i];p=out/f'{i:06d}.npz'
  if p.exists():
   z=np.load(p);assert str(z['rgb_sha256'])==rgbsha and str(z['model_sha256'])==EXPECTED and str(z['producer_sha256'])==producer;K=z['K_source'];d=z['depth'];assert d.shape==(384,512)
  else:
   im=np.asarray(Image.open(src).convert('RGB'));assert [im.shape[1],im.shape[0]]==wh;image=torch.from_numpy((im.astype(np.float32)/255).transpose(2,0,1).copy()).to('cuda')
   with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):result=model.infer(image,force_projection=False)
   points=result['points'].detach().float().cpu().numpy();kn=result['intrinsics'].detach().float().cpu().numpy().reshape(3,3);assert points.shape==(*im.shape[:2],3)
   d=np.asarray(Image.fromarray(points[...,2]).resize((512,384),Image.Resampling.BILINEAR));d=np.where(np.isfinite(d)&(d>0),d,0.);assert (d>0).mean()>.5
   K=np.diag([wh[0],wh[1],1.])@kn;np.savez_compressed(p,depth=d.astype(np.float16),K_source=K,rgb_sha256=rgbsha,model_sha256=EXPECTED,producer_sha256=producer);del result,points,image
  assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[:2,2],np.array(wh)/2,atol=.01)
  intrinsics.append(K);outputs[str(p)]=sha(p)
  if i%32==0:print('MOGE',args.episode,i,case['frames'],'sec',round(time.monotonic()-start,1),flush=True)
 K=np.median(np.asarray(intrinsics),axis=0);K[2]=[0,0,1];assert K[0,0]>0 and np.isfinite(K).all();assert abs(K[0,0]-K[1,1])<.01 and np.allclose(K[:2,2],np.array(wh)/2,atol=.01)
 run=dict(frames=case['frames'],source_wh=wh,model_wh=[512,384],K_fixed_source=K.tolist(),raw_K_source_per_frame=np.asarray(intrinsics).tolist(),source='official MoGeModel.infer(force_projection=False); samecheckpoint/FP16autocast/RGB path as SAM3D internal depth_model',P3D_depth_equivalence='camera_to_pytorch3d_camera flips only X/Y; Z identical; no pointmap clip configured',metric_scale_recovered=False,external_human_calibration=False,GT_inputs=False,model_sha256=EXPECTED,producer_sha256=producer,input_rgb_sha256=case['frame_rgb_sha256'],output_sha256=outputs,seconds=time.monotonic()-start,torch=torch.__version__)
 (out/'RUN.json').write_text(json.dumps(run,indent=2)+'\n');print('MOGE_COMPLETE',args.episode,run['seconds'],K.tolist(),flush=True)
if __name__=='__main__':main()
