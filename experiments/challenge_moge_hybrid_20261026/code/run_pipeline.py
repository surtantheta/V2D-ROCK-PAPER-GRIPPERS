#!/usr/bin/env python3
"""Two disjoint full-MoGe producer shards and rolling sm80 FP/S7 consumer."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];W=R/'work';C=R/'frozen_main_code';P=Path('/home/tanuj/V2D_challenge')
PY='/home/tanuj/.conda/envs/v2d/bin/python';SAM='/home/tanuj/.conda/envs/sam3d-objects/bin/python';PREP='/home/tanuj/.conda/envs/cari4d/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def call(argv,cwd=None,extra=None):
 env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');env.update(extra or {});print('COMMAND',argv,flush=True);subprocess.run(argv,cwd=cwd,env=env,check=True)
def verify(paths):
 for p,h in paths.items():assert sha(p)==h,('source changed',p)
def output_hashes(paths):return {str(p):sha(p) for root in paths for p in ([root] if root.is_file() else sorted(root.rglob('*'))) if p.is_file() and not p.name.endswith('.tmp')}
def inject(ep,n):
 e=f'episode_{ep:06d}';run=json.loads((R/'depth'/e/'RUN.json').read_text());target=W/'s3_human'/e;meta=json.loads((target/'RUN.json').read_text());f=float(run['K_fixed_source'][0][0]);assert abs(meta['focal_source_px']-f)<1e-6
 for i in range(n):
  src=R/'depth'/e/f'{i:06d}.npz';assert sha(src)==run['output_sha256'][str(src)];z=np.load(src);path=target/f'{i:06d}.npz'
  with np.load(path) as original:d={k:original[k] for k in original.files}
  assert 'depth_metav3rse'not in d and d['depth'].shape==(384,512);d['depth_metav3rse']=d['depth'].copy();d['depth']=z['depth'].astype(np.float32);np.savez_compressed(path,**d)
 meta.update(depth_source='fresh RGB MoGe, float16cache->float32 modelgrid withoutcrop',depth_run=str(R/'depth'/e/'RUN.json'),depth_run_sha256=sha(R/'depth'/e/'RUN.json'),known_source_K=run['K_fixed_source']);write(target/'RUN.json',meta)
def verify_case_inputs(spec,case):
 e=f"episode_{case['episode']:06d}"
 paths=[Path(case['source_frames'])/'FRAMES.json',Path(case['source_masks'])/'MASKS.json']
 paths.extend((W/'frames'/e/f'{i:06d}.jpg').resolve() for i in range(case['frames']))
 paths.extend(Path(x['mask']) for x in json.loads(Path(case['mask_manifest']).read_text())['rows'])
 for p in paths:
  assert sha(p)==spec['source_sha256'][str(p)],('case input changed',p)
def human_and_object(case):
 ep=case['episode'];n=case['frames'];e=f'episode_{ep:06d}';frames=W/'frames'/e;masks=W/'masks'/e;human=W/'s3_human'/e;ds=W/'depth_scale'/f'{e}.npz';ds.parent.mkdir(exist_ok=True)
 call([SAM,str(R/'code/moge_depth.py'),'--episode',str(ep)],cwd='/rhome/tanuj/projects/sam-3d-objects',extra={'CONDA_PREFIX':'/home/tanuj/.conda/envs/sam3d-objects','CUDA_HOME':'/home/tanuj/.conda/envs/sam3d-objects'})
 depth=json.loads((R/'depth'/e/'RUN.json').read_text());f=float(depth['K_fixed_source'][0][0]);modelK=np.diag([512/case['source_wh'][0],384/case['source_wh'][1],1.])@np.array(depth['K_fixed_source'])
 call(['/home/tanuj/.conda/envs/human3r_new/bin/python',str(C/'metav3rse_human_infer.py'),'--frames',str(frames),'--ckpt','/rhome/tanuj/unicon3rpp_runs/U3PP-Gates/M10MeshQueryD1/spatial-gtcrop25/checkpoint-25.pth','--out',str(human),'--mode','full','--focal',str(f),'--window','16'],cwd='/home/tanuj/UniCon3Rpp-push-b3c6624',extra={'PYTHONPATH':'/home/tanuj/UniCon3Rpp-push-b3c6624/src:/home/tanuj/UniCon3Rpp-push-b3c6624/src/croco'})
 inject(ep,n)
 call(['/home/tanuj/envs/sam3dbody/bin/python',str(C/'sam3d_body_infer.py'),'--frames',str(frames),'--human',str(human),'--out',str(W/'s3db'/e),'--focal',str(f)])
 call([PY,str(C/'mhr_single_identity.py'),'--sam3d-body',str(W/'s3db'/e/'sam3d_body.npz'),'--out',str(W/'s4b'/e),'--smooth-sigma-xy','2.0','--smooth-sigma-z','5.0'])
 call([PY,str(C/'depth_scale.py'),'--human',str(human),'--s3db',str(W/'s3db'/e),'--masks',str(masks),'--out',str(ds)])
 call([SAM,str(C/'object_mesh_sam3d.py'),'--frames',str(frames),'--masks',str(masks),'--human',str(human),'--focal',str(f),'--out',str(W/'s5'/e),'--depth-scale',str(ds)],cwd='/rhome/tanuj/projects/sam-3d-objects',extra={'CONDA_PREFIX':'/home/tanuj/.conda/envs/sam3d-objects','CUDA_HOME':'/home/tanuj/.conda/envs/sam3d-objects','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
 assert np.load(W/'s3db'/e/'sam3d_body.npz')['vertices_cam'].shape[0]==n;assert np.load(W/'s4b'/e/'mhr_params.npz')['pose'].shape==(n,136)
 return dict(episode=ep,frames=n,source_wh=case['source_wh'],K_source=depth['K_fixed_source'],K_model=modelK.tolist(),focal_source_px=f,status='READY',GT_inputs=False,output_sha256=output_hashes([human,W/'s3db'/e,W/'s4b'/e,ds,ds.with_suffix('.json'),W/'s5'/e,R/'depth'/e/'RUN.json']))
def consume(case,receipt):
 ep=case['episode'];e=f'episode_{ep:06d}';verify(receipt['output_sha256']);frames=W/'frames'/e;masks=W/'masks'/e;human=W/'s3_human'/e;ds=W/'depth_scale'/f'{e}.npz'
 call([PREP,str(C/'object_track_fp.py'),'--frames',str(frames),'--masks',str(masks),'--human',str(human),'--object',str(W/'s5'/e),'--out',str(W/'s6'/e),'--depth-scale',str(ds)],cwd='/home/tanuj/projects/CARI4D')
 tr=np.load(W/'s6'/e/'poses.npz');assert len(tr['T_obj2cam'])==case['frames'] and np.isfinite(tr['T_obj2cam']).all() and np.max(abs(tr['K_model']-np.array(receipt['K_model'])))<1e-4
 call([PY,str(C/'assemble_episode.py'),'--episode',str(ep),'--human',str(W/'s4b'/e),'--object',str(W/'s5'/e),'--track',str(W/'s6'/e),'--masks',str(masks),'--s3db',str(W/'s3db'/e),'--s3-human',str(human),'--depth-scale',str(ds),'--out',str(W/'submission')])
 out=W/'submission'/f'{e}.npz';data=np.load(out);assert len(data['pose'])==case['frames'] and all(np.isfinite(data[k]).all() for k in data.files);assert float(data['object_scale'])>0
 return dict(episode=ep,frames=case['frames'],status='READY',K_model=receipt['K_model'],S7_repairs_preserved=True,input_A_receipt_sha256=sha(R/'receipts'/f'A_READY_{ep:06d}.json'),output_sha256=output_hashes([W/'s6'/e,out,W/'submission'/f'{e}_object.ply',W/'submission'/f'ASSEMBLE_{ep:06d}.json']))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A0','A1','B'],required=True);args=ap.parse_args();assert os.environ.get('PBS_JOBID');started=time.monotonic();spec=json.loads((R/'INPUTS.json').read_text());freeze=json.loads((R/'SOURCE_FREEZE.json').read_text());verify(freeze['source_sha256']);static_sources={p:h for p,h in freeze['source_sha256'].items() if p not in spec['source_sha256']};models=json.loads((R/'MODEL_EXPECTED.json').read_text());verify(models['A' if args.stage.startswith('A') else 'B']);cases=spec['cases'];print('SCHEDULER',os.environ.get('PBS_JOBID'),args.stage,flush=True);call(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader']);failures=[];done=[]
 if args.stage.startswith('A'):
  selected=[c for c in cases if c['stage_shard']==args.stage]
  for c in selected:
   try:
    assert shutil.disk_usage(R).free>8*1024**3,'less than8GBdiskfree';ep=c['episode'];print('EP_BEGIN',args.stage,ep,'elapsed',time.monotonic()-started,'diskfreeGB',shutil.disk_usage(R).free/1024**3,flush=True)
    verify_case_inputs(spec,c)
    rec=human_and_object(c);verify_case_inputs(spec,c);verify(static_sources);write(R/'receipts'/f'A_READY_{ep:06d}.json',rec);done.append(ep);print('EP_READY',ep,'elapsed',time.monotonic()-started,flush=True)
   except Exception as exc:
    rec=dict(episode=c['episode'],stage=args.stage,error=repr(exc));failures.append(rec);write(R/'receipts'/f'A_FAILED_{c["episode"]:06d}.json',rec);print('EP_FAILED',rec,flush=True)
  verify(freeze['source_sha256']);write(R/f'STAGE_{args.stage}_COMPLETE.json',dict(stage=args.stage,cases=len(done),episodes=done,failures=failures,seconds=time.monotonic()-started))
 else:
  remaining={c['episode']:c for c in cases};deadline=time.monotonic()+7200;last_log=0
  while remaining:
   ready=[ep for ep in remaining if (R/'receipts'/f'A_READY_{ep:06d}.json').exists()];failed=[ep for ep in remaining if (R/'receipts'/f'A_FAILED_{ep:06d}.json').exists()]
   for ep in failed:failures.append(dict(episode=ep,error='producer failed'));remaining.pop(ep)
   if not ready:
    assert time.monotonic()<deadline,'no producerprogress for2hours'
    if time.monotonic()-last_log>60:print('WAIT_PRODUCERS',len(remaining),'elapsed',time.monotonic()-started,flush=True);last_log=time.monotonic()
    time.sleep(10);continue
   ep=ready[0];c=remaining.pop(ep)
   try:
    verify_case_inputs(spec,c);rec=consume(c,json.loads((R/'receipts'/f'A_READY_{ep:06d}.json').read_text()));verify_case_inputs(spec,c);verify(static_sources);write(R/'receipts'/f'B_READY_{ep:06d}.json',rec);done.append(ep);print('B_READY',ep,'elapsed',time.monotonic()-started,flush=True)
   except Exception as exc:
    rec=dict(episode=ep,error=repr(exc));failures.append(rec);write(R/'receipts'/f'B_FAILED_{ep:06d}.json',rec);print('B_FAILED',rec,flush=True)
   deadline=time.monotonic()+7200
  outputs={}
  for ep in done:outputs.update(json.loads((R/'receipts'/f'B_READY_{ep:06d}.json').read_text())['output_sha256'])
  verify(freeze['source_sha256']);write(R/'STAGE_B_COMPLETE.json',dict(stage='B',cases=len(done),frames=sum(c['frames'] for c in cases if c['episode']in done),total_scored=9877,episodes=done,failures=failures,output_sha256=outputs,source_freeze_sha256=sha(R/'SOURCE_FREEZE.json'),seconds=time.monotonic()-started))
 if failures:raise RuntimeError(f'{args.stage} failures retained: {failures}')
if __name__=='__main__':main()
