#!/usr/bin/env python3
"""Official challenge30 packing, exact roundtrip checks, and commit-only finalization.

No Kaggle APIs, authentication, submission or source publication are invoked.
Stage creates a clearly pending payload. Finalize requires a published, public,
exact source snapshot receipt and changes only code_commit_url.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT = Path('/home/tanuj/V2D_challenge')
ROOT = PROJECT/'experiments/challenge_moge_hybrid_20261026'
KIT = PROJECT/'data/assets/submission_kit/v2d_submission_kit'
SAMPLE = KIT/'data/track_1_sample_submission.parquet'
MAIN = PROJECT/'work/submission_v2'
CANDIDATE = ROOT/'hybrid_v1/submission'
NAME = 'track1_moge_hybrid_v1.parquet'
PLACEHOLDER = 'https://github.com/placeholder/placeholder/commit/'+'0'*40
sys.path.insert(0, str(KIT))
from v2dlb import mhr_submission as ms
from v2dlb.mesh_budget import budget_mesh
from v2dlb.mesh_common import _mesh_faces

def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
 return h.hexdigest()

def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def verify(guards):
 for p,h in guards.items(): assert sha(p) == h, ('hash changed', p)

def digest(frame):
 h = hashlib.sha256('\n'.join(frame.row_id.astype(str)).encode())
 h.update(frame[['x','y','z']].to_numpy('float64').tobytes())
 return h.hexdigest()

def official_uploader():
 spec = importlib.util.spec_from_file_location('official_v2d_submit_hybrid', KIT/'submit/v2d_submit.py')
 module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
 return module

def export_receipt():
 p = ROOT/'hybrid_v1/COMPLETE.json'; j = json.loads(p.read_text())
 assert j['status'] == 'PASS' and j['cases'] == 30 and j['frames'] == 16563 and j['total_scored'] == 9877
 assert j['GT_used'] is False and j['contact_cleanup'] is False
 verify(j['source_sha256']); verify(j['output_sha256'])
 return p,j

def validate(path):
 receipt,j = export_receipt()
 config = json.loads((KIT/'config/tracks.json').read_text())
 df = official_uploader().validate_submission(path,'track_1',config)
 expected = pd.read_parquet(SAMPLE, columns=['row_id'])
 assert list(df.columns) == ['row_id','x','y','z','code_commit_url']
 assert df.row_id.equals(expected.row_id) and df.row_id.is_unique and not df.isna().any().any()
 assert len(df) == 740780 and np.isfinite(df[['x','y','z']].to_numpy()).all()
 assert df.code_commit_url.nunique() == 1
 parts = df.row_id.str.split('_',expand=True); keys = df.copy()
 keys['episode_index'] = parts[1].astype(int); keys['frame_index'] = parts[2].astype(int)
 keys['array_name'] = parts[3].astype(int).map(dict(enumerate(ms.array_order())))
 keys['point_index'] = parts[4].astype(int)
 records = {r['episode']:r for r in j['records']}; checks=[]
 for ep,block in keys.groupby('episode_index',sort=True):
  rec=records[int(ep)]; npz=CANDIDATE/f'episode_{ep:06d}.npz'; mesh=CANDIDATE/f'episode_{ep:06d}_object.ply'
  a = ms._to_arrays(block,['x','y','z'])
  frames=np.sort(block.loc[block.frame_index!=999999,'frame_index'].unique())
  assert frames.tolist() == rec['scored_frame_indices']
  with np.load(npz,allow_pickle=False) as d, np.load(MAIN/npz.name,allow_pickle=False) as m:
   assert all(np.isfinite(d[k]).all() for k in d.files)
   assert d['pose'].shape == (rec['full_frames'],136)
   for packed,original,n in [('mhr_pose','pose',136),('mhr_scales','scales',68),('mhr_shape','shape',45)]:
    recovered=ms._unpack(a[packed],n,packed); target=d[original][frames] if original=='pose' else d[original]
    assert np.array_equal(recovered,target),('roundtrip',ep,original)
    assert d[original].dtype==m[original].dtype and d[original].shape==m[original].shape and d[original].tobytes()==m[original].tobytes()
   ms._rotations(a['object_rotation'])
   assert np.array_equal(a['object_rotation'],d['object_rotation'][frames])
   assert np.array_equal(a['object_translation'][:,0],d['object_translation'][frames])
   scale=ms._object_scale(a['object_scale']); assert scale==float(d['object_scale']) and scale>0
   v=a['object_mesh_vertices']; f=_mesh_faces(a['object_mesh_faces'],len(v))
   bv,bf=budget_mesh(mesh,len(f),len(v))
   assert np.array_equal(v,bv) and np.array_equal(f,bf)
   assert len(v)==len(f)==4096 and np.isfinite(v).all()
   minz=float('inf')
   for start in range(0,len(frames),64):
    ix=frames[start:start+64]
    z=scale*np.einsum('tj,vj->tv',d['object_rotation'][ix,2],v)+d['object_translation'][ix,2,None]
    minz=min(minz,float(z.min()))
   assert minz>.02,('packed budget mesh not in front',ep,minz)
   checks.append(dict(episode=int(ep),full_frames=rec['full_frames'],scored_frames=len(frames),
    first_scored_frame=int(frames[0]),mesh_vertices=len(v),mesh_faces=len(f),object_scale=scale,
    min_scored_budget_mesh_z_m=minz,human_MAIN_bitexact=True,scorer_array_roundtrip_exact=True))
 assert len(checks)==30 and sum(r['scored_frames'] for r in checks)==9877
 # Independent human-row comparison against the already packed original MAIN file.
 mainpath=PROJECT/'submissions/upload_prep_20261026/ready/track1_main_v21.parquet'
 original=pd.read_parquet(mainpath,columns=['row_id','x','y','z'])
 assert original.row_id.equals(df.row_id)
 human_rows=parts[3].astype(int).isin([0,1,2])
 assert np.array_equal(original.loc[human_rows,['x','y','z']].to_numpy(),df.loc[human_rows,['x','y','z']].to_numpy())
 verify(j['source_sha256']);verify(j['output_sha256'])
 return dict(status='PASS',file=str(path),file_sha256=sha(path),bytes=path.stat().st_size,
   rows=len(df),cases=30,scored_frames=9877,full_frames=16563,code_commit_url=df.code_commit_url.iloc[0],
   numerical_digest_excluding_commit=digest(df),official_uploader='PASS',scorer_roundtrip='PASS',
   human_MAIN_payload_exact=True,MAIN_parquet_sha256=sha(mainpath),episode_checks=checks,
   export_receipt=str(receipt),export_receipt_sha256=sha(receipt),sample_sha256=sha(SAMPLE))

def stage():
 assert os.environ.get('PBS_JOBID'),'packing/validation requires reserved CPU job'
 export_receipt(); out=ROOT/'pending_commit'; assert not out.exists(),'preserve previous pending payload';out.mkdir()
 path=out/NAME
 command=[sys.executable,str(KIT/'eval_reconstruction.py'),'--episodes',str(CANDIDATE),
          '--sample',str(SAMPLE),'--commit',PLACEHOLDER,'--out',str(path)]
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (ROOT/'pack_moge_hybrid.log').open('w') as log:subprocess.run(command,check=True,cwd=KIT,env=env,stdout=log,stderr=subprocess.STDOUT)
 result=validate(path)
 write(ROOT/'PACKAGING_MANIFEST.json',dict(status='PAYLOAD_VALIDATED_PENDING_PUBLISHED_SOURCE',upload_ready=False,
    experimental=True,local_promotion_gate='FAIL; user explicitly requested experimental challenge candidate',
    no_Kaggle_submission_performed=True,candidate=result,packager_sha256=sha(Path(__file__)),
    frozen_input_sha256=sha(ROOT/'PACKAGING_INPUTS.json')))
 print('PENDING_PAYLOAD_PASS',path,flush=True)

def finalize(url,published,repository):
 match=re.fullmatch(r'https://github\.com/([\w.-]+)/([\w.-]+)/commit/([0-9a-fA-F]{40})/?',url)
 assert match and all('placeholder' not in s.lower() for s in match.groups()[:2]) and match[3]!='0'*40
 head=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
 assert head.lower()==match[3].lower(), 'URL must match actual prepared source HEAD'
 assert not subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True).strip(),'source snapshot must be clean'
 public=json.loads(published.read_text())
 assert public['commit_url'].rstrip('/')==url.rstrip('/') and public['commit_sha'].lower()==head.lower()
 assert public['public'] is True and public['commit_access_without_auth']=='PASS'
 assert public['remote_commit_tree_matches_prepared_snapshot'] is True
 tree=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD^{tree}'],text=True).strip()
 assert tree==public['tree_sha'],'public receipt tree must match prepared HEAD tree'
 # Ensure published HEAD contains the precise new adapter/packager bytes.
 for name in ['challenge_hybrid.py','pack_challenge_hybrid.py','run_hybrid_packaging.py']:
  rel='experiments/challenge_moge_hybrid_20261026/code/'+name
  committed=subprocess.check_output(['git','-C',str(repository),'show',head+':'+rel])
  assert hashlib.sha256(committed).hexdigest()==sha(ROOT/'code'/name),('published adapter differs',name)
 manifest=json.loads((ROOT/'PACKAGING_MANIFEST.json').read_text()); pending=Path(manifest['candidate']['file'])
 assert sha(pending)==manifest['candidate']['file_sha256']
 verify(json.loads((ROOT/'PACKAGING_INPUTS.json').read_text())['source_sha256']);export_receipt()
 df=pd.read_parquet(pending);before=digest(df);assert before==manifest['candidate']['numerical_digest_excluding_commit']
 df['code_commit_url']=url.rstrip('/');tmp=ROOT/'finalizing';ready=ROOT/'ready'
 assert not tmp.exists() and not ready.exists(),'preserve previous release';tmp.mkdir()
 path=tmp/NAME;df.to_parquet(path,index=False);result=validate(path)
 assert result['numerical_digest_excluding_commit']==before
 result['file']=str(ready/NAME)
 write(tmp/'FINAL_MANIFEST.json',dict(status='FINAL_LOCAL_VALIDATION_PASS',experimental=True,
    no_Kaggle_submission_performed=True,local_promotion_gate='FAIL; experimental candidate only',
    candidate=result,published_receipt=str(published),published_receipt_sha256=sha(published),source_commit=head,
    source_tree=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD^{tree}'],text=True).strip()))
 tmp.rename(ready);print('FINAL_PAYLOAD_PASS',ready/NAME,flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True)
 g.add_argument('--stage',action='store_true');g.add_argument('--finalize',action='store_true');g.add_argument('--validate',type=Path)
 ap.add_argument('--commit-url');ap.add_argument('--published-receipt',type=Path);ap.add_argument('--source-repository',type=Path)
 a=ap.parse_args()
 if a.stage:stage()
 elif a.validate:print(json.dumps(validate(a.validate),indent=2))
 else:
  assert a.commit_url and a.published_receipt and a.source_repository
  finalize(a.commit_url,a.published_receipt,a.source_repository)

if __name__=='__main__':main()
