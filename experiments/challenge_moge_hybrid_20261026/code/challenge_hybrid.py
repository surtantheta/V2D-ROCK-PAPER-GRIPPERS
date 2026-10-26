#!/usr/bin/env python3
"""Freeze challenge30 inputs, then map generated objects into submitted MAIN human gauge.

No GT, scorer, provider selection, contact cleanup, GPU, or upload is used here.
Heavy export requires an allocated scheduler job; --self-test-only is CPU-light.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path('/home/tanuj/V2D_challenge')
ROOT = PROJECT / 'experiments/challenge_moge_hybrid_20261026'
KIT = PROJECT / 'data/assets/submission_kit/v2d_submission_kit'
SAMPLE = KIT / 'data/track_1_sample_submission.parquet'
META = PROJECT / 'data/video_to_data_challenge/track_1/meta/episodes.jsonl'
MAIN = PROJECT / 'work/submission_v2'
MOGE = ROOT / 'work/submission'
OUT = ROOT / 'hybrid_v1'
INPUTS = ROOT / 'PACKAGING_INPUTS.json'
PROTOCOL = {
 'schema': 'challenge30-predicted-human-gauge-v1', 'episodes': list(range(30)),
 'expected_full_frames': 16563, 'expected_scored_frames': 9877, 'expected_rows': 740780,
 'MAIN_human': str(MAIN), 'MoGe_object': str(MOGE),
 'fit_frame': 'first actual scored video-frame index from official sample, per episode',
 'fit': 'equal-weight corresponding18439 MHR vertices; positive proper Sim3; no ICP',
 'decoder': 'official TorchScript MHR; pose136+scales68,shape45,expression72zero,correctivesTrue; cm/100*[1,-1,-1]',
 'mapping': 'X_MAIN=s_map*(X_MoGe@R_map.T)+t_map',
 'object_composition': 'scale_out=s_map*scale_in; R_out=R_map@R_in; t_out=s_map*(t_in@R_map.T)+t_map',
 'preserve': 'MAIN human arrays dtype/shape/bytes; generated local mesh bytes and origin',
 'diagnostic_frames': 'first, one-third, two-thirds and last scored frame; only first fits',
 'uniform_policy': True, 'all_30_cases_retained': True, 'GT_or_score_inputs': False,
 'contact_cleanup': False, 'provider_selector': False,
 'experimental': True, 'local_full12_promotion_gate': 'FAILED in previous controlled experiments',
 'limitations': 'Approximate predicted-human registration; different estimated cameras; no exact image reprojection or improved challenge score claim.',
}

def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
 return h.hexdigest()

def write(path, data):
 Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def verify(guards):
 for p, h in guards.items(): assert sha(p) == h, ('source changed', p)

def fit_sim3(x, y):
 x, y = np.asarray(x, float), np.asarray(y, float)
 assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3
 assert np.isfinite(x).all() and np.isfinite(y).all()
 xm, ym = x.mean(0), y.mean(0); xc, yc = x-xm, y-ym
 assert np.linalg.matrix_rank(xc) == 3
 u, v, vt = np.linalg.svd(xc.T@yc)
 sign = np.ones(3); sign[-1] = np.sign(np.linalg.det(vt.T@u.T))
 r = vt.T@np.diag(sign)@u.T
 s = float(v@sign/np.sum(xc*xc)); t = ym-s*(xm@r.T)
 assert s > 0 and np.isfinite(s) and np.isfinite(t).all()
 assert np.allclose(r.T@r, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1., atol=1e-12)
 return s, r, t

def transform(x, s, r, t): return s*(np.asarray(x)@r.T)+t

def compose(scale, rotations, translations, s, r, t):
 return float(scale)*s, np.einsum('ij,njk->nik', r, rotations), transform(translations, s, r, t)

def sample_contract():
 import pandas as pd
 sample = pd.read_parquet(SAMPLE, columns=['row_id'])
 assert len(sample) == 740780 and sample.row_id.is_unique
 parts = sample.row_id.str.split('_', expand=True)
 assert parts.shape[1] == 5 and (parts[0] == 't1').all()
 eps, frames = parts[1].astype(int), parts[2].astype(int)
 meta = {int(j['episode_index']): int(j['length']) for j in map(json.loads, META.read_text().splitlines())}
 assert set(meta) == set(range(30)) and sum(meta.values()) == 16563
 records = []
 for ep in range(30):
  ids = np.sort(frames[(eps == ep)&(frames != 999999)].unique())
  assert len(ids) and ids[0] >= 0 and ids[-1] < meta[ep]
  diagnostic = ids[np.unique([0, len(ids)//3, 2*len(ids)//3, len(ids)-1])]
  records.append(dict(episode=ep, full_frames=meta[ep], scored_frame_indices=ids.tolist(),
                      first_scored_frame=int(ids[0]), diagnostic_frames=diagnostic.tolist()))
 assert sum(len(r['scored_frame_indices']) for r in records) == 9877
 return records

def self_tests():
 rng = np.random.default_rng(20261026); x = rng.normal(size=(997, 3))
 r = Rotation.from_rotvec([.3, -.2, .4]).as_matrix(); s = 1.37; t = np.array([.2, -.8, .7])
 y = transform(x, s, r, t); ss, rr, tt = fit_sim3(x, y)
 fiterr = float(np.max(abs(transform(x, ss, rr, tt)-y))); assert fiterr < 1e-12
 ro = Rotation.from_rotvec([[-.4, .1, .7], [.3, .2, -.5]]).as_matrix()
 to = np.array([[.1, .8, 3.2], [-.2, .3, 4.1]]); scale = .73
 so, rro, tto = compose(scale, ro, to, s, r, t)
 expected = transform(scale*np.einsum('vj,tij->tvi', x, ro)+to[:, None], s, r, t)
 actual = so*np.einsum('vj,tij->tvi', x, rro)+tto[:, None]
 ce = float(np.max(abs(expected-actual))); assert ce < 1e-12
 offset = np.array([.3, .8, -.2]); x2 = x-offset
 so2, rro2, tto2 = compose(scale, ro, to+scale*(offset@ro.transpose(0, 2, 1)), s, r, t)
 oe = float(np.max(abs(so2*np.einsum('vj,tij->tvi', x2, rro2)+tto2[:, None]-actual))); assert oe < 1e-12
 sf, rf, tf = fit_sim3(x, x*np.array([-1, 1, 1])); assert np.linalg.det(rf) > 0
 # Frame zero deliberately disagrees; fitting the scored frame must recover its map.
 sources = np.stack([x+4, x]); targets = np.stack([y-3, y]); first_scored = 1
 ss, rr, tt = fit_sim3(sources[first_scored], targets[first_scored])
 frameerr = float(np.max(abs(transform(x, ss, rr, tt)-y))); assert frameerr < 1e-12
 return dict(known_sim3_error_m=fiterr, composition_error_m=ce, local_origin_invariance_error_m=oe,
             nonzero_first_scored_frame_error_m=frameerr, reflection_proper=True)

def freeze():
 assert not INPUTS.exists(), 'preserve prior frozen inputs'
 records = sample_contract(); guards = {}
 def pin(p): guards[str(p)] = sha(p)
 for p in [SAMPLE, META, PROJECT/'models/mhr/mhr_model.pt', PROJECT/'models/mhr/mhr_faces_lod1.npy',
           KIT/'v2dlb/mhr_metrics.py', KIT/'v2dlb/mhr_submission.py', KIT/'v2dlb/mesh_budget.py',
           KIT/'v2dlb/mesh_common.py', KIT/'tools/pack_reconstruction.py', KIT/'eval_reconstruction.py',
           KIT/'submit/v2d_submit.py', KIT/'config/tracks.json', KIT/'v2dlb/report_schema.py',
           Path(__file__).resolve(), ROOT/'code/pack_challenge_hybrid.py', ROOT/'code/run_hybrid_packaging.py']: pin(p)
 for rec in records:
  p = MAIN/f"episode_{rec['episode']:06d}.npz"; pin(p)
  with np.load(p, allow_pickle=False) as d:
   assert d['pose'].shape == (rec['full_frames'], 136)
   assert d['scales'].shape == (68,) and d['shape'].shape == (45,)
   assert all(np.isfinite(d[k]).all() for k in d.files)
 protocol = ROOT/'HYBRID_PROTOCOL.json'; assert not protocol.exists(); write(protocol, PROTOCOL); pin(protocol)
 write(INPUTS, dict(protocol=PROTOCOL, synthetic_tests=self_tests(), episodes=records, source_sha256=guards))
 print('INPUTS_FROZEN', len(guards), 'sources', len(records), 'cases', flush=True)

def decode(model, data, ids, torch):
 p = np.concatenate([data['pose'][ids], np.repeat(data['scales'][None], len(ids), axis=0)], axis=1).astype(np.float32)
 shape = np.repeat(np.asarray(data['shape'], np.float32).reshape(1, 45), len(ids), axis=0)
 assert p.shape == (len(ids), 204)
 with torch.inference_mode(): v, _ = model(torch.from_numpy(shape), torch.from_numpy(p), torch.zeros(len(ids), 72), True)
 v = v.double().numpy()/100*np.array([1., -1., -1.])
 assert v.shape == (len(ids), 18439, 3) and np.isfinite(v).all() and (v[:, :, 2] > 0).all()
 return v

def residual(x, y):
 d = np.linalg.norm(x-y, axis=-1)*100
 return dict(mean_cm=float(d.mean()), median_cm=float(np.median(d)), p95_cm=float(np.percentile(d, 95)), max_cm=float(d.max()))

def export():
 assert os.environ.get('PBS_JOBID'), 'heavy MHR export requires reserved CPU job'
 assert not OUT.exists(), 'output must be new'
 frozen = json.loads(INPUTS.read_text()); guards = dict(frozen['source_sha256']); verify(guards)
 guards[str(INPUTS)] = sha(INPUTS)
 final = ROOT/'STAGE_B_COMPLETE.json'; producer = json.loads(final.read_text())
 assert producer['stage'] == 'B' and producer['cases'] == 30
 assert producer['frames'] == 16563 and producer['total_scored'] == 9877
 verify(producer['output_sha256']); verify(producer.get('source_sha256', {})); guards.update(producer['output_sha256'])
 guards.update(producer.get('source_sha256', {})); guards[str(final)] = sha(final)
 import torch, trimesh
 torch.set_num_threads(2); torch.set_num_interop_threads(2)
 model = torch.jit.load(str(PROJECT/'models/mhr/mhr_model.pt'), map_location='cpu').eval()
 sys.path.insert(0, str(KIT))
 from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform
 OUT.mkdir(); outsub = OUT/'submission'; outsub.mkdir(); records = []; outputs = {}
 for rec in frozen['episodes']:
  ep = rec['episode']; stem = f'episode_{ep:06d}'; mp = MAIN/f'{stem}.npz'; qp = MOGE/f'{stem}.npz'
  meshsrc = MOGE/f'{stem}_object.ply'; posefile = ROOT/f'work/s6/{stem}/poses.npz'
  for p in [qp, meshsrc, posefile]:
   assert str(p) in producer['output_sha256'], ('unbound producer output', p)
  main, q = dict(np.load(mp, allow_pickle=False)), dict(np.load(qp, allow_pickle=False))
  n = rec['full_frames']; assert set(main) == set(q) == {'pose','scales','shape','object_rotation','object_translation','object_scale'}
  for d in (main, q):
   assert d['pose'].shape == (n,136) and d['object_rotation'].shape == (n,3,3) and d['object_translation'].shape == (n,3)
   assert d['scales'].shape == (68,) and d['shape'].shape == (45,) and d['object_scale'].shape == ()
   assert all(np.isfinite(a).all() for a in d.values()) and float(d['object_scale']) > 0
  ids = np.asarray(rec['diagnostic_frames']); assert ids[0] == rec['first_scored_frame']
  mv, qv = decode(model, main, ids, torch), decode(model, q, ids, torch)
  s, r, t = fit_sim3(qv[0], mv[0]); mapped = transform(qv, s, r, t)
  official = apply_similarity_transform(qv[0], fit_similarity_transform(qv[0], mv[0]))
  parity = float(np.max(abs(official-mapped[0]))); assert parity < 1e-9
  sc, ro, tr = compose(q['object_scale'], q['object_rotation'], q['object_translation'], s, r, t)
  assert np.isfinite(sc) and sc > 0 and np.isfinite(ro).all() and np.isfinite(tr).all()
  assert np.allclose(ro@ro.transpose(0,2,1), np.eye(3), atol=1e-5) and (np.linalg.det(ro)>0).all()
  meshout = outsub/f'{stem}_object.ply'; shutil.copyfile(meshsrc, meshout); assert sha(meshout) == guards[str(meshsrc)]
  mesh = trimesh.load(meshout, process=False); verts = np.asarray(mesh.vertices)
  assert verts.ndim == 2 and verts.shape[1] == 3 and len(verts) and np.isfinite(verts).all()
  minz = float('inf')
  for start in range(0, n, 64):
   z = sc*np.einsum('tj,vj->tv', ro[start:start+64,2], verts)+tr[start:start+64,2,None]
   minz = min(minz, float(z.min()))
  assert minz > .02, ('mesh behind camera; halt without dropping clip', ep, minz)
  output = copy.deepcopy(main); output.update(object_scale=np.float64(sc), object_rotation=ro, object_translation=tr)
  npz = outsub/f'{stem}.npz'; np.savez_compressed(npz, **output)
  with np.load(npz, allow_pickle=False) as check:
   for k in ['pose','scales','shape']:
    assert check[k].dtype == main[k].dtype and check[k].shape == main[k].shape and check[k].tobytes() == main[k].tobytes()
  bodypath = OUT/f'{stem}_PREDICTED_HUMAN_FIT.npz'
  np.savez_compressed(bodypath, frame_index=ids[0], main_first_vertices=mv[0], moge_first_vertices=qv[0])
  for p in [npz, meshout, bodypath]: outputs[str(p)] = sha(p)
  records.append(dict(**rec, predicted_map=dict(scale=s,R=r.tolist(),t=t.tolist(),convention=PROTOCOL['mapping']),
    source_object_scale=float(q['object_scale']), output_object_scale=sc, minimum_full_mesh_camera_z_m=minz,
    official_fit_max_error_m=parity, human_MAIN_bitexact=True, mesh_MoGe_byteexact=True,
    body_residuals=[dict(frame=int(i),before=residual(x,y),after=residual(z,y)) for i,x,y,z in zip(ids,qv,mv,mapped)],
    MAIN_npz=str(mp),MoGe_npz=str(qp),npz=str(npz),mesh=str(meshout)))
  print('EXPORTED', ep, 'fit_frame', ids[0], 'map_scale', s, 'first_fit_cm', residual(mapped[0],mv[0])['mean_cm'], 'min_z', minz, flush=True)
 verify(guards)
 assert len(records) == 30 and sum(x['full_frames'] for x in records) == 16563
 write(OUT/'COMPLETE.json', dict(schema='challenge30-hybrid-export-v1',status='PASS',protocol=PROTOCOL,
    cases=30,frames=16563,total_scored=9877,source_sha256=guards,output_sha256=outputs,records=records,
    synthetic_tests=frozen['synthetic_tests'],GT_used=False,contact_cleanup=False,experimental=True,
    no_submission_performed=True,runtime=dict(torch=torch.__version__,numpy=np.__version__,PBS_JOBID=os.environ['PBS_JOBID'])))

def main():
 ap = argparse.ArgumentParser(description=__doc__); group = ap.add_mutually_exclusive_group(required=True)
 group.add_argument('--freeze-inputs', action='store_true'); group.add_argument('--export', action='store_true'); group.add_argument('--self-test-only', action='store_true')
 args = ap.parse_args()
 if args.self_test_only: print(json.dumps(self_tests(), indent=2))
 elif args.freeze_inputs: freeze()
 else: export()

if __name__ == '__main__': main()
