#!/usr/bin/env python3
"""CPU-only reference-MHR decode of three scored main-pilot frames; main read-only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 out=ROOT/'main_pilot_readiness_v1';out.mkdir(exist_ok=True);dest=out/'HOST_DECODE.json';assert not dest.exists()
 packed=PROJECT/'work/pack_test/track1_subset.parquet';npz=PROJECT/'work/submission_v0/episode_000000.npz';bodyfile=PROJECT/'work/s3db/episode_000000/sam3d_body.npz';modelpath=PROJECT/'models/mhr/mhr_model.pt';convert=PROJECT/'work/s4b/episode_000000/CONVERT.json';kit=PROJECT/'data/assets/submission_kit/v2d_submission_kit'
 paths=[Path(__file__),packed,npz,bodyfile,modelpath,convert,kit/'tools/track1/mesh_to_mhr_params.py',kit/'metric_code/track_1/CD-H.py'];sources={str(p):sha(p) for p in paths}
 d=np.load(npz,allow_pickle=False);table=pd.read_parquet(packed);parts=table.row_id.str.split('_',expand=True);frames=np.sort(parts.loc[parts[3].astype(int)==0,2].astype(int).unique());ids=frames[[0,len(frames)//2,-1]];assert ids[0]==90 and ids[-1]==699
 params=np.concatenate([d['pose'][ids],np.repeat(d['scales'][None],len(ids),0)],axis=1).astype(np.float32);shape=np.repeat(d['shape'][None],len(ids),0).astype(np.float32)
 assert params.shape==(3,204) and shape.shape==(3,45)
 model=torch.jit.load(str(modelpath),map_location='cpu').eval();t0=time.monotonic()
 with torch.inference_mode():
  v,j=model(torch.from_numpy(shape),torch.from_numpy(params),torch.zeros(3,72),True)
  shifted=params[:1].copy();shifted[0,0]+=1.
  vp,_=model(torch.from_numpy(shape[:1]),torch.from_numpy(shifted),torch.zeros(1,72),True)
 V=v.detach().cpu().numpy()*np.array([1.,-1.,-1.])/100.;J=j.detach().cpu().numpy()[...,:3]*np.array([1.,-1.,-1.])/100.;Vp=vp.detach().cpu().numpy()*np.array([1.,-1.,-1.])/100.
 assert V.shape==(3,18439,3) and J.shape==(3,127,3) and np.isfinite(V).all() and np.isfinite(J).all() and (V[...,2]>0).all()
 translation_response=Vp[0]-V[0];unit_error=float(abs(translation_response-[.1,0,0]).max());assert unit_error<1e-5
 body=np.load(bodyfile,allow_pickle=False)['vertices_cam'];sigma=float(json.loads(convert.read_text())['smooth_sigma']);assert sigma==1.
 radius=int(np.ceil(3*sigma));offset=np.arange(-radius,radius+1);weights=np.exp(-.5*(offset/sigma)**2);weights/=weights.sum();target=np.stack([sum(float(w)*body[int(np.clip(i+o,0,len(body)-1))].astype(np.float64) for o,w in zip(offset,weights)) for i in ids]);err=np.linalg.norm(V-target,axis=-1)*1000
 # This validates units/layout against the official TorchScript asset and host frame recipe.
 # Hosted NumPy decoder asset/512 hand-role indices are omitted from participant kit.
 report={'schema':'main-pilot-reference-MHR-host-frame-decode-v1','cpu_only':True,'scored_frames':ids.tolist(),'pose_shape':[790,136],'shared_scales':68,'shared_shape':45,'decoded_vertices_shape':list(V.shape),'decoded_joints_shape':list(J.shape),'native_translation_parameter_plus1_response_m':[float(x) for x in translation_response.mean(0)],'translation_parameter_response_max_error_m':unit_error,'mean_vertex_error_vs_sigma1_smoothed_SAM3DBody_mm':err.mean(1).tolist(),'perframe_p99_vertex_error_mm':np.percentile(err,99,axis=1).tolist(),'human_depth_minmax_m':[[float(x[:,2].min()),float(x[:,2].max())] for x in V],'host_frame_recipe':'TorchScript cm -> componentwise[1,-1,-1]/100, params=[pose136,scales68], shape45, expression72zero, correctivesTrue','exact_hosted_numpy_asset_parity_tested':False,'hosted_512_hand_role_indices_available':False,'PEN_quality_tested':False,'GT_inputs_read':False,'main_files_modified':False,'decode_seconds':time.monotonic()-t0,'source_sha256':sources}
 assert err.mean(1).max()<10.,'reference decode mismatch >10mm versus known converter targets'
 dest.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
