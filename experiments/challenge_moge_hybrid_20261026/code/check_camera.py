#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];W,H=1536,1152;K=np.array([[1076.,0,W/2],[0,1076.,H/2],[0,0,1.]])
kn=np.diag([1/W,1/H,1.])@K;km=np.diag([512/W,384/H,1.])@K
uv=np.array([[.5,.5],[W-.5,H-.5],[W/2+.5,H/2+.5]]);z=np.array([2.,3.,4.]);xyz=np.c_[((uv[:,0]-K[0,2])/K[0,0])*z,((uv[:,1]-K[1,2])/K[1,1])*z,z];pp=xyz*np.array([-1.,-1.,1.]);recovered=pp*np.array([-1.,-1.,1.]);projected=(recovered@K.T)[:,:2]/z[:,None];error=float(np.max(abs(projected-uv)));assert error<1e-12
assert np.allclose(np.diag([W,H,1.])@kn,K) and np.allclose(km[:2,2],[256,192])
old=Path('/home/tanuj/V2D_challenge/experiments/post_feedback_20261026/frozen_main_code/assemble_episode.py');new=R/'frozen_main_code/assemble_episode.py';assert old.read_bytes()==new.read_bytes()
print(json.dumps(dict(source_to_model_K=km.tolist(),pixel_P3D_projection_error_px=error,S7_repair_source_bitexact=True,GT_inputs=False)))
