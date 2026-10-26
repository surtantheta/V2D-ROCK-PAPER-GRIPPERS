#!/usr/bin/env python3
"""Use the actual submission mesh budget while preserving the source mesh origin."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('output exists')
    project=Path('/home/tanuj/V2D_challenge')
    kit=project/'data/assets/submission_kit/v2d_submission_kit'
    sys.path.insert(0,str(kit))
    from v2dlb.mesh_budget import budget_mesh
    src=project/'work/s5/episode_000000/object_mesh.ply'
    prior=project/'experiments/offline_opt_20261020/runs/torus_v6_conic/RESULT.json'
    mesh=trimesh.load(src,process=False)
    vertices=np.asarray(mesh.vertices)
    center=vertices.mean(0)
    values,axes=np.linalg.eigh(np.cov((vertices-center).T))
    # Largest/second-largest PCA axes span the ring, smallest is the normal.
    basis=axes[:,[2,1,0]]
    if np.linalg.det(basis)<0:basis[:,2]*=-1
    local=(vertices-center)@basis
    planar_diameter=float(np.mean(np.ptp(local[:,:2],axis=0)))
    control=json.loads(prior.read_text())
    # Choose the positive solution using its fitting subset, never conditional IoU.
    ratio=float(control['results'][0]['tube_radius_m_prior_scaled']/control['fixed_major_radius_prior_m'])
    radius=planar_diameter/(2*(1+ratio));minor=radius*ratio
    u,v=np.meshgrid(np.arange(128)*2*np.pi/128,np.arange(12)*2*np.pi/12,indexing='ij')
    ring=np.stack([(radius+minor*np.cos(v))*np.cos(u),(radius+minor*np.cos(v))*np.sin(u),minor*np.sin(v)],-1).reshape(-1,3)
    ring=ring@basis.T+center
    faces=[]
    for i in range(128):
        for j in range(12):
            a=i*12+j;b=((i+1)%128)*12+j;c=((i+1)%128)*12+(j+1)%12;d=i*12+(j+1)%12
            faces.extend([[a,b,c],[a,c,d]])
    args.out.mkdir(parents=True)
    ring_path=args.out/'torus_matched_source.ply'
    trimesh.Trimesh(ring,np.array(faces),process=False).export(ring_path)
    manifest={'schema':'v2d-budget-comparison-assets-v1',
              'input_contract':'predicted geometry only; torus dimensions/orientation/center matched to SAM3D proposal, tube ratio from fitting subset',
              'source_sha256':{str(p):sha(p) for p in [src,prior,Path(__file__),kit/'v2dlb/mesh_budget.py']},
              'metric_scale_recovered':False,'center':center.tolist(),'pca_basis_columns':basis.tolist(),
              'torus_major_radius_m_from_generated_prior':radius,'torus_minor_radius_m':minor,'assets':{}}
    np.random.seed(0)
    for name,path in [('generated',src),('torus_matched',ring_path)]:
        vv,ff=budget_mesh(path,faces=4096,vertices=4096)
        artifact=args.out/f'{name}_4096.npz'
        np.savez_compressed(artifact,vertices=vv.astype(np.float32),faces=ff.astype(np.int32))
        tri=vv[ff]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        manifest['assets'][name]={'array_path':str(artifact),'sha256':sha(artifact),'vertices':len(vv),'faces':len(ff),
                                 'nondegenerate_faces':int((area>1e-12).sum()),'extent_m':np.ptp(vv,axis=0).tolist(),
                                 'origin_changed':False}
        assert len(vv)==4096 and len(ff)==4096 and np.isfinite(vv).all()
    (args.out/'ASSETS.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
