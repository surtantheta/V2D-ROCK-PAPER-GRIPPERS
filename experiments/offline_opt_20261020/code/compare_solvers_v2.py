#!/usr/bin/env python3
"""Bounded synthetic test: supersampled autodiff versus derivative-free pose fitting."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as functional
import nvdiffrast.torch as dr
import silhouette_pilot_v3 as pilot


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('output exists')
    ctx=dr.RasterizeCudaContext();device='cuda'
    vv,ff=pilot.torus_mesh(.45,.015)
    vertices=torch.tensor(vv,device=device);faces=torch.tensor(ff,device=device)
    k=torch.tensor([[[250.,0,64],[0,250.,64],[0,0,1]]],device=device)
    gt=np.array([.7,.2,.3,.08,.1,math.log(3.)],np.float32)
    start=gt+np.array([.06,-.04,.05,.03,-.02,math.log(3.05/3)],np.float32)
    base_render=pilot.render
    results=[]
    for factor in [1,4]:
        def renderer(context,vc,tri,kk,size):
            if factor==1:return base_render(context,vc,tri,kk,size)
            upscale=kk.clone();upscale[:,:2,:]*=factor
            image=base_render(context,vc,tri,upscale,[size[0]*factor,size[1]*factor])
            return functional.avg_pool2d(image[:,None],factor)[:,0]
        pilot.render=renderer
        def rendered(p):
            rotation=pilot.rotation_xyz(p[:3][None]);translation=torch.stack([p[3],p[4],p[5].exp()])
            return renderer(ctx,vertices[None]@rotation.transpose(-1,-2)+translation[None,None],faces,k,[128,128])
        target=rendered(torch.tensor(gt,device=device)).detach();distance=pilot.make_distance(target)
        t0=time.time()
        best,history=pilot.fit(ctx,vertices,faces,target,k,start[:3][None],start[3:][None],steps=300)
        results.append({'renderer_supersample':factor,'solver':'annealed_adam','iou_before':pilot.ious(rendered(torch.tensor(start,device=device)),target)[0],
                        'iou_after':pilot.ious(best[3],target)[0],'loss':best[0],'seconds':time.time()-t0,'history':history})
        print(json.dumps({key:val for key,val in results[-1].items() if key!='history'}),flush=True)
        if factor==1:
            evaluations=0
            def objective(parameters):
                nonlocal evaluations
                evaluations+=1
                with torch.no_grad():
                    pp=torch.tensor(parameters,device=device,dtype=torch.float32)
                    return float(pilot.loss_silhouette(rendered(pp),target,distance))
            bounds=[(float(value-delta),float(value+delta)) for value,delta in zip(start,[.25,.25,.25,.15,.15,.2])]
            t0=time.time();res=minimize(objective,start.astype(np.float64),method='Powell',bounds=bounds,
                                       options={'maxiter':60,'maxfev':4000,'xtol':1e-5,'ftol':1e-6})
            pred=rendered(torch.tensor(res.x,device=device,dtype=torch.float32))
            estimated_translation=np.array([res.x[3],res.x[4],np.exp(res.x[5])])
            truth_translation=np.array([gt[3],gt[4],np.exp(gt[5])])
            results.append({'renderer_supersample':1,'solver':'bounded_powell','iou_before':pilot.ious(rendered(torch.tensor(start,device=device)),target)[0],
                            'iou_after':pilot.ious(pred,target)[0],'loss':float(res.fun),'evaluations':evaluations,
                            'seconds':time.time()-t0,'converged':bool(res.success),'parameters':res.x.tolist(),
                            'translation_error_m':float(np.linalg.norm(estimated_translation-truth_translation))})
            print(json.dumps(results[-1]),flush=True)
    pilot.render=base_render
    args.out.mkdir(parents=True)
    report={'scope':'synthetic solver comparison; no V2D data or challenge metrics','same_target_and_start_parameters':True,
            'gt_parameters':gt.tolist(),'initial_parameters':start.tolist(),'results':results,
            'source_sha256':{str(Path(__file__).resolve()):pilot.digest(__file__),str(Path(pilot.__file__).resolve()):pilot.digest(pilot.__file__)}}
    (args.out/'SOLVERS.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
