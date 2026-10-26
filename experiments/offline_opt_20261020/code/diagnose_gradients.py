#!/usr/bin/env python3
"""Compare the actual torus-loss derivatives to finite differences before more tuning."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import nvdiffrast.torch as dr
import silhouette_pilot_v3 as pilot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('output exists')
    ctx = dr.RasterizeCudaContext()
    v, f = pilot.torus_mesh(.45, .015)
    v, f = torch.tensor(v, device='cuda'), torch.tensor(f, device='cuda')
    k = torch.tensor([[[250., 0, 64], [0, 250., 64], [0, 0, 1]]], device='cuda')
    gt = torch.tensor([.7, .2, .3, .08, .1, math.log(3.)], device='cuda')
    start = gt + torch.tensor([.06, -.04, .05, .03, -.02, math.log(3.05/3)], device='cuda')

    def rendered(p):
        rotation = pilot.rotation_xyz(p[:3][None])
        translation = torch.stack([p[3], p[4], p[5].exp()])
        return pilot.render(ctx, v[None] @ rotation.transpose(-1, -2) + translation[None, None], f, k, [128, 128])

    target = rendered(gt).detach()
    distance = pilot.make_distance(target)
    def loss(p):
        return pilot.loss_silhouette(rendered(p), target, distance)

    p = start.clone().requires_grad_()
    value = loss(p)
    value.backward()
    analytic = p.grad.detach().cpu().numpy()
    rows = []
    for epsilon in [1e-5, 1e-4, 1e-3, 1e-2]:
        numeric = []
        for j in range(6):
            offset = torch.zeros_like(start); offset[j] = epsilon
            numeric.append(float((loss(start+offset) - loss(start-offset)) / (2 * epsilon)))
        numeric = np.array(numeric)
        rows.append({'epsilon': epsilon, 'numeric': numeric.tolist(),
                     'cosine': float(np.dot(analytic,numeric)/(np.linalg.norm(analytic)*np.linalg.norm(numeric))),
                     'sign_agreement': (np.sign(analytic)==np.sign(numeric)).tolist()})
    candidates = []
    for direction_name, direction in [('autograd', analytic), ('finite_difference',np.array(rows[1]['numeric']))]:
        direction = torch.tensor(direction, device='cuda', dtype=torch.float32)
        direction = direction / direction.norm()
        for step in [.0001,.001,.005,.01]:
            candidate = start - step*direction
            candidates.append({'direction':direction_name,'step':step,'loss':float(loss(candidate)),
                               'iou':pilot.ious(rendered(candidate),target)[0]})
    report = {'scope':'synthetic actual-objective derivative diagnostic only',
              'parameters':['rx','ry','rz','tx','ty','log_z'], 'initial_parameters':start.cpu().tolist(),
              'ground_truth_parameters':gt.cpu().tolist(), 'initial_loss':float(value),
              'ground_truth_loss':float(loss(gt)), 'initial_iou':pilot.ious(rendered(start),target)[0],
              'autograd':analytic.tolist(),'finite_differences':rows,'descent_probes':candidates,
              'source_sha256':{str(Path(__file__).resolve()):pilot.digest(__file__),
                               str(Path(pilot.__file__).resolve()):pilot.digest(pilot.__file__)}}
    args.out.mkdir(parents=True)
    (args.out/'GRADIENTS.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
