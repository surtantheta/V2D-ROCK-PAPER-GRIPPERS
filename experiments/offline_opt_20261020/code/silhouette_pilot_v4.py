#!/usr/bin/env python3
"""Fixed-size object-pose/shape diagnostic, isolated from the V2D submission path."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import nvdiffrast.torch as dr


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def torus_mesh(major=0.45, minor=0.0075, nu=128, nv=12):
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing='ij')
    vertices = np.stack([(major + minor * np.cos(v)) * np.cos(u),
                         (major + minor * np.cos(v)) * np.sin(u), minor * np.sin(v)], -1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            faces.extend([[a, b, c], [a, c, d]])
    return vertices.astype(np.float32), np.asarray(faces, np.int32)


def rotation_xyz(angles):
    x, y, z = angles.unbind(-1)
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    cx, sx, cy, sy, cz, sz = torch.cos(x), torch.sin(x), torch.cos(y), torch.sin(y), torch.cos(z), torch.sin(z)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(-1, 3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(-1, 3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    return rz @ ry @ rx


def render(context, vertices_camera, faces, intrinsics, size):
    """OpenCV +x right,+y down,+z forward -> raw raster arrays in image row order.

    The clip y axis is deliberately increasing with image row. nvdiffrast's raw
    bottom-up array convention then produces a directly indexable top-down image.
    No OpenCV-to-OpenGL mesh rotation or array flip is applied elsewhere.
    """
    h, w = size
    x, y, z = vertices_camera.unbind(-1)
    fx, fy = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]
    cx, cy = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]
    near, far = 0.01, 100.0
    clip = torch.stack([2 * fx / w * x + (2 * cx / w - 1) * z,
                        2 * fy / h * y + (2 * cy / h - 1) * z,
                        (far + near) / (far - near) * z - 2 * far * near / (far - near), z], -1)
    rast, _ = dr.rasterize(context, clip.contiguous(), faces.contiguous(), resolution=[h, w])
    hard = (rast[..., 3:] > 0).float()
    return dr.antialias(hard, rast, clip.contiguous(), faces.contiguous())[..., 0]


def loss_silhouette(pred, target, distance):
    mass = target.sum((1, 2)).clamp_min(1)
    coverage = ((pred - target).square().sum((1, 2)) / mass).mean()
    boundary = ((pred * distance).sum((1, 2)) / mass).mean()
    return coverage + 0.2 * boundary


def make_distance(target):
    return torch.as_tensor(np.stack([distance_transform_edt(m < 0.5) / 10 for m in target.cpu().numpy()]),
                           device=target.device, dtype=torch.float32)


def load_frames(project, episode, frames, focal, size, sources):
    images, targets, intrinsics, initial, metadata = [], [], [], [], []
    name = f'episode_{episode:06d}'
    manifest = project / 'work/masks' / name / 'MASKS.json'
    sources[str(manifest)] = digest(manifest)
    rows = json.loads(manifest.read_text())['rows']
    for frame in frames:
        record = rows[frame]
        if not record['selected'] or record['score'] < 0.8:
            raise ValueError(f'frame {frame} lacks a confident observed mask')
        mask_path = Path(record['mask'])
        image_path = project / 'work/frames' / name / f'{frame:06d}.jpg'
        mask = np.asarray(Image.open(mask_path)) > 127
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            raise ValueError('insufficient mask pixels')
        center = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
        covariance = np.cov(np.stack([xs, ys]))
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        phi = math.atan2(direction[1], direction[0])
        tilt = math.acos(np.clip(math.sqrt(values[0] / values[1]), 0.03, 1.0))
        major_px = math.sqrt(2 * values[1])
        side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 1.5))
        left, top = math.floor(center[0] - side / 2), math.floor(center[1] - side / 2)
        box = (left, top, left + side, top + side)
        image = Image.open(image_path).convert('RGB').crop(box).resize((size, size), Image.Resampling.BILINEAR)
        target = Image.fromarray(mask.astype('uint8') * 255).crop(box).resize((size, size), Image.Resampling.NEAREST)
        scale = size / side
        k = np.array([[focal * scale, 0, (mask.shape[1] / 2 - left) * scale],
                      [0, focal * scale, (mask.shape[0] / 2 - top) * scale], [0, 0, 1]], np.float32)
        images.append(np.asarray(image))
        targets.append(np.asarray(target, dtype=np.float32) / 255)
        intrinsics.append(k)
        initial.append(dict(center=center, major_px=major_px, tilt=tilt, phi=phi, source_wh=mask.shape[::-1]))
        metadata.append(dict(frame=frame, crop=list(box), focal_source_px=focal, mask_score=record['score'],
                             mask_pixels=len(xs), source_wh=list(mask.shape[::-1])))
        sources[str(mask_path)] = digest(mask_path)
        sources[str(image_path)] = digest(image_path)
    return images, np.stack(targets), np.stack(intrinsics), initial, metadata


def fit_adam(context, vertices, faces, target, intrinsics, angles0, translation0, steps=180,
        tube_basis=None, shape_initial=0.0, fit_shape=False):
    device = target.device
    angles = torch.nn.Parameter(torch.as_tensor(angles0, device=device).float())
    # Translation xy is camera metric; z is constrained positive by exponentiation.
    translation = torch.nn.Parameter(torch.as_tensor(translation0, device=device).float())
    log_tube = torch.nn.Parameter(torch.tensor(float(shape_initial), device=device), requires_grad=fit_shape)
    parameters = [{'params': [angles], 'lr': 0.003}, {'params': [translation], 'lr': 0.002}]
    if fit_shape:
        parameters.append({'params': [log_tube], 'lr': 0.005})
    optimizer = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, steps // 2), max(2, 4 * steps // 5)], gamma=0.2)
    distance = make_distance(target)
    history, best = [], None
    for step in range(steps + 1):
        optimizer.zero_grad()
        vv = vertices if tube_basis is None else vertices + tube_basis * (log_tube.exp() - 1)
        r = rotation_xyz(angles)
        t = torch.stack([translation[:, 0], translation[:, 1], translation[:, 2].exp()], -1)
        vc = vv[None] @ r.transpose(-1, -2) + t[:, None]
        if step == 0 and (not torch.isfinite(vc).all() or vc[..., 2].min() <= .05):
            raise ValueError('initializer must place the complete object in front of the near plane')
        pred = render(context, vc, faces, intrinsics, target.shape[-2:])
        if step == 0 and (pred.sum((1, 2)) < .1 * target.sum((1, 2))).any():
            raise ValueError('initializer has empty/mostly offscreen silhouette; refusing uninformative optimization')
        loss = loss_silhouette(pred, target, distance)
        # Tube radius stays within a factor of two of its explicit initial prior.
        loss = loss + 0.001 * log_tube.square()
        number = float(loss.detach())
        if not math.isfinite(number):
            raise RuntimeError('nonfinite loss')
        visible = bool((vc[..., 2] > .05).all() and (pred.sum((1, 2)) >= .1 * target.sum((1, 2))).all())
        if visible and (best is None or number < best[0]):
            best = (number, r.detach().clone(), t.detach().clone(), pred.detach().clone(), float(log_tube.detach()))
        if step % 30 == 0:
            history.append({'step': step, 'loss': number, 'iou': ious(pred.detach(), target)})
        if step < steps:
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in [angles, translation, log_tube]):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                log_tube.clamp_(-math.log(2), math.log(2))
    return best, history


def fit(context, vertices, faces, target, intrinsics, angles0, translation0, steps=180,
        tube_basis=None, shape_initial=0.0, fit_shape=False):
    """Bounded derivative-free pose fitting; shared tube shape alternates with pose."""
    from scipy.optimize import minimize, minimize_scalar
    distance=make_distance(target)
    parameters=np.concatenate([np.asarray(angles0),np.asarray(translation0)],axis=1).astype(np.float64)
    origins=parameters.copy()
    log_tube=float(shape_initial)
    history=[]
    def prediction(pp, shape, index=None):
        v=vertices if tube_basis is None else vertices+tube_basis*(math.exp(shape)-1)
        angles=torch.tensor(pp[:,:3],device=target.device,dtype=torch.float32)
        r=rotation_xyz(angles)
        values=torch.tensor(pp[:,3:],device=target.device,dtype=torch.float32)
        t=torch.stack([values[:,0],values[:,1],values[:,2].exp()],-1)
        vc=v[None]@r.transpose(-1,-2)+t[:,None]
        if not torch.isfinite(vc).all() or vc[...,2].min()<=.05:
            return None,r,t
        kk=intrinsics if index is None else intrinsics[index:index+1]
        return render(context,vc,faces,kk,target.shape[-2:]),r,t
    def overall(pp,shape):
        with torch.no_grad():
            pred,r,t=prediction(pp,shape)
            if pred is None or (pred.sum((1,2))<.1*target.sum((1,2))).any():
                return None
            value=float(loss_silhouette(pred,target,distance))+.001*shape*shape
            return (value,r.detach(),t.detach(),pred.detach(),shape)
    best=overall(parameters,log_tube)
    if best is None:raise ValueError('invalid/offscreen initializer')
    history.append({'step':0,'loss':best[0],'iou':ious(best[3],target)})
    evaluations=0
    for outer in range(2 if fit_shape else 1):
        for index in range(len(parameters)):
            def objective(pp):
                nonlocal evaluations
                evaluations+=1
                with torch.no_grad():
                    pred,_,_=prediction(np.asarray(pp)[None],log_tube,index)
                    if pred is None:return 1e6
                    observed=target[index:index+1]
                    if pred.sum()<.1*observed.sum():return 1e6
                    return float(loss_silhouette(pred,observed,distance[index:index+1]))
            before=objective(parameters[index])
            bounds=[(float(v-d),float(v+d)) for v,d in zip(origins[index],[.4,.4,.4,.2,.2,.25])]
            result=minimize(objective,parameters[index],method='Powell',bounds=bounds,
                            options={'maxiter':40,'maxfev':2000,'xtol':1e-5,'ftol':1e-6})
            if result.fun<before:parameters[index]=result.x
            print('pose_fit',outer,index,'loss',min(float(result.fun),before),'evaluations',result.nfev,flush=True)
        if fit_shape:
            def shape_objective(shape):
                candidate=overall(parameters,shape)
                return candidate[0] if candidate is not None else 1e6
            previous=shape_objective(log_tube)
            result=minimize_scalar(shape_objective,bounds=(-math.log(2),math.log(2)),method='bounded',
                                   options={'xatol':1e-4,'maxiter':60})
            if result.fun<previous:log_tube=float(result.x)
        candidate=overall(parameters,log_tube)
        if candidate is not None and candidate[0]<best[0]:best=candidate
        history.append({'step':outer+1,'loss':candidate[0] if candidate is not None else None,
                        'iou':ious(candidate[3],target) if candidate is not None else None,
                        'pose_evaluations':evaluations})
    return best,history


def ious(pred, targets):
    pred = pred > 0.5
    targets = targets > 0.5
    return ((pred & targets).sum((1, 2)) / (pred | targets).sum((1, 2)).clamp_min(1)).cpu().tolist()


def renderer_gate(context, device):
    k = torch.tensor([[[80., 0, 32], [0, 80., 32], [0, 0, 1]]], device=device)
    vertices = torch.tensor([[[-.2, -.1, 2.], [.2, -.1, 2.], [.2, .1, 2.], [-.2, .1, 2.]]], device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int32)
    pred = render(context, vertices, faces, k, [64, 64])
    ys, xs = torch.where(pred[0] > .5)
    assert abs(float(xs.float().mean()) - 31.5) < .6
    assert abs(float(ys.float().mean()) - 31.5) < .6
    shifted = vertices.clone(); shifted[..., 1] += .2
    shifted_pred = render(context, shifted, faces, k, [64, 64])
    ys2, _ = torch.where(shifted_pred[0] > .5)
    assert abs(float(ys2.float().mean() - ys.float().mean()) - 8) < .6, 'OpenCV y-down mapping failed'
    offcenter_k = k.clone(); offcenter_k[:, 0, 2] = 20; offcenter_k[:, 1, 2] = 40
    offcenter = render(context, vertices, faces, offcenter_k, [64, 64])
    yo, xo = torch.where(offcenter[0] > .5)
    assert abs(float(xo.float().mean()) - 19.5) < .6
    assert abs(float(yo.float().mean()) - 39.5) < .6
    same = render(context, vertices * 2, faces, k, [64, 64])
    assert torch.max(torch.abs(same - pred)) < 1e-4, 'perspective gauge invariance failed'
    movable = vertices.clone().requires_grad_()
    probe = render(context, movable, faces, k, [64, 64])
    weights = torch.arange(64, device=device)[None, None, :]
    (probe * weights).sum().backward()
    assert torch.isfinite(movable.grad).all() and movable.grad.abs().sum() > 0
    delta = torch.tensor(.003, device=device, requires_grad=True)
    shift_vector = torch.tensor([1., 0, 0], device=device)
    def centroid_loss(dx):
        silhouette = render(context, vertices + dx * shift_vector, faces, k, [64, 64])
        return (silhouette * weights).sum() / silhouette.sum()
    centroid_loss(delta).backward()
    epsilon = 1e-4
    numeric = float((centroid_loss(delta.detach() + epsilon) - centroid_loss(delta.detach() - epsilon)) / (2 * epsilon))
    automatic = float(delta.grad)
    assert abs(automatic - numeric) < .05 * max(abs(numeric), 1), (automatic, numeric)
    assert abs(automatic - 40) < 2, automatic
    return {'projection_y_down': True, 'offcenter_principal_point': True,
            'scale_depth_gauge': True, 'finite_nonzero_gradient': True,
            'edge_gradient_autograd': automatic, 'edge_gradient_finite_difference': numeric}


def optimizer_gate(context, device):
    v, f = torus_mesh(0.45, 0.015)
    v = torch.tensor(v, device=device)
    f = torch.tensor(f, device=device)
    k = torch.tensor([[[250., 0, 64], [0, 250., 64], [0, 0, 1]]], device=device)
    gt_angles = torch.tensor([[0.7, 0.2, 0.3]], device=device)
    gt_translation = torch.tensor([[0.08, 0.1, 3.0]], device=device)
    target = render(context, v[None] @ rotation_xyz(gt_angles).transpose(-1, -2) + gt_translation[:, None], f, k, [128, 128]).detach()
    initial_angles = gt_angles.cpu().numpy() + np.array([[.06, -.04, .05]])
    initial_translation = np.array([[.11, .08, math.log(3.05)]])
    initial_t = torch.tensor([[.11, .08, 3.05]], device=device)
    before = render(context, v[None] @ rotation_xyz(torch.tensor(initial_angles, device=device).float()).transpose(-1, -2) + initial_t[:, None], f, k, [128, 128])
    before_iou = ious(before, target)[0]
    best, history = fit(context, v, f, target, k, initial_angles, initial_translation, steps=300)
    after_iou = ious(best[3], target)[0]
    assert after_iou > before_iou + .05 and after_iou > .75, (before_iou, after_iou, history)
    assert torch.max(torch.abs(best[1] @ best[1].transpose(-1, -2) - torch.eye(3, device=device))) < 1e-5
    assert torch.det(best[1]).min() > .99999
    return {'synthetic_iou_before': before_iou, 'synthetic_iou_after': after_iou,
            'synthetic_translation_error_m': float(torch.linalg.vector_norm(best[2] - gt_translation)),
            'rotation_so3': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path('/home/tanuj/V2D_challenge'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--mesh', type=Path, help='Optional generated mesh; use its fixed dimensions, no shape deformation.')
    ap.add_argument('--initialization', type=Path, help='Required with --mesh: opencv_obj2cam_m_v1 JSON, poses keyed by video frame.')
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 35, 90, 743, 15, 60, 730, 770])
    ap.add_argument('--fit-count', type=int, default=4)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=180)
    ap.add_argument('--major-radius', type=float, default=0.45, help='Explicit fixed metric prior, NOT an estimated size.')
    ap.add_argument('--minor-radius', type=float, default=0.0075)
    ap.add_argument('--gate-only', action='store_true')
    args = ap.parse_args()
    if args.out.exists():
        ap.error('output must be a new directory')
    if args.mesh and not args.initialization:
        ap.error('--mesh requires --initialization; torus ellipse initialization is not valid for a generic mesh')
    if len(set(args.frames)) != len(args.frames) or not 0 < args.fit_count < len(args.frames):
        ap.error('require distinct frames and nonempty fit/shape-evaluation subsets')
    device = 'cuda'
    context = dr.RasterizeCudaContext()
    gates = renderer_gate(context, device)
    gates.update(optimizer_gate(context, device))
    args.out.mkdir(parents=True)
    if args.gate_only:
        (args.out / 'GATES.json').write_text(json.dumps(gates, indent=2))
        print(gates, flush=True); return
    start = time.time()
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    focal_path = args.project / 'work/s3_focal' / f'episode_{args.episode:06d}' / 'FOCAL.json'
    sources[str(focal_path)] = digest(focal_path)
    focal = json.loads(focal_path.read_text())['focal_source_px']
    images, targets_np, k_np, initial, metadata = load_frames(args.project, args.episode, args.frames, focal, args.size, sources)
    vertices, faces = torus_mesh(args.major_radius, args.minor_radius)
    major_only, _ = torus_mesh(args.major_radius, 0)
    tube_basis = torch.tensor(vertices - major_only, device=device)
    shape_model = 'procedural_torus_fixed_major_radius'
    initialized = None
    mesh_center = np.zeros(3)
    if args.mesh:
        import trimesh
        mesh = trimesh.load(args.mesh, process=False)
        vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
        initialized = json.loads(args.initialization.read_text())
        if initialized.get('schema') != 'opencv_obj2cam_m_v1':
            raise ValueError('initializer must explicitly declare OpenCV object-to-camera metres')
        fixed_scale = initialized.get('object_scale', 1.0)
        if not math.isfinite(fixed_scale) or fixed_scale <= 0:
            raise ValueError('invalid fixed object scale')
        vertices = vertices * fixed_scale
        mesh_center = vertices.mean(0)
        vertices = vertices - mesh_center
        tube_basis = None
        sources[str(args.mesh.resolve())] = digest(args.mesh)
        sources[str(args.initialization.resolve())] = digest(args.initialization)
        shape_model = 'generated_mesh_fixed_dimensions'
    vertices = torch.tensor(vertices, device=device)
    faces = torch.tensor(faces, device=device)
    targets, k = torch.tensor(targets_np, device=device), torch.tensor(k_np, device=device)
    results = []
    panels = []
    # ± plane-tilt hypotheses are equivalent under weak perspective but differ in perspective.
    for sign in ([1, -1] if initialized is None else [1]):
        angle0, trans0 = [], []
        for frame, obs in zip(args.frames, initial):
            if initialized is None:
                z = focal * args.major_radius / obs['major_px']
                w, h = obs['source_wh']
                xy = (obs['center'] - np.array([w / 2, h / 2])) * z / focal
                angle0.append([sign * obs['tilt'], 0, obs['phi']])
                trans0.append([*xy, math.log(z)])
            else:
                from scipy.spatial.transform import Rotation
                pose = initialized['poses'][str(frame)]
                rotation = np.asarray(pose['R_obj2cam'], dtype=float)
                translation = np.asarray(pose['t_obj2cam_m'], dtype=float) + rotation @ mesh_center
                if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0:
                    raise ValueError('invalid initializer pose')
                if np.max(np.abs(rotation @ rotation.T - np.eye(3))) > 1e-3 or np.linalg.det(rotation) < .999:
                    raise ValueError('initializer rotation must be SO(3)')
                angle0.append(Rotation.from_matrix(rotation).as_euler('xyz'))
                trans0.append([*translation[:2], math.log(translation[2])])
        angle0, trans0 = np.array(angle0), np.array(trans0)
        n = args.fit_count
        train, history = fit(context, vertices, faces, targets[:n], k[:n], angle0[:n], trans0[:n],
                             args.steps, tube_basis=tube_basis, fit_shape=tube_basis is not None)
        evaluated, held_history = fit(context, vertices, faces, targets[n:], k[n:], angle0[n:], trans0[n:],
                                    args.steps, tube_basis=tube_basis, shape_initial=train[4], fit_shape=False)
        pred = torch.cat([train[3], evaluated[3]])
        record = {'tilt_sign': sign, 'fit_iou': ious(pred[:n], targets[:n]),
                  'shape_evaluation_iou': ious(pred[n:], targets[n:]),
                  'initial_fit_iou': history[0]['iou'],
                  'initial_shape_evaluation_iou': held_history[0]['iou'],
                  'tube_radius_m_prior_scaled': args.minor_radius * math.exp(train[4]) if tube_basis is not None else None,
                  'fit_history': history, 'shape_evaluation_pose_fit_history': held_history,
                  'R_obj2cam': torch.cat([train[1], evaluated[1]]).cpu().tolist(),
                  't_obj2cam_m_given_size_prior': torch.cat([train[2], evaluated[2]]).cpu().tolist()}
        results.append(record)
        for image, target, prediction in zip(images, targets_np, pred.cpu().numpy()):
            panel = image.copy()
            boundary = cv2.morphologyEx((prediction > .5).astype('uint8'), cv2.MORPH_GRADIENT, np.ones((3, 3), 'uint8')) > 0
            panel[boundary] = [0, 255, 255]
            panels.append(Image.fromarray(panel))
        print(json.dumps({key: record[key] for key in ['tilt_sign', 'fit_iou', 'shape_evaluation_iou', 'tube_radius_m_prior_scaled']}), flush=True)
    montage = Image.new('RGB', (args.size * len(args.frames), args.size * 2))
    for i, panel in enumerate(panels):
        montage.paste(panel, ((i % len(args.frames)) * args.size, (i // len(args.frames)) * args.size))
    montage.save(args.out / 'OVERLAYS.jpg')
    report = {'experiment': 'isolated_object_silhouette_pose_shape_pilot', 'episode': args.episode,
              'shape_model': shape_model, 'fixed_major_radius_prior_m': args.major_radius if not args.mesh else None,
              'metric_scale_recovered': False, 'pose_evaluation_uses_image_masks': True,
              'validation_scope': 'Shared shape is frozen on evaluation frames; their poses are still fit to their masks. No GT accuracy or unseen-image pose prediction claim.',
              'temporal_optimization': False, 'human_occlusion_modeled': False,
              'solver': 'bounded_powell_alternating_shape',
              'solver_budget': {'per_pose_max_evaluations':2000,'pose_rounds_with_shape':2,
                               'bounds_about_initializer':['rx+-0.4rad','ry+-0.4rad','rz+-0.4rad','tx+-0.2m','ty+-0.2m','logz+-0.25']},
              'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'environment': {'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                              'device': torch.cuda.get_device_name(), 'nvdiffrast_module': dr.__file__},
              'mesh_recenter_offset_after_fixed_scale_m': mesh_center.tolist(),
              'gates': gates, 'frames': metadata, 'fit_count': args.fit_count,
              'results': results, 'source_sha256': sources, 'seconds': time.time() - start,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    (args.out / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('complete', args.out, 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
