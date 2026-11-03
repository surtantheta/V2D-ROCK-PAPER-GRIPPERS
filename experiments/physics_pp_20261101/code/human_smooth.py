#!/usr/bin/env python3
"""Joint-space human smoothing for one episode of a packed Track 1 submission.

The 22 body joints scored by ACC-H are smoothed in closed form (second-difference penalty within each
stretch of consecutive scored frames; the first scored frame is pinned exactly because the scorer fits its
Sim(3) there, unless --free-first, which lets smoothing denoise that frame and so the alignment). Each later frame's non-hand MHR pose parameters are then refit to the smoothed joints by
damped Gauss-Newton through the MHR TorchScript. Nothing is unwrapped or averaged in parameter space, so
the Euler/translation/scale channels of the MHR parameter transform cannot be mixed up (the failure of
physics_pp_v2 --lam-human). Hand-only parameters, identity, object and mesh are unchanged. An optional
physics_pp_v2 episode npz supplies the starting object translation/pose (for example a smoothed object).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp
from hand_clear import FREE as HAND_FREE

FIT = [i for i in range(136) if i not in set(HAND_FREE)]


def smooth_targets(j, frames, lam, pin_first=True):
    """Penalized second-difference smoothing of [T, J, 3] joints; frame 0 pinned exactly unless pin_first=False."""
    T = len(j)
    D = pp.d2_matrix(frames)
    w = np.ones(T); w[0] = 1e9 if pin_first else 1.0
    A = (sparse.diags(w) + lam * (D.T @ D)).tocsc()
    out = np.asarray(spsolve(A, w[:, None] * j.reshape(T, -1))).reshape(j.shape)
    if pin_first:
        out[0] = j[0]
    return out


def fit_chunk(ts, shape, scales, p0, target_cm, iters, gamma, max_step):
    """Damped Gauss-Newton on the FIT parameters of each frame (frames are independent)."""
    n = len(p0)
    fit = torch.tensor(FIT)
    base = torch.from_numpy(p0.astype(np.float32))
    x0 = base[:, fit].double()
    x = x0.clone()
    tgt = torch.from_numpy(target_cm.astype(np.float32)).reshape(n, -1)
    eye = torch.eye(len(FIT), dtype=torch.float64)

    def joints(xv, grad):
        xv = xv.float().requires_grad_(grad)
        full = base.index_copy(1, fit, xv)
        params = torch.cat([full, scales.expand(n, 68)], 1)
        _, k = ts(shape.expand(n, 45), params, torch.zeros(n, 72), True)
        return xv, k[:, pp.BODY_J, :3].reshape(n, -1)

    def cost(xv):
        with torch.no_grad():
            _, J = joints(xv, False)
        return ((tgt - J).double() ** 2).sum(1) + gamma * ((xv - x0) ** 2).sum(1)

    c = cost(x)
    for _ in range(iters):
        xv, J = joints(x, True)
        r = (tgt - J.detach()).double()
        jac = torch.empty(n, J.shape[1], len(FIT), dtype=torch.float64)
        for k in range(J.shape[1]):
            g, = torch.autograd.grad(J[:, k].sum(), xv, retain_graph=k < J.shape[1] - 1)
            jac[:, k] = g.double()
        A = jac.transpose(1, 2) @ jac + gamma * eye
        b = jac.transpose(1, 2) @ r[..., None] - gamma * (x - x0)[..., None]
        step = torch.linalg.solve(A, b).squeeze(-1)
        norm = step.norm(dim=1, keepdim=True)
        step = step * torch.clamp(max_step / (norm + 1e-12), max=1.0)
        accepted = torch.zeros(n, dtype=torch.bool)
        for scale in (1.0, 0.5, 0.25):                          # per-frame backtracking
            trial = x + scale * step * (~accepted)[:, None]
            ct = cost(trial)
            better = (ct < c) & ~accepted
            x = torch.where(better[:, None], trial, x); c = torch.where(better, ct, c)
            accepted |= better
            if accepted.all():
                break
    out = p0.copy(); out[:, FIT] = x.detach().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', default=None, help='physics_pp_v2 / hand_clear episode npz to start from')
    ap.add_argument('--lam', type=float, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=6); ap.add_argument('--gamma', type=float, default=1e-3)
    ap.add_argument('--max-step', type=float, default=0.5); ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--free-first', action='store_true',
                    help='smooth and refit the first scored frame too (it sets the scorer Sim(3), so its noise '
                         'misaligns every frame; pinned by default)')
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    pose, t, s = ep['pose'].copy(), ep['t'].copy(), ep['s']
    if a.start:
        st = np.load(a.start); pose, t = st['mhr_pose'].copy(), st['object_translation'].copy()
        s = float(st['object_scale']) if 'object_scale' in st.files else s
    frames = ep['frames']
    v_before, j_before = pp.decode(pose, ep['scales'], ep['shape'])
    target = smooth_targets(j_before[:, pp.BODY_J], frames, a.lam, pin_first=not a.free_first)
    target_cm = target * 100.0 * pp.FLIP                          # back to the native MHR frame
    ts = torch.jit.load(pp.MHR_MODEL, map_location='cpu').eval()
    shape = torch.from_numpy(ep['shape'].reshape(1, 45).astype(np.float32))
    scales = torch.from_numpy(ep['scales'].astype(np.float32))
    new_pose = pose.copy()
    for c0 in range(0 if a.free_first else 1, len(pose), a.chunk):  # frame 0 stays exact unless --free-first
        sl = slice(c0, min(len(pose), c0 + a.chunk))
        new_pose[sl] = fit_chunk(ts, shape, scales, pose[sl], target_cm[sl], a.iters, a.gamma, a.max_step)
    v_after, j_after = pp.decode(new_pose, ep['scales'], ep['shape'])
    hand_ids = np.load(pp.HAND_IDS)
    resid = np.linalg.norm(j_after[:, pp.BODY_J] - target, axis=2) * 1000
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'start': a.start, 'lam': a.lam, 'free_first': a.free_first,
           'own_acc_h_before': pp.own_acc(j_before[:, pp.BODY_J], frames),
           'own_acc_h_target': pp.own_acc(target, frames),
           'own_acc_h_after': pp.own_acc(j_after[:, pp.BODY_J], frames),
           'fit_residual_mm_mean': float(resid.mean()), 'fit_residual_mm_max': float(resid.max()),
           'joint_change_cm_mean': float(np.linalg.norm(j_after[:, pp.BODY_J] - j_before[:, pp.BODY_J], axis=2).mean() * 100),
           'joint_change_cm_max': float(np.linalg.norm(j_after[:, pp.BODY_J] - j_before[:, pp.BODY_J], axis=2).max() * 100),
           'human_mean_vertex_change_cm': float(np.linalg.norm(v_after - v_before, axis=2).mean() * 100),
           'human_max_vertex_change_cm': float(np.linalg.norm(v_after - v_before, axis=2).max() * 100),
           'human_first_frame_exact': bool(np.array_equal(new_pose[0], pose[0])),
           'pen_before': pp.pen_proxy(ep, v_before[:, hand_ids], t, s),
           'pen_after': pp.pen_proxy(ep, v_after[:, hand_ids], t, s),
           'own_acc_o': pp.own_acc(t, frames),
           'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__), 'parquet_sha256': pp.sha(a.parquet)}
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=new_pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
