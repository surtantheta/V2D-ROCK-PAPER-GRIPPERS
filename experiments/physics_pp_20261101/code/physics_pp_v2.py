#!/usr/bin/env python3
"""Physics post-processor for one episode of a packed Track 1 submission.

Input is the submitted Parquet itself; no ground truth, images or other models are read.
  1. Human (optional, --lam-human > 0): penalized second-difference smoothing of the 136 MHR pose
     parameters inside each scored stretch, with the first scored frame held exactly (it fixes the
     evaluation Sim(3)). Identity and scales are unchanged.
  2. Object (v2): translation is first smoothed (second-difference penalty --lam-object, linear).
     Then every scored frame gets the smallest translation correction that puts every proxy hand vertex
     at signed distance >= --margin-mm, found per frame from exact winding-number / closest-point
     geometry (the object moves AGAINST the outward normal at each closest surface point). Frames that
     no translation can clear (hands squeezing from opposite sides) shrink the episode's object scale
     in 1% steps, at most --max-shrink. Finally the corrections are smoothed inside the half-spaces that
     the feasible per-frame corrections satisfy, and any frame still inside gets its own last correction.
     Rotation and mesh are unchanged.
Outputs <out>/episode_XXXXXX.npz (new translation, scale and pose) and a JSON report.
v1 (physics_pp.py) diverged on episode 29: stale accumulated linearizations plus Huber weights let the
object drift metres; v2 only ever constrains around a known-feasible correction.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numba
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

KIT = Path('/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit')
_spec = importlib.util.spec_from_file_location('v2d_scorer', KIT / 'metric_code/track_1/PEN.py')
sc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sc)
ARRAY_NAMES = list(sc._ARRAYS)
MHR_MODEL = '/home/tanuj/V2D_challenge/models/mhr/mhr_model.pt'
HAND_IDS = '/rhome/tanuj/v2d_joint_kaggle_20261031/reproduction_hand_vertices/HAND_VERTEX_IDS.npy'
FLIP = np.array([1.0, -1.0, -1.0])
BODY_J = list(sc.MHR_TABLE3_BODY_JOINT_INDICES)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- submission I/O
def load_episode(parquet, episode):
    df = pd.read_parquet(parquet, columns=['row_id', 'x', 'y', 'z'])
    df = df[df['row_id'].str.startswith(f't1_{episode:06d}_')]
    p = df['row_id'].str.split('_', expand=True)
    block = pd.DataFrame({'episode_index': episode, 'frame_index': p[2].astype(int).to_numpy(),
                          'array_name': [ARRAY_NAMES[i] for i in p[3].astype(int)],
                          'point_index': p[4].astype(int).to_numpy(),
                          'x': df['x'].to_numpy(), 'y': df['y'].to_numpy(), 'z': df['z'].to_numpy()})
    raw = sc._to_arrays(block, ['x', 'y', 'z'])
    frames = np.unique(block.loc[block['array_name'].map(lambda n: sc._ARRAYS[n][0]) == 'frame', 'frame_index'])
    ep = {'frames': frames,
          'pose': sc._unpack(raw['mhr_pose'], 136, 'mhr_pose'),
          'scales': sc._unpack(raw['mhr_scales'], 68, 'mhr_scales'),
          'shape': sc._unpack(raw['mhr_shape'], 45, 'mhr_shape'),
          'R': sc._rotations(raw['object_rotation']),
          't': raw['object_translation'][:, 0, :].copy(),
          's': sc._object_scale(raw['object_scale'])}
    verts = raw['object_mesh_vertices']
    faces = sc._mesh_faces(raw['object_mesh_faces'], len(verts))
    a, b, c = (verts[faces[:, k]] for k in range(3))
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 0.0      # the scorer drops zero-area padding
    ep['verts'], ep['faces'] = verts, faces[keep]
    return ep


# ---------------------------------------------------------------- MHR decoding
_TS = None


def decode(pose, scales, shape, need='both', chunk=128):
    global _TS
    if _TS is None:
        _TS = torch.jit.load(MHR_MODEL, map_location='cpu').eval()
    params = np.concatenate([pose, np.broadcast_to(scales, (len(pose), 68))], 1).astype(np.float32)
    vs, js = [], []
    with torch.no_grad():
        for s in range(0, len(params), chunk):
            pp = torch.from_numpy(params[s:s + chunk])
            v, k = _TS(torch.from_numpy(shape.reshape(1, 45).astype(np.float32)).expand(len(pp), 45), pp,
                       torch.zeros(len(pp), 72), True)
            vs.append(v.double().numpy() / 100.0 * FLIP)
            js.append(k[..., :3].double().numpy() / 100.0 * FLIP)
    return np.concatenate(vs), np.concatenate(js)


# ---------------------------------------------------------------- temporal operators
def stretches(frames):
    return sc._stretches(frames)


def d2_matrix(frames):
    """Second-difference operator restricted to each stretch of consecutive scored frames."""
    rows, cols, vals, r = [], [], [], 0
    for part in stretches(frames):
        for k in range(part.start, part.stop - 2):
            rows += [r, r, r]; cols += [k, k + 1, k + 2]; vals += [1.0, -2.0, 1.0]; r += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(r, len(frames)))


def own_acc(x, frames):
    """Official ACC per-frame series against a zero-acceleration reference (padded endpoints)."""
    out = [sc.acceleration_error_per_frame(x[p], np.zeros_like(x[p])) for p in stretches(frames)]
    return float(np.mean(np.concatenate(out)) * 100.0)


def smooth_pose(pose, frames, lam):
    """Penalized smoothing of every pose parameter; first scored frame pinned exactly."""
    if lam <= 0:
        return pose.copy()
    T = len(pose)
    x = pose.copy(); x[:, 3:] = np.unwrap(x[:, 3:], axis=0)
    D = d2_matrix(frames)
    w = np.ones(T); w[0] = 1e9
    A = (sparse.diags(w) + lam * (D.T @ D)).tocsc()
    out = spsolve(A, (w[:, None] * x))
    out = np.asarray(out).reshape(T, -1)
    out[0] = pose[0]
    out[:, 3:] = out[:, 3:] + (pose[0, 3:] - x[0, 3:])          # undo unwrap offset at the anchor
    return out


# ---------------------------------------------------------------- exact geometry kernels
@numba.njit(parallel=True, cache=True)
def closest_points(points, a, b, c):
    """Closest point on a triangle soup (Ericson, Real-Time Collision Detection 5.1.5)."""
    n = points.shape[0]
    out = np.empty((n, 3)); dist = np.empty(n)
    for i in numba.prange(n):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        best = 1e300; qx = qy = qz = 0.0
        for f in range(a.shape[0]):
            ax, ay, az = a[f, 0], a[f, 1], a[f, 2]
            abx, aby, abz = b[f, 0] - ax, b[f, 1] - ay, b[f, 2] - az
            acx, acy, acz = c[f, 0] - ax, c[f, 1] - ay, c[f, 2] - az
            apx, apy, apz = px - ax, py - ay, pz - az
            d1 = abx * apx + aby * apy + abz * apz; d2 = acx * apx + acy * apy + acz * apz
            if d1 <= 0.0 and d2 <= 0.0:
                cx, cy, cz = ax, ay, az
            else:
                bpx, bpy, bpz = px - b[f, 0], py - b[f, 1], pz - b[f, 2]
                d3 = abx * bpx + aby * bpy + abz * bpz; d4 = acx * bpx + acy * bpy + acz * bpz
                if d3 >= 0.0 and d4 <= d3:
                    cx, cy, cz = b[f, 0], b[f, 1], b[f, 2]
                else:
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        v = d1 / (d1 - d3); cx, cy, cz = ax + v * abx, ay + v * aby, az + v * abz
                    else:
                        cpx, cpy, cpz = px - c[f, 0], py - c[f, 1], pz - c[f, 2]
                        d5 = abx * cpx + aby * cpy + abz * cpz; d6 = acx * cpx + acy * cpy + acz * cpz
                        if d6 >= 0.0 and d5 <= d6:
                            cx, cy, cz = c[f, 0], c[f, 1], c[f, 2]
                        else:
                            vb = d5 * d2 - d1 * d6
                            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                w = d2 / (d2 - d6); cx, cy, cz = ax + w * acx, ay + w * acy, az + w * acz
                            else:
                                va = d3 * d6 - d5 * d4
                                if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                                    cx = b[f, 0] + w * (c[f, 0] - b[f, 0])
                                    cy = b[f, 1] + w * (c[f, 1] - b[f, 1])
                                    cz = b[f, 2] + w * (c[f, 2] - b[f, 2])
                                else:
                                    den = 1.0 / (va + vb + vc); v = vb * den; w = vc * den
                                    cx, cy, cz = ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w
            dd = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if dd < best:
                best = dd; qx, qy, qz = cx, cy, cz
        out[i, 0], out[i, 1], out[i, 2] = qx, qy, qz
        dist[i] = np.sqrt(best)
    return out, dist


class Geometry:
    """Signed distance of local-frame points to one object mesh: coarse grid filter, exact near the surface."""

    def __init__(self, verts, faces, cells=64):
        self.a, self.b, self.c = (np.ascontiguousarray(verts[faces[:, k]], dtype=np.float64) for k in range(3))
        self.winding = sc._PENETRATION_KERNELS[0]
        self.lo, self.hi = verts.min(0), verts.max(0)
        ext = self.hi - self.lo
        self.cell = float(ext.max() / cells)
        pad = 3 * self.cell
        self.glo = self.lo - pad
        n = np.ceil((ext + 2 * pad) / self.cell).astype(int) + 1
        axes = [self.glo[k] + self.cell * np.arange(n[k]) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        # Magnitude from dense surface samples, sign from exact winding numbers.
        area = np.linalg.norm(np.cross(self.b - self.a, self.c - self.a), axis=1)
        rng = np.random.default_rng(0)
        k = rng.choice(len(area), size=200000, p=area / area.sum())
        u, v = rng.random(200000), rng.random(200000); flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        samples = self.a[k] + u[:, None] * (self.b[k] - self.a[k]) + v[:, None] * (self.c[k] - self.a[k])
        self.tree = cKDTree(samples)
        mag, _ = self.tree.query(grid)
        inside = np.abs(self.winding(np.ascontiguousarray(grid), self.a, self.b, self.c)) > 0.5
        self.grid = np.where(inside, -mag, mag).reshape(n)
        self.n = n

    def coarse(self, p):
        g = (p - self.glo) / self.cell
        i0 = np.clip(np.floor(g).astype(int), 0, self.n - 2); f = np.clip(g - i0, 0.0, 1.0)
        out = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    out += w * self.grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        outside_grid = ((g < 0) | (g > self.n - 1)).any(1)
        out[outside_grid] = np.inf
        return out

    def signed(self, p, band):
        """Exact signed distance and outward unit direction for points with coarse distance < band."""
        d = np.full(len(p), np.inf); nrm = np.zeros((len(p), 3))
        idx = np.flatnonzero(self.coarse(p) < band + 2.0 * self.cell)
        if idx.size:
            q = np.ascontiguousarray(p[idx])
            cp, dist = closest_points(q, self.a, self.b, self.c)
            inside = np.abs(self.winding(q, self.a, self.b, self.c)) > 0.5
            dvec = np.where(inside[:, None], cp - q, q - cp)
            nn = np.linalg.norm(dvec, axis=1, keepdims=True)
            nrm[idx] = dvec / np.maximum(nn, 1e-12)
            d[idx] = np.where(inside, -dist, dist)
        return d, nrm


# ---------------------------------------------------------------- object refit
def smooth_translation(t, frames, lam):
    D = d2_matrix(frames)
    A = (sparse.identity(len(frames)) + lam * (D.T @ D)).tocsc()
    return np.asarray(spsolve(A, t)).reshape(-1, 3)


def frame_geometry(geo, hands_f, t_f, R_f, s, band):
    """Signed world distance and outward world direction of one frame's hand points."""
    local = (hands_f - t_f) @ R_f / s
    d, nl = geo.signed(local, band / s)
    return d * s, nl @ R_f.T


def min_correction(geo, hands_f, base_f, R_f, s, margin, c0, anchor=None, band=0.006, per_frame=160, iters=15):
    """c closest to `anchor` (default 0) with every hand point at signed distance >= margin at base_f + c.

    Re-linearizes exact geometry each iteration; returns (c, feasible, constraints at the final c).
    """
    from scipy.optimize import minimize
    anchor = np.zeros(3) if anchor is None else anchor
    c = c0.copy(); A = np.zeros((0, 3)); b = np.zeros(0)
    for _ in range(iters):
        d, nw = frame_geometry(geo, hands_f, base_f + c, R_f, s, band)
        sel = np.flatnonzero(d < margin + 0.003)
        if sel.size == 0 or d[sel].min() >= margin - 1e-6:
            sel = sel[np.argsort(d[sel])[:per_frame]]
            return c, True, (nw[sel], nw[sel] @ c + d[sel] - margin)
        sel = sel[np.argsort(d[sel])[:per_frame]]
        A, b = nw[sel], nw[sel] @ c + d[sel] - margin          # A c' <= b keeps every selected point clear
        res = minimize(lambda x: (x - anchor) @ (x - anchor), c, jac=lambda x: 2 * (x - anchor), method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': lambda x: b - A @ x, 'jac': lambda x: -A}],
                       options={'maxiter': 200, 'ftol': 1e-12})
        step = res.x - c
        if np.linalg.norm(step) > 0.03:                          # trust region: 3 cm per linearization
            step *= 0.03 / np.linalg.norm(step)
        c = c + step
    d, nw = frame_geometry(geo, hands_f, base_f + c, R_f, s, band)
    sel = np.flatnonzero(d < margin + 0.003); sel = sel[np.argsort(d[sel])[:per_frame]]
    return c, bool(d.min() >= margin - 1e-6), (nw[sel], nw[sel] @ c + d[sel] - margin)


def clear_all(geo, hands, base, R, s, margin, c_init=None, frames_to_check=None):
    T = len(base); c = np.zeros((T, 3)) if c_init is None else c_init.copy()
    cons, infeasible = {}, []
    check = range(T) if frames_to_check is None else frames_to_check
    for f in check:
        d, nw = frame_geometry(geo, hands[f], base[f] + c[f], R[f], s, 0.006)
        if c_init is None and d.min() >= margin - 1e-6:
            sel = np.flatnonzero(d < margin + 0.003)
            if sel.size:                               # clear but close: still bounds the smoothing step
                sel = sel[np.argsort(d[sel])[:160]]
                cons[f] = (nw[sel], nw[sel] @ c[f] + d[sel] - margin)
            continue
        anchor = None if c_init is None else c_init[f]
        c[f], ok, cons[f] = min_correction(geo, hands[f], base[f], R[f], s, margin, c[f], anchor=anchor)
        if not ok:
            infeasible.append(int(f))
    return c, cons, infeasible


def smooth_corrections(base, c, cons, frames, lam, mu=1e7):
    """Smooth corrections while staying inside each frame's feasible half-spaces (penalty active set)."""
    T = len(base)
    D = d2_matrix(frames); K = sparse.kron((D.T @ D).tocsr(), sparse.identity(3)).tocsr()
    target = c.reshape(-1); x = c.copy()
    for _ in range(60):
        blocks = np.zeros((T, 3, 3)); rhs = np.zeros((T, 3))
        for f, (N, B) in cons.items():
            if len(N) == 0:
                continue
            act = N @ x[f] > B - 1e-10
            if act.any():
                blocks[f] = mu * N[act].T @ N[act]; rhs[f] = mu * N[act].T @ B[act]
        # minimize |x - c|^2 + lam |D2 (base + x)|^2 + penalties
        A = (sparse.identity(3 * T) + lam * K + sparse.block_diag(list(blocks), format='csr')).tocsc()
        new = np.asarray(spsolve(A, target - lam * (K @ base.reshape(-1)) + rhs.reshape(-1))).reshape(T, 3)
        change = float(np.abs(new - x).max()); x = new
        if change < 1e-8:
            break
    return x


def refit_translation(ep, hands, lam, margin, max_shrink=0.08):
    """v2: smooth, feasible per-frame corrections (with scale shrink if needed), constrained smoothing."""
    frames, R = ep['frames'], ep['R']
    geo = Geometry(ep['verts'], ep['faces'])
    base = smooth_translation(ep['t'], frames, lam)
    hist = []
    s = ep['s']
    for k in range(int(round(max_shrink * 100)) + 1):
        s = ep['s'] * (1.0 - 0.01 * k)
        c, cons, infeasible = clear_all(geo, hands, base, R, s, margin)
        hist.append({'shrink_percent': k, 'frames_constrained': len(cons), 'frames_corrected': int((np.abs(c) > 0).any(1).sum()),
                     'infeasible_frames': len(infeasible)})
        if not infeasible:
            break
    x = smooth_corrections(base, c, cons, frames, lam)
    # Exact check of every frame; any frame not clear gets its own final correction.
    c2, cons2, infeasible2 = clear_all(geo, hands, base, R, s, margin, c_init=x,
                                       frames_to_check=[f for f in range(len(base))
                                                        if frame_geometry(geo, hands[f], base[f] + x[f], R[f], s, 0.006)[0].min() < margin - 1e-6])
    hist.append({'final_frames_recorrected': len(cons2), 'final_infeasible_frames': len(infeasible2)})
    return base + c2, s, hist


# ---------------------------------------------------------------- diagnostics
def pen_proxy(ep, hands, t, s=None):
    """Mean proxy penetration (cm) with the scorer's own exact kernels, alignment scale 1."""
    s = ep['s'] if s is None else s
    local = np.einsum('tpi,tij->tpj', hands - t[:, None], ep['R']) / s
    depth = sc._penetration_depth(local.reshape(-1, 3), ep['verts'], ep['faces']).reshape(hands.shape[:2])
    per_frame = sc.sdf_interpenetration_distance_per_frame(-depth * s)
    return float(np.mean(per_frame)), float(depth.max() * s * 100.0), int((depth > 0).any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--lam-object', type=float, required=True)
    ap.add_argument('--lam-human', type=float, default=0.0); ap.add_argument('--margin-mm', type=float, default=1.0)
    ap.add_argument('--no-clearance', action='store_true'); ap.add_argument('--max-shrink', type=float, default=0.08)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ep = load_episode(a.parquet, a.episode)
    hand_ids = np.load(HAND_IDS)
    frames = ep['frames']
    v0, j0 = decode(ep['pose'], ep['scales'], ep['shape'])
    pose = smooth_pose(ep['pose'], frames, a.lam_human)
    if a.lam_human > 0:
        v1, j1 = decode(pose, ep['scales'], ep['shape'])
    else:
        v1, j1 = v0, j0
    hands = v1[:, hand_ids]
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'stretches': len(stretches(frames)),
           'lam_object': a.lam_object, 'lam_human': a.lam_human, 'margin_mm': a.margin_mm,
           'own_acc_h_before': own_acc(j0[:, BODY_J], frames), 'own_acc_h_after': own_acc(j1[:, BODY_J], frames),
           'human_first_frame_exact': bool(np.array_equal(pose[0], ep['pose'][0])),
           'human_mean_vertex_change_cm': float(np.linalg.norm(v1 - v0, axis=2).mean() * 100.0),
           'own_acc_o_before': own_acc(ep['t'], frames)}
    rep['pen_before'] = pen_proxy(ep, hands, ep['t'])
    if a.no_clearance:
        t, s, hist = smooth_translation(ep['t'], frames, a.lam_object), ep['s'], []
    else:
        t, s, hist = refit_translation(ep, hands, a.lam_object, a.margin_mm / 1000.0, a.max_shrink)
    rep.update({'own_acc_o_after': own_acc(t, frames), 'pen_after': pen_proxy(ep, hands, t, s), 'history': hist,
                'object_scale_before': ep['s'], 'object_scale_after': s,
                'translation_change_cm_mean': float(np.linalg.norm(t - ep['t'], axis=1).mean() * 100.0),
                'translation_change_cm_max': float(np.linalg.norm(t - ep['t'], axis=1).max() * 100.0),
                'seconds': time.time() - t0,
                'source_sha256': sha(__file__), 'parquet_sha256': sha(a.parquet)})
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
