#!/usr/bin/env python3
"""Hand-pose-only penetration removal for one episode of a packed Track 1 submission.

Optimizes only the 60 MHR pose parameters that move hand vertices and none of the 22 joints scored by
ACC-H (wrists 37-39, 47-49 and fingers 68-121; strategy_20261031/HAND_DOF_PROBE). The loss pushes every
proxy hand vertex to signed distance >= margin from the episode's own object mesh, using exact
winding-number / closest-point geometry re-linearized every few steps, plus a small pull back to the
original pose. Object pose, scale, mesh, identity and every other pose parameter are unchanged.
Reads an optional episode npz from physics_pp_v2 (translation/pose) to start from a smoothed trajectory.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp

FREE = [37, 38, 39, 47, 48, 49] + list(range(68, 122))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', default=None, help='physics_pp_v2 episode npz to start from')
    ap.add_argument('--out', required=True); ap.add_argument('--margin-mm', type=float, default=1.0)
    ap.add_argument('--outer', type=int, default=10); ap.add_argument('--inner', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=1e-4); ap.add_argument('--chunk', type=int, default=64)
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    pose, t, s = ep['pose'].copy(), ep['t'].copy(), ep['s']
    if a.start:
        st = np.load(a.start); pose, t = st['mhr_pose'].copy(), st['object_translation'].copy()
        s = float(st['object_scale']) if 'object_scale' in st.files else s
    hand_ids = np.load(pp.HAND_IDS); margin = a.margin_mm / 1000.0
    geo = pp.Geometry(ep['verts'], ep['faces'])
    ts = torch.jit.load(pp.MHR_MODEL, map_location='cpu').eval()
    flip = torch.tensor(pp.FLIP, dtype=torch.float32)
    shape = torch.from_numpy(ep['shape'].reshape(1, 45).astype(np.float32))
    scales = torch.from_numpy(ep['scales'].astype(np.float32))
    hid = torch.from_numpy(hand_ids.astype(np.int64))
    R = ep['R']; T = len(pose)
    new_pose = pose.copy(); log = []

    def hands_of(p_chunk, delta):
        p = p_chunk.clone(); p[:, FREE] = p[:, FREE] + delta
        params = torch.cat([p, scales.expand(len(p), 68)], 1)
        v, _ = ts(shape.expand(len(p), 45), params, torch.zeros(len(p), 72), True)
        return v[:, hid] / 100.0 * flip

    def signed(hands_np, sl):
        local = np.einsum('tpi,tij->tpj', hands_np - t[sl, None], R[sl]) / s
        d, nl = geo.signed(local.reshape(-1, 3), 0.01 / s)
        n = len(hands_np)
        return d.reshape(n, -1) * s, np.einsum('tij,tpj->tpi', R[sl], nl.reshape(n, -1, 3))

    for c0 in range(0, T, a.chunk):
        sl = slice(c0, min(T, c0 + a.chunk))
        p0 = torch.from_numpy(pose[sl].astype(np.float32))
        delta = torch.zeros(len(p0), len(FREE), requires_grad=True)
        with torch.no_grad():
            h = hands_of(p0, delta).double().numpy()
        d, n = signed(h, sl)
        if not (d < margin).any():
            continue
        opt = torch.optim.Adam([delta], lr=0.02)
        for outer in range(a.outer):
            mask = torch.from_numpy((d < margin + 0.004).astype(np.float32))
            if mask.sum() == 0:
                break
            dt = torch.from_numpy(np.where(np.isfinite(d), d, 1.0).astype(np.float32))
            nt = torch.from_numpy(n.astype(np.float32)); h_lin = torch.from_numpy(h.astype(np.float32))
            for inner in range(a.inner):
                opt.zero_grad()
                hv = hands_of(p0, delta)
                dl = dt + ((hv - h_lin) * nt).sum(-1)                   # linearized signed distance
                viol = torch.relu(margin - dl) * mask
                loss = (viol ** 2).sum() * 1e4 + a.alpha * (delta ** 2).sum()
                loss.backward(); opt.step()
            with torch.no_grad():
                h = hands_of(p0, delta).double().numpy()
            d, n = signed(h, sl)
            log.append({'chunk': c0, 'outer': outer, 'inside_points': int((d < 0).sum()),
                        'frames_inside': int((d < 0).any(1).sum()), 'max_depth_mm': float(max(0.0, -np.nanmin(d)) * 1000)})
            if not (d < margin).any():
                break
        new_pose[sl, FREE] = pose[sl, FREE] + delta.detach().double().numpy()
    v_before, _ = pp.decode(pose, ep['scales'], ep['shape'])
    v_after, j_after = pp.decode(new_pose, ep['scales'], ep['shape'])
    _, j_before = pp.decode(pose, ep['scales'], ep['shape'])
    rep = {'episode': a.episode, 'frames': T, 'start': a.start, 'margin_mm': a.margin_mm,
           'pen_before': pp.pen_proxy(ep, v_before[:, hand_ids], t, s),
           'pen_after': pp.pen_proxy(ep, v_after[:, hand_ids], t, s),
           'scored_joint_max_change_m': float(np.abs(j_after[:, pp.BODY_J] - j_before[:, pp.BODY_J]).max()),
           'hand_vertex_change_cm_max': float(np.linalg.norm(v_after - v_before, axis=2).max() * 100),
           'body_vertex_change_cm_mean': float(np.linalg.norm(v_after - v_before, axis=2).mean() * 100),
           'free_param_change_rad_max': float(np.abs(new_pose[:, FREE] - pose[:, FREE]).max()),
           'seconds': time.time() - t0, 'log': log, 'source_sha256': pp.sha(__file__)}
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=ep['frames'], object_translation=t, mhr_pose=new_pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
