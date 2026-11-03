#!/usr/bin/env python3
"""Whole-episode hand/arm penetration clearance for one episode of a packed Track 1 submission.

Unlike hand_clear.py (hand-only parameters, chunked, which removed only 30-60% of penetration because large
parts of the palm sit about 1 cm inside the object), every non-root MHR pose parameter may move. The loss is
  penetration:  sum relu(margin - signed distance)^2 over the proxy hand vertices (cm), with the exact
                winding-number / closest-point signed distance re-linearized every few steps;
  joints:       w_joint * sum |J22 - J22_start|^2 (cm), keeping the 22 ACC-H joints where they were;
  temporal:     lam_t * sum |D2 (J22 - J22_start)|^2 within each scored stretch, so no jitter is added;
  pull:         gamma * |delta|^2.
Root parameters (0:6), the first scored frame (the scorer's Sim(3) anchor), the object, identity and mesh are
unchanged. Starts from a physics_pp_v2 / human_smooth episode npz.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp

FREE = list(range(6, 136))


def triples(frames):
    """Index triples (k, k+1, k+2) of the second difference inside each stretch."""
    out = [(k, k + 1, k + 2) for part in pp.stretches(frames) for k in range(part.start, part.stop - 2)]
    return torch.tensor(out, dtype=torch.long).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--margin-mm', type=float, default=1.0); ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--relin', type=int, default=10); ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--w-pen', type=float, default=100.0); ap.add_argument('--w-joint', type=float, default=1.0)
    ap.add_argument('--lam-t', type=float, default=1000.0); ap.add_argument('--gamma', type=float, default=1e-2)
    ap.add_argument('--band-mm', type=float, default=5.0)
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t = st['mhr_pose'].copy(), st['object_translation'].copy()
    s = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    frames, R, T = ep['frames'], ep['R'], len(pose)
    hand_ids = np.load(pp.HAND_IDS)
    geo = pp.Geometry(ep['verts'], ep['faces'])
    margin_cm, band = a.margin_mm / 10.0, a.band_mm / 1000.0
    v0, j0 = pp.decode(pose, ep['scales'], ep['shape'])
    pen0 = pp.pen_proxy(ep, v0[:, hand_ids], t, s)
    rep = {'episode': a.episode, 'frames': int(T), 'start': a.start, 'args': vars(a), 'pen_before': pen0}
    new_pose = pose.copy(); log = []
    if pen0[2] > 0:
        ts = torch.jit.load(pp.MHR_MODEL, map_location='cpu').eval()
        shape = torch.from_numpy(ep['shape'].reshape(1, 45).astype(np.float32)).expand(T, 45)
        scales = torch.from_numpy(ep['scales'].astype(np.float32)).expand(T, 68)
        zeros = torch.zeros(T, 72)
        base = torch.from_numpy(pose.astype(np.float32))
        hid = torch.from_numpy(hand_ids.astype(np.int64)); bj = torch.tensor(pp.BODY_J)
        flip = torch.tensor(pp.FLIP, dtype=torch.float32)
        Rt = torch.from_numpy(R.astype(np.float32)); tt = torch.from_numpy(t.astype(np.float32))
        tri = triples(frames)
        fit = torch.tensor(FREE)
        mask0 = torch.ones(T, 1); mask0[0] = 0.0                       # first scored frame stays exact
        delta = torch.zeros(T, len(FREE), requires_grad=True)
        opt = torch.optim.Adam([delta], lr=a.lr)

        def forward(d):
            full = base.index_copy(1, fit, base[:, fit] + d * mask0)
            v, k = ts(shape, torch.cat([full, scales], 1), zeros, True)
            hands = v[:, hid] / 100.0 * flip                            # scorer frame, metres
            return hands, k[:, bj, :3]                                   # joints in native cm

        with torch.no_grad():
            _, J0 = forward(torch.zeros_like(delta))
        D0 = J0[tri[:, 0]] - 2 * J0[tri[:, 1]] + J0[tri[:, 2]]

        def linearize(hands):
            h = hands.detach().double().numpy()
            local = np.einsum('tpi,tij->tpj', h - t[:, None], R) / s
            d, nl = geo.signed(local.reshape(-1, 3), band / s)
            d = (d.reshape(T, -1) * s) * 100.0                           # cm
            n = np.einsum('tij,tpj->tpi', R, nl.reshape(T, -1, 3))
            active = d < margin_cm + band * 100.0
            return (torch.from_numpy(np.where(np.isfinite(d), d, 1e3).astype(np.float32)),
                    torch.from_numpy(n.astype(np.float32)), hands.detach().clone(),
                    torch.from_numpy(active.astype(np.float32)), d)

        for step in range(a.steps):
            hands, J = forward(delta)
            if step % a.relin == 0:
                dl, nl, hl, act, dnp = linearize(hands)
                inside = int((dnp < 0).sum()); frames_in = int((dnp < 0).any(1).sum())
                log.append({'step': step, 'inside_points': inside, 'frames_inside': frames_in,
                            'max_depth_mm': float(max(0.0, -dnp.min()) * 10.0)})
                if not (dnp < margin_cm).any():
                    break
            sd = dl + ((hands - hl) * nl).sum(-1) * 100.0                # linearized signed distance, cm
            pen = (torch.relu(margin_cm - sd) ** 2 * act).sum()
            dev = ((J - J0) ** 2).sum()
            D = J[tri[:, 0]] - 2 * J[tri[:, 1]] + J[tri[:, 2]]
            acc = ((D - D0) ** 2).sum()
            loss = a.w_pen * pen + a.w_joint * dev + a.lam_t * acc + a.gamma * (delta ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        new_pose[:, FREE] = pose[:, FREE] + (delta.detach() * mask0).double().numpy()
    v1, j1 = pp.decode(new_pose, ep['scales'], ep['shape'])
    rep.update({'pen_after': pp.pen_proxy(ep, v1[:, hand_ids], t, s),
                'own_acc_h_before': pp.own_acc(j0[:, pp.BODY_J], frames),
                'own_acc_h_after': pp.own_acc(j1[:, pp.BODY_J], frames),
                'joint_change_cm_mean': float(np.linalg.norm(j1[:, pp.BODY_J] - j0[:, pp.BODY_J], axis=2).mean() * 100),
                'joint_change_cm_max': float(np.linalg.norm(j1[:, pp.BODY_J] - j0[:, pp.BODY_J], axis=2).max() * 100),
                'human_mean_vertex_change_cm': float(np.linalg.norm(v1 - v0, axis=2).mean() * 100),
                'human_max_vertex_change_cm': float(np.linalg.norm(v1 - v0, axis=2).max() * 100),
                'param_change_max': float(np.abs(new_pose - pose).max()),
                'human_first_frame_exact': bool(np.array_equal(new_pose[0], pose[0])),
                'own_acc_o': pp.own_acc(t, frames), 'seconds': time.time() - t0, 'log': log,
                'source_sha256': pp.sha(__file__), 'parquet_sha256': pp.sha(a.parquet)})
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=new_pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('log', 'args')}), flush=True)


if __name__ == '__main__':
    main()
