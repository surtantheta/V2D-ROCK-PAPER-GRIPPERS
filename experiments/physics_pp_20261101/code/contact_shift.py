#!/usr/bin/env python3
"""Constant 3D object shift that closes a large hand-object gap, for one episode of a packed Track 1 submission.

Scored frames lie inside the hand-object contact span, yet in some episodes the closest proxy hand vertex is a
median 10-70 cm from our object. contact_snap.py could only move the object along the camera ray (r14: worse), which
cannot close a lateral offset. Here, per scored frame, the closest proxy hand vertex p_t and its nearest object
surface sample q_t give the gap vector p_t - q_t; when the median gap exceeds --min-gap-cm, the whole episode
translation moves by the component-wise median gap vector, shortened by --target-mm. Otherwise the episode is
copied unchanged. A constant shift adds no acceleration. Rotation, size, mesh and human are unchanged; no GT used.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True, help='one subdir per gate')
    ap.add_argument('--min-gap-cm', default='5:10', help='colon or comma separated gates')
    ap.add_argument('--target-mm', type=float, default=10.0)
    a = ap.parse_args()
    t0 = time.time()
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t = st['mhr_pose'], st['object_translation']
    s = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    rot = {'object_rotation': st['object_rotation']} if 'object_rotation' in st.files else {}
    if rot:
        ep['R'] = rot['object_rotation']
    frames, R = ep['frames'], ep['R']
    hands = pp.decode(pose, ep['scales'], ep['shape'])[0][:, np.load(pp.HAND_IDS)]
    geo = pp.Geometry(ep['verts'], ep['faces'])
    local = np.einsum('tpi,tij->tpj', hands - t[:, None], R) / s
    d, idx = geo.tree.query(local.reshape(-1, 3))
    d, idx = d.reshape(local.shape[:2]), idx.reshape(local.shape[:2])
    k = d.argmin(1); rows = np.arange(len(frames))
    p = hands[rows, k]                                                    # closest hand vertex, world
    q = s * np.einsum('tij,tj->ti', R, geo.tree.data[idx[rows, k]]) + t  # its nearest surface sample, world
    gap = p - q
    gap_cm = np.linalg.norm(gap, axis=1) * 100
    med = np.median(gap, axis=0); norm = np.linalg.norm(med)
    shift = med * max(0.0, 1.0 - a.target_mm / 1000.0 / max(norm, 1e-9))
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'start': a.start,
           'gap_cm_p10_p50_p90': [float(np.percentile(gap_cm, x)) for x in (10, 50, 90)],
           'median_gap_vector_cm': (med * 100).tolist(), 'shift_cm': float(np.linalg.norm(shift) * 100),
           'pen_before': pp.pen_proxy(ep, hands, t, s), 'own_acc_o_before': pp.own_acc(t, frames)}
    for g in [float(x) for x in a.min_gap_cm.replace(':', ',').split(',')]:
        apply = float(np.median(gap_cm)) > g
        t_new = t + shift if apply else t
        res = {'min_gap_cm': g, 'applied': bool(apply), 'pen_after': pp.pen_proxy(ep, hands, t_new, s),
               'own_acc_o_after': pp.own_acc(t_new, frames),
               'translation_change_cm_mean': float(np.linalg.norm(t_new - t, axis=1).mean() * 100)}
        out = Path(a.out) / f'g{g:g}' / 'episodes'; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t_new, mhr_pose=pose,
                 object_scale=np.array(s), **rot)
        (out / f'episode_{a.episode:06d}.json').write_text(json.dumps({**rep, **res, 'human_first_frame_exact': True,
                                                                       'seconds': time.time() - t0}, indent=1))
        print(json.dumps({**rep, **res}), flush=True)


if __name__ == '__main__':
    main()
