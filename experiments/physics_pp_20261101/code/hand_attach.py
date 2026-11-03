#!/usr/bin/env python3
"""Hand-attached object translation for one episode of a packed Track 1 submission.

Scored frames lie inside the hand-object contact span, so the object should move with the hand(s) holding it.
The anchor is the centroid of the contact hand's proxy vertices (both hands' midpoint when both are near the
object). The object translation becomes anchor + one constant offset per episode, either in the world frame
(`world`) or in the object's own frame through its estimated rotation (`local`); the offset is the median of
the original offsets, so no GT is used. An optional second-difference smoothing (frame 0 not pinned) follows.
The human pose comes from a human_smooth / physics_pp_v2 episode npz; rotation, scale, mesh are unchanged.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def two_hands(hand_ids, v_first):
    """Split the proxy hand vertices into two hands by 2-means on the first frame."""
    p = v_first[hand_ids]
    c = p[[np.argmin(p[:, 0]), np.argmax(p[:, 0])]]
    for _ in range(20):
        lab = np.linalg.norm(p[:, None] - c[None], axis=2).argmin(1)
        c = np.stack([p[lab == k].mean(0) for k in (0, 1)])
    return hand_ids[lab == 0], hand_ids[lab == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--human', required=True, help='episode npz whose mhr_pose is used (and kept)')
    ap.add_argument('--mode', choices=['world', 'local'], required=True)
    ap.add_argument('--lam', type=float, default=0.0); ap.add_argument('--near-cm', type=float, default=10.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    pose = np.load(a.human)['mhr_pose']
    frames, R, t_raw, s = ep['frames'], ep['R'], ep['t'], ep['s']
    v, _ = pp.decode(pose, ep['scales'], ep['shape'])
    left, right = two_hands(np.load(pp.HAND_IDS), v[0])
    geo = pp.Geometry(ep['verts'], ep['faces'])
    cents = np.stack([v[:, left].mean(1), v[:, right].mean(1)], 1)          # [T, 2, 3]
    dist = []
    for k in range(2):
        local = np.einsum('ti,tij->tj', cents[:, k] - t_raw, R) / s
        d, _ = geo.tree.query(local)                                        # unsigned, surface samples
        dist.append(float(np.median(d) * s * 100.0))
    near = [k for k in range(2) if dist[k] <= a.near_cm] or [int(np.argmin(dist))]
    anchor = cents[:, near].mean(1)
    if a.mode == 'world':
        off = np.median(t_raw - anchor, axis=0)
        t_new = anchor + off
        spread = np.linalg.norm((t_raw - anchor) - off, axis=1)
    else:
        g = np.median(np.einsum('tji,tj->ti', R, anchor - t_raw), axis=0)   # grasp point in object frame
        t_new = anchor - np.einsum('tij,j->ti', R, g)
        spread = np.linalg.norm(np.einsum('tji,tj->ti', R, anchor - t_raw) - g, axis=1)
    if a.lam > 0:
        t_new = pp.smooth_translation(t_new, frames, a.lam)
    hand_ids = np.load(pp.HAND_IDS)
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'mode': a.mode, 'lam': a.lam, 'human': a.human,
           'hand_object_median_distance_cm': dist, 'anchor_hands': near,
           'offset_spread_cm_median': float(np.median(spread) * 100), 'offset_spread_cm_p90': float(np.percentile(spread, 90) * 100),
           'own_acc_o_before': pp.own_acc(t_raw, frames), 'own_acc_o_after': pp.own_acc(t_new, frames),
           'translation_change_cm_mean': float(np.linalg.norm(t_new - t_raw, axis=1).mean() * 100),
           'translation_change_cm_max': float(np.linalg.norm(t_new - t_raw, axis=1).max() * 100),
           'pen_before': pp.pen_proxy(ep, v[:, hand_ids], t_raw, s),
           'pen_after': pp.pen_proxy(ep, v[:, hand_ids], t_new, s),
           'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__), 'parquet_sha256': pp.sha(a.parquet)}
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t_new, mhr_pose=pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
