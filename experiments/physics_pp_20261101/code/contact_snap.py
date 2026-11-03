#!/usr/bin/env python3
"""Per-episode depth correction of the object from hand contact, for one episode of a Track 1 submission.

Scored frames lie inside the hand-object contact span, yet in our files the closest hand is often 20-80 cm
from the object surface, which points at monocular depth/scale ambiguity of the object relative to the human.
One scalar per episode is chosen so that the closest proxy hand vertex sits `target` mm from the object
surface in the median scored frame:
  scale: t -> k t and s -> k s about the camera centre (the object's image projection is unchanged);
  depth: t -> t + k * t/|t| per frame (moves along the viewing ray, size unchanged).
Rotation, mesh, human and the temporal shape of the trajectory are unchanged. No GT is used.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def per_frame_min(geo, hands, t, R, s, exact, band=0.02):
    """Minimum signed distance (m) per frame of the hand points to the posed object."""
    local = np.einsum('tpi,tij->tpj', hands - t[:, None], R) / s
    if exact:
        d, _ = geo.signed(local.reshape(-1, 3), band / s)
        d = d.reshape(local.shape[:2])
        far = ~np.isfinite(d).any(1)
        if far.any():                                                   # no point near: unsigned distance
            dd, _ = geo.tree.query(local[far].reshape(-1, 3))
            d[far] = dd.reshape(far.sum(), -1)
    else:
        d = geo.coarse(local.reshape(-1, 3)).reshape(local.shape[:2])
        far = ~np.isfinite(d)
        if far.any():
            dd, _ = geo.tree.query(local[far])
            d[far] = dd
    return np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--human', required=True, help='episode npz supplying mhr_pose')
    ap.add_argument('--object', required=True, help='episode npz supplying object_translation')
    ap.add_argument('--mode', choices=['scale', 'depth'], required=True)
    ap.add_argument('--target-mm', type=float, default=3.0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    pose = np.load(a.human)['mhr_pose']
    ob = np.load(a.object)
    t_in = ob['object_translation']; s_in = float(ob['object_scale']) if 'object_scale' in ob.files else ep['s']
    frames, R = ep['frames'], ep['R']
    hand_ids = np.load(pp.HAND_IDS)
    v, _ = pp.decode(pose, ep['scales'], ep['shape'])
    hands = v[:, hand_ids]
    geo = pp.Geometry(ep['verts'], ep['faces'])
    sub = slice(None, None, max(1, len(frames) // 60))                  # about 60 frames for the search
    ray = t_in / np.linalg.norm(t_in, axis=1, keepdims=True)
    target = a.target_mm / 1000.0

    def apply(k):
        return (k * t_in, k * s_in) if a.mode == 'scale' else (t_in + k * ray, s_in)

    def cost(k, exact=False, sl=sub):
        t, s = apply(k)
        m = per_frame_min(geo, hands[sl], t[sl], R[sl], s, exact)
        return float(np.median(np.abs(m - target))), m

    grid = np.exp(np.linspace(np.log(0.4), np.log(2.5), 81)) if a.mode == 'scale' else np.linspace(-1.5, 1.5, 121)
    coarse = [cost(k)[0] for k in grid]
    k0 = grid[int(np.argmin(coarse))]
    step = (grid[1] / grid[0]) if a.mode == 'scale' else (grid[1] - grid[0])
    fine = np.exp(np.linspace(np.log(k0 / step ** 2), np.log(k0 * step ** 2), 21)) if a.mode == 'scale' else \
        np.linspace(k0 - 2 * step, k0 + 2 * step, 21)
    exact = [cost(k, exact=True)[0] for k in fine]
    k_best = float(fine[int(np.argmin(exact))])
    identity = 1.0 if a.mode == 'scale' else 0.0
    c_before, m_before = cost(identity, exact=True, sl=slice(None))
    c_after, m_after = cost(k_best, exact=True, sl=slice(None))
    t_new, s_new = apply(k_best)
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'mode': a.mode, 'k': k_best, 'target_mm': a.target_mm,
           'human': a.human, 'object': a.object,
           'median_abs_gap_cm_before': c_before * 100, 'median_abs_gap_cm_after': c_after * 100,
           'closest_hand_cm_before_p10_p50_p90': [float(np.percentile(m_before, q) * 100) for q in (10, 50, 90)],
           'closest_hand_cm_after_p10_p50_p90': [float(np.percentile(m_after, q) * 100) for q in (10, 50, 90)],
           'object_scale_before': s_in, 'object_scale_after': float(s_new),
           'translation_change_cm_mean': float(np.linalg.norm(t_new - t_in, axis=1).mean() * 100),
           'translation_change_cm_max': float(np.linalg.norm(t_new - t_in, axis=1).max() * 100),
           'own_acc_o_before': pp.own_acc(t_in, frames), 'own_acc_o_after': pp.own_acc(t_new, frames),
           'pen_before': pp.pen_proxy(ep, hands, t_in, s_in), 'pen_after': pp.pen_proxy(ep, hands, t_new, s_new),
           'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__), 'parquet_sha256': pp.sha(a.parquet)}
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t_new, mhr_pose=pose,
             object_scale=np.array(s_new))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
