#!/usr/bin/env python3
"""Smallest object translation that leaves every proxy hand vertex outside the object, per scored frame.

For each frame with any hand vertex inside (or within the margin of) the episode's object mesh, candidate
directions (14 axis/diagonal unit vectors, the direction from the hand centroid to the object centre and
the previous frame's choice) are searched: for each, the smallest clearing distance is bracketed by
doubling and refined by bisection with the exact winding-number / closest-point signed distance, and the
direction with the smallest distance wins. Corrections are then smoothed in time, and frames that the
smoothing made infeasible are re-solved from the smoothed position until every frame passes the scorer's
own penetration kernel on all 5,840 proxy vertices. Only object translation changes.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp

DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] +
                [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)


def min_signed(geo, hands_f, t_f, R_f, s, band):
    """Minimum signed distance (m) of one frame's hand points to the posed object (inf if none near)."""
    local = (hands_f - t_f) @ R_f / s
    d, _ = geo.signed(local, band / s)
    return float(np.min(d) * s)


def clear_distance(geo, hands_f, t_f, R_f, s, u, margin, band, max_dist):
    """Smallest alpha with min signed distance >= margin after moving the object by alpha * u."""
    ok = lambda a: min_signed(geo, hands_f, t_f + a * u, R_f, s, band) >= margin
    if ok(0.0):
        return 0.0
    hi = 0.005
    while not ok(hi):
        hi *= 2.0
        if hi > max_dist:
            return np.inf
    lo = hi / 2.0 if hi > 0.005 else 0.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def solve_frame(geo, hands_f, t_f, R_f, s, margin, band, max_dist, prev_u, centre_world):
    cand = list(DIRS)
    away = centre_world - hands_f.mean(0)
    if np.linalg.norm(away) > 1e-6:
        cand.append(away / np.linalg.norm(away))
    if prev_u is not None:
        cand.append(prev_u)
    best = (np.inf, None)
    for u in cand:
        a = clear_distance(geo, hands_f, t_f, R_f, s, u, margin, band, min(max_dist, best[0]))
        if prev_u is not None and u is prev_u:
            a *= 0.9                                                   # mild preference for continuity
        if a < best[0]:
            best = (a, u)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--margin-mm', type=float, default=1.5); ap.add_argument('--max-cm', type=float, default=60.0)
    ap.add_argument('--lam', type=float, default=100.0); ap.add_argument('--rounds', type=int, default=6)
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t_start = st['mhr_pose'].copy(), st['object_translation'].copy()
    s = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    frames, R, T = ep['frames'], ep['R'], len(pose)
    hand_ids = np.load(pp.HAND_IDS)
    v, _ = pp.decode(pose, ep['scales'], ep['shape'])
    hands = v[:, hand_ids]
    geo = pp.Geometry(ep['verts'], ep['faces'])
    margin, band, max_dist = a.margin_mm / 1000.0, 0.01, a.max_cm / 100.0
    centre_local = 0.5 * (ep['verts'].min(0) + ep['verts'].max(0))
    pen0 = pp.pen_proxy(ep, hands, t_start, s)

    def solve(t_base, todo, prev=None):
        corr, unsolved = np.zeros((T, 3)), []
        for i in todo:
            centre = s * R[i] @ centre_local + t_base[i]
            alpha, u = solve_frame(geo, hands[i], t_base[i], R[i], s, margin, band, max_dist,
                                   None if prev is None else prev.get(i - 1), centre)
            if np.isfinite(alpha):
                corr[i] = alpha * u
                if prev is not None:
                    prev[i] = u
            else:
                unsolved.append(int(i))
        return corr, unsolved

    need = [i for i in range(T) if min_signed(geo, hands[i], t_start[i], R[i], s, band) < margin]
    dirs = {}
    corr, unsolved = solve(t_start, need, dirs)
    t = t_start + (pp.smooth_translation(corr, frames, a.lam) if a.lam > 0 else corr)
    history = [{'round': 0, 'frames_needing': len(need), 'unsolved': len(unsolved)}]
    for r in range(1, a.rounds + 1):
        bad = [i for i in range(T) if min_signed(geo, hands[i], t[i], R[i], s, band) < margin]
        history.append({'round': r, 'frames_failing_after_smoothing': len(bad)})
        if not bad:
            break
        fix, _ = solve(t, bad)
        t = t + (pp.smooth_translation(fix, frames, a.lam / (2 ** r)) if r < a.rounds else fix)
    pen1 = pp.pen_proxy(ep, hands, t, s)
    rep = {'episode': a.episode, 'frames': int(T), 'start': a.start, 'args': vars(a), 'pen_before': pen0, 'pen_after': pen1,
           'history': history, 'unsolved_frames': unsolved,
           'own_acc_o_before': pp.own_acc(t_start, frames), 'own_acc_o_after': pp.own_acc(t, frames),
           'translation_change_cm_mean': float(np.linalg.norm(t - t_start, axis=1).mean() * 100),
           'translation_change_cm_max': float(np.linalg.norm(t - t_start, axis=1).max() * 100),
           'frames_moved': int((np.linalg.norm(t - t_start, axis=1) > 1e-4).sum()),
           'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__), 'parquet_sha256': pp.sha(a.parquet)}
    np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=pose,
             object_scale=np.array(s))
    (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'args'}), flush=True)


if __name__ == '__main__':
    main()
