#!/usr/bin/env python3
"""Zero proxy penetration with a smooth object trajectory, for one episode of a packed Track 1 submission.

pen_zero.py clears every proxy hand vertex, but its per-frame directions and unsmoothed last repair raise the
object's own ACC 10-100x (ep27 0.012 -> 1.00). Here:
  1. shrink: the object scale shrinks about its local bounding-box centre by the smallest 1% step (at most
     --max-shrink) that brings the deepest proxy hand vertex within --residual-mm of the surface, so hands
     gripping from opposite sides are cleared without pulling the object out of the grip;
  2. translate: the whole episode moves along one direction u by alpha_t >= 0, where alpha solves the bounded
     least squares  min w |D2 alpha|^2 + |alpha|^2  s.t.  alpha_t >= a_t  (D2 inside each scored stretch,
     scipy lsq_linear) and a_t is the exact clearing distance of frame t along u (doubling, then bisection,
     all frames at once). Frames that fail at their alpha (the clear set along u need not be an interval)
     raise their bound to the next clearing distance beyond it and the problem is re-solved. u is the
     candidate (14 axis/diagonal directions, +-camera ray, the median hand-to-object direction) with the
     smallest mean squared clearing distance on a subsample of penetrating frames.
The final check is the scorer's own penetration kernel on all 5,840 proxy vertices. One output per --w value.
Rotation, mesh and human are unchanged.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp
from pen_zero import DIRS


def frame_min(geo, hands, t, R, s, band=0.01):
    """Minimum signed distance (m) per frame of the hand points to the posed object (inf when none is near)."""
    local = np.einsum('tpi,tij->tpj', hands - t[:, None], R) / s
    d, _ = geo.signed(local.reshape(-1, 3), band / s)
    return d.reshape(local.shape[:2]).min(1) * s


def clear_dist(geo, hands, t, R, s, u, margin, lo, max_dist):
    """Per frame, a verified alpha >= lo (smallest up to bisection) with frame_min >= margin at t + alpha u."""
    ok = lambda al, i: frame_min(geo, hands[i], t[i] + al[:, None] * u, R[i], s) >= margin
    n = len(t); hi = np.full(n, np.inf); low = lo.copy()
    good = ok(lo, np.arange(n)); hi[good] = lo[good]
    todo = np.flatnonzero(~good); step = 0.005
    while todo.size and step <= 2 * max_dist:
        g = ok(lo[todo] + step, todo)
        hit = todo[g]; hi[hit] = lo[hit] + step
        if step > 0.005:
            low[hit] = lo[hit] + step / 2
        todo = todo[~g]; step *= 2
    idx = np.flatnonzero(np.isfinite(hi) & ~good)
    for _ in range(8):
        if not idx.size:
            break
        mid = 0.5 * (low[idx] + hi[idx]); g = ok(mid, idx)
        hi[idx[g]] = mid[g]; low[idx[~g]] = mid[~g]
    return hi


def smooth_alpha(lb, D2, w):
    """min w |D2 alpha|^2 + |alpha|^2 subject to alpha >= lb."""
    T = len(lb)
    A = np.vstack([np.sqrt(w) * D2.toarray(), np.eye(T)])
    res = lsq_linear(A, np.zeros(len(A)), bounds=(lb, np.full(T, np.inf)), method='bvls')
    return np.maximum(res.x, lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True, help='one subdir per w')
    ap.add_argument('--margin-mm', type=float, default=1.5); ap.add_argument('--max-cm', type=float, default=80.0)
    ap.add_argument('--max-shrink', type=float, default=0.05); ap.add_argument('--residual-mm', type=float, default=3.0)
    ap.add_argument('--w', default='1e2,1e4,1e6', help='colon or comma separated'); ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--subsample', type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t_start = st['mhr_pose'].copy(), st['object_translation'].copy()
    s0 = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    frames, R, T = ep['frames'], ep['R'], len(pose)
    hand_ids = np.load(pp.HAND_IDS)
    v, _ = pp.decode(pose, ep['scales'], ep['shape'])
    hands = v[:, hand_ids]
    geo = pp.Geometry(ep['verts'], ep['faces'])
    margin, max_dist = a.margin_mm / 1000.0, a.max_cm / 100.0
    centre = 0.5 * (ep['verts'].min(0) + ep['verts'].max(0))
    D2 = pp.d2_matrix(frames)
    rep = {'episode': a.episode, 'frames': int(T), 'start': a.start, 'args': vars(a),
           'pen_before': pp.pen_proxy(ep, hands, t_start, s0), 'own_acc_o_before': pp.own_acc(t_start, frames)}

    # 1. shrink about the local bounding-box centre
    shrink_log = []
    for k in range(int(round(a.max_shrink * 100)) + 1):
        f = 1.0 - 0.01 * k
        t1 = t_start + (1.0 - f) * s0 * np.einsum('tij,j->ti', R, centre); s1 = f * s0
        m = frame_min(geo, hands, t1, R, s1)
        depth = float(max(0.0, -m.min()))
        shrink_log.append({'shrink_percent': k, 'max_depth_mm': depth * 1000, 'frames_below_margin': int((m < margin).sum())})
        if depth * 1000 <= a.residual_mm:
            break
    rep.update({'shrink': shrink_log, 'object_scale_after': s1})

    # 2. one direction, smooth bounded alpha
    need = np.flatnonzero(m < margin)
    rep['frames_needing_translation'] = int(need.size)
    ray = np.mean(t1 / np.linalg.norm(t1, axis=1, keepdims=True), 0); ray /= np.linalg.norm(ray)
    results = {}
    if need.size:
        cents = s1 * np.einsum('tij,j->ti', R, centre) + t1
        away = np.median(cents[need] - hands[need].mean(1), 0); away /= max(np.linalg.norm(away), 1e-9)
        cand = list(DIRS) + [ray, -ray, away]
        sub = np.unique(np.concatenate([need[np.linspace(0, need.size - 1, min(a.subsample, need.size)).astype(int)],
                                        [need[np.argmin(m[need])]]]))
        scores = []
        for u in cand:
            d = clear_dist(geo, hands[sub], t1[sub], R[sub], s1, u, margin, np.zeros(sub.size), max_dist)
            scores.append(float(np.mean(d ** 2)) if np.isfinite(d).all() else np.inf)
        u = cand[int(np.argmin(scores))]
        rep.update({'direction': u.tolist(), 'direction_scores_cm2': [x * 1e4 for x in scores],
                    'direction_index': int(np.argmin(scores)), 'n_candidates': len(cand)})
        lb0 = np.zeros(T)
        lb0[need] = clear_dist(geo, hands[need], t1[need], R[need], s1, u, margin, np.zeros(need.size), max_dist)
        unsolved = np.flatnonzero(~np.isfinite(lb0)); lb0[unsolved] = 0.0
        rep['unsolved_frames'] = unsolved.tolist()
        rep['clear_dist_cm_p50_p90_max'] = [float(np.percentile(lb0[need], q) * 100) for q in (50, 90, 100)]
    for w in [float(x) for x in a.w.replace(':', ',').split(',')]:
        hist = []
        if need.size:
            lb = lb0 + 1e-5 * (lb0 > 0)
            for r in range(a.rounds):
                alpha = smooth_alpha(lb, D2, w)
                bad = np.flatnonzero(frame_min(geo, hands, t1 + alpha[:, None] * u, R, s1) < margin)
                hist.append({'round': r, 'frames_failing': int(bad.size)})
                if not bad.size:
                    break
                nb = clear_dist(geo, hands[bad], t1[bad], R[bad], s1, u, margin, alpha[bad], max_dist)
                fin = np.isfinite(nb)
                lb[bad[fin]] = nb[fin] + 1e-5
                if not fin.any():
                    break
            t_new = t1 + alpha[:, None] * u
        else:
            t_new = t1
        pen = pp.pen_proxy(ep, hands, t_new, s1)
        res = {'w': w, 'history': hist, 'pen_after': pen, 'own_acc_o_after': pp.own_acc(t_new, frames),
               'translation_change_cm_mean': float(np.linalg.norm(t_new - t_start, axis=1).mean() * 100),
               'translation_change_cm_max': float(np.linalg.norm(t_new - t_start, axis=1).max() * 100)}
        results[f'{w:g}'] = res
        out = Path(a.out) / f'w{w:g}' / 'episodes'; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t_new, mhr_pose=pose,
                 object_scale=np.array(s1))
        (out / f'episode_{a.episode:06d}.json').write_text(json.dumps({**rep, **res, 'object_scale_before': s0,
                                                                       'human_first_frame_exact': True}, indent=1))
    rep.update({'results': results, 'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__),
                'parquet_sha256': pp.sha(a.parquet)})
    print(json.dumps({k: v for k, v in rep.items() if k not in ('args', 'direction_scores_cm2')}), flush=True)


if __name__ == '__main__':
    main()
