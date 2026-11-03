#!/usr/bin/env python3
"""Global object size factor for one episode of a packed Track 1 submission.

Tests for a systematic bias in the estimated object size: object_scale -> k * object_scale about the object's
local bounding-box centre (translation compensated so the centre stays put), the same k for every episode.
Rotation, mesh and human are unchanged; the translation shift is (1 - k) s R c, so own ACC-O changes only through
the rotation of c (r18: k 0.95 raised hosted ACC-O 0.0740 -> 0.0755 where c is far from the origin or the
rotation flips); --comp-lam smooths that shift in time. --about camera instead scales translation and size
together about the camera centre (image projection unchanged), a global object depth test.
One output per --k value.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True, help='one subdir per k')
    ap.add_argument('--k', default='0.95:1.05', help='colon or comma separated')
    ap.add_argument('--about', choices=['centre', 'camera'], default='centre')
    ap.add_argument('--comp-lam', type=float, default=0.0, help='centre mode: smooth the compensating shift')
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
    centre = 0.5 * (ep['verts'].min(0) + ep['verts'].max(0))
    hands = pp.decode(pose, ep['scales'], ep['shape'])[0][:, np.load(pp.HAND_IDS)]
    rep = {'episode': a.episode, 'frames': int(len(frames)), 'start': a.start, 'object_scale_before': s,
           'centre_local_norm_m': float(np.linalg.norm(centre) * s), 'pen_before': pp.pen_proxy(ep, hands, t, s),
           'own_acc_o_before': pp.own_acc(t, frames)}
    for k in [float(x) for x in a.k.replace(':', ',').split(',')]:
        if a.about == 'camera':
            t_new = k * t
        else:
            shift = (1.0 - k) * s * np.einsum('tij,j->ti', R, centre)
            t_new = t + (pp.smooth_translation(shift, frames, a.comp_lam) if a.comp_lam > 0 else shift)
        res = {'k': k, 'about': a.about, 'comp_lam': a.comp_lam,'object_scale_after': k * s, 'pen_after': pp.pen_proxy(ep, hands, t_new, k * s),
               'own_acc_o_after': pp.own_acc(t_new, frames),
               'translation_change_cm_mean': float(np.linalg.norm(t_new - t, axis=1).mean() * 100)}
        out = Path(a.out) / f'k{k:g}' / 'episodes'; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t_new, mhr_pose=pose,
                 object_scale=np.array(k * s), **rot)
        (out / f'episode_{a.episode:06d}.json').write_text(json.dumps({**rep, **res, 'human_first_frame_exact': True,
                                                                       'seconds': time.time() - t0}, indent=1))
        print(json.dumps({**rep, **res}), flush=True)


if __name__ == '__main__':
    main()
