#!/usr/bin/env python3
"""Capped wrist/finger parameter smoothing on top of a finished human stage, for one episode.

human_smooth.py --lam-hand smooths the hand angles per channel. Where one joint rotation flips between Euler
representations, that bends the hand: 3% of frames moved a hand vertex > 5 cm (up to 19 cm). Here the body,
object and first-frame handling of the start npz are kept exactly; each hand angle moves by its smoothed change
(unwrapped, second-difference penalty) only where that change is at most --cap-rad, and keeps its raw value
elsewhere. One output per (lam, cap) pair. No GT used.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp
from hand_clear import FREE as HAND_FREE
from human_smooth import smooth_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True, help='one subdir per lam/cap')
    ap.add_argument('--lam', default='100', help='colon or comma separated')
    ap.add_argument('--cap-rad', default='0.25', help='colon or comma separated')
    ap.add_argument('--free-first', action='store_true', help='let the first scored frame move too')
    a = ap.parse_args()
    t0 = time.time()
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t = st['mhr_pose'], st['object_translation']
    s = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    rot = {'object_rotation': st['object_rotation']} if 'object_rotation' in st.files else {}
    frames = ep['frames']
    hid = np.load(pp.HAND_IDS)
    raw = pose[:, HAND_FREE]
    hp = np.unwrap(raw, axis=0)
    v0 = pp.decode(pose, ep['scales'], ep['shape'])[0][:, hid]
    for lam in [float(x) for x in a.lam.replace(':', ',').split(',')]:
        d = smooth_targets(hp, frames, lam, pin_first=not a.free_first) - hp   # real angular change per entry
        for cap in [float(x) for x in a.cap_rad.replace(':', ',').split(',')]:
            keep = np.abs(d) <= cap
            new = pose.copy(); new[:, HAND_FREE] = raw + np.where(keep, d, 0.0)
            v1 = pp.decode(new, ep['scales'], ep['shape'])[0][:, hid]
            mv = np.linalg.norm(v1 - v0, axis=2).max(1) * 100
            res = {'episode': a.episode, 'frames': int(len(frames)), 'start': a.start, 'lam': lam, 'cap_rad': cap,
                   'free_first': a.free_first, 'frac_entries_kept': float(keep.mean()),
                   'own_acc_hand_vertices_before': pp.own_acc(v0, frames),
                   'own_acc_hand_vertices_after': pp.own_acc(v1, frames),
                   'frame_max_hand_move_cm_p50_p99_max': [float(np.percentile(mv, q)) for q in (50, 99, 100)],
                   'non_hand_identical': bool(np.array_equal(np.delete(new, HAND_FREE, 1), np.delete(pose, HAND_FREE, 1))),
                   'human_first_frame_exact': bool(np.array_equal(new[0], pose[0])),
                   'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__)}
            out = Path(a.out) / f'l{lam:g}_c{cap:g}' / 'episodes'; out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=new,
                     object_scale=np.array(s), **rot)
            (out / f'episode_{a.episode:06d}.json').write_text(json.dumps(res, indent=1))
            print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
