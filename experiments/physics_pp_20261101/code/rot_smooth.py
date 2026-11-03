#!/usr/bin/env python3
"""Temporal smoothing of the object rotation for one episode of a packed Track 1 submission.

Object translation smoothing alone moved CD-O 36.53 -> 35.13; rotation was never smoothed. Unit quaternions are
sign-aligned within each scored stretch, smoothed with the same second-difference penalty (I + lam D2^T D2)^-1
and renormalized. Stretches are also split at frame-to-frame jumps above --break-deg (e.g. flips of a symmetric
object), so no rotation is averaged across a flip. Translation, scale and human come from --start unchanged.
ACC-O reads translation only, so it is unaffected; CD-O (posed surface samples) and PEN can change.
One output per --lam value.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import physics_pp_v2 as pp


def rel_angles(R):
    """World-frame angular velocity rotvecs between consecutive rotations."""
    return Rotation.from_matrix(R[1:] @ np.swapaxes(R[:-1], 1, 2)).as_rotvec()


def segments(frames, R, break_deg):
    out = []
    for part in pp.stretches(frames):
        idx = np.arange(part.start, part.stop)
        cuts = np.flatnonzero(np.degrees(np.linalg.norm(rel_angles(R[idx]), axis=1)) > break_deg) + 1 if len(idx) > 1 else []
        out += np.split(idx, cuts)
    return out


def ang_acc_deg(R, segs):
    vals = [np.linalg.norm(np.diff(rel_angles(R[s]), axis=0), axis=1) for s in segs if len(s) >= 3]
    return float(np.degrees(np.concatenate(vals)).mean()) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--start', required=True); ap.add_argument('--out', required=True, help='one subdir per lam')
    ap.add_argument('--lam', default='1e3:3e4:3e5', help='colon or comma separated')
    ap.add_argument('--break-deg', type=float, default=45.0)
    a = ap.parse_args()
    t0 = time.time()
    ep = pp.load_episode(a.parquet, a.episode)
    st = np.load(a.start)
    pose, t = st['mhr_pose'], st['object_translation']
    s = float(st['object_scale']) if 'object_scale' in st.files else ep['s']
    R0 = st['object_rotation'] if 'object_rotation' in st.files else ep['R']
    frames, T = ep['frames'], len(pose)
    segs = segments(frames, R0, a.break_deg)
    q = Rotation.from_matrix(R0).as_quat()
    rows, cols, vals, r = [], [], [], 0
    for sg in segs:
        for k in range(1, len(sg)):
            if q[sg[k]] @ q[sg[k - 1]] < 0:
                q[sg[k]] *= -1.0
        for k in range(len(sg) - 2):
            rows += [r, r, r]; cols += [sg[k], sg[k + 1], sg[k + 2]]; vals += [1.0, -2.0, 1.0]; r += 1
    D = sparse.csr_matrix((vals, (rows, cols)), shape=(r, T))
    hands = pp.decode(pose, ep['scales'], ep['shape'])[0][:, np.load(pp.HAND_IDS)]
    rep = {'episode': a.episode, 'frames': int(T), 'start': a.start, 'args': vars(a), 'segments': len(segs),
           'flip_breaks': len(segs) - len(pp.stretches(frames)), 'ang_acc_deg_before': ang_acc_deg(R0, segs),
           'pen_before': pp.pen_proxy(dict(ep, R=R0), hands, t, s), 'own_acc_o_after': pp.own_acc(t, frames)}
    results = {}
    for lam in [float(x) for x in a.lam.replace(':', ',').split(',')]:
        qs = np.asarray(spsolve((sparse.identity(T) + lam * (D.T @ D)).tocsc(), q)).reshape(T, 4)
        Rs = Rotation.from_quat(qs / np.linalg.norm(qs, axis=1, keepdims=True)).as_matrix()
        change = np.degrees(np.linalg.norm(Rotation.from_matrix(Rs @ np.swapaxes(R0, 1, 2)).as_rotvec(), axis=1))
        res = {'lam': lam, 'ang_acc_deg_after': ang_acc_deg(Rs, segs), 'rotation_change_deg_mean': float(change.mean()),
               'rotation_change_deg_max': float(change.max()), 'pen_after': pp.pen_proxy(dict(ep, R=Rs), hands, t, s)}
        results[f'{lam:g}'] = res
        out = Path(a.out) / f'l{lam:g}' / 'episodes'; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f'episode_{a.episode:06d}.npz', frames=frames, object_translation=t, mhr_pose=pose,
                 object_scale=np.array(s), object_rotation=Rs)
        (out / f'episode_{a.episode:06d}.json').write_text(json.dumps({**rep, **res, 'human_first_frame_exact': True}, indent=1))
    rep.update({'results': results, 'seconds': time.time() - t0, 'source_sha256': pp.sha(__file__),
                'parquet_sha256': pp.sha(a.parquet)})
    print(json.dumps({k: v for k, v in rep.items() if k != 'args'}), flush=True)


if __name__ == '__main__':
    main()
