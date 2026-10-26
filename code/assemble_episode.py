#!/usr/bin/env python3
"""S7 v2: assemble one episode's packer input (kit tools/pack_reconstruction.py contract).

  <out>/episode_XXXXXX.npz : pose[T,136], scales[68], shape[45] (S4b), object_rotation[T,3,3],
                              object_translation[T,3], object_scale () = 1.0
  <out>/episode_XXXXXX_object.ply : S5 mesh (object frame, metres) or a fallback mesh
Frame: OpenCV camera frame, metres, for human (host decode diag(1,-1,-1)@MHR/100) and object.

Object trajectory post-processing (validator 03 must-fix 4, 6, 7), all recorded in ASSEMBLE_XXXXXX.json:
  * the first frame of every MetaV3RSE window (i % --window) has its translation interpolated (depth outlier);
  * static spans: runs of >= --static-min frames that stay at their ANCHOR (mask-box IoU vs the span's first frame
    >= --static-box-iou and tracked translation within --static-move-m) get ONE pose: median translation and the
    rotation of the span frame with the best S6 IoU; the hold is applied BEFORE smoothing;
  * translation is then Gaussian-smoothed (--t-sigma, default 3; 5 when S6 mean IoU < --low-iou) and rotation is
    smoothed only inside segments split at steps > --rot-split-deg (default 45), so symmetric flips are not averaged;
  * fallbacks: no S6 -> S5 keyframe pose held constant; no S5 mesh -> 0.2 m cube at the subject pelvis (flagged).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def R_to_q(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def q_to_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


def smooth(x, sigma):
    if sigma <= 0 or len(x) < 2:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    return sum(wk * pad[j:j + len(x)] for j, wk in enumerate(k))


def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mask_box(path, size=(256, 192)):
    m = np.asarray(Image.open(path).resize(size, Image.NEAREST)) > 127
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def static_spans(masks, ok, t, min_len, box_thr, move_m):
    """Spans where the object stays at its ANCHOR (first span frame): mask box IoU vs the anchor >= box_thr AND the
    tracked translation within move_m of the anchor (validator 04: chained consecutive IoU accepted slow drift)."""
    T = len(masks)
    boxes = [mask_box(m["mask"]) if ok[i] else None for i, m in enumerate(masks)]
    spans, i = [], 0
    while i < T:
        if boxes[i] is None:
            i += 1
            continue
        j = i
        while (j + 1 < T and boxes[j + 1] is not None and box_iou(boxes[i], boxes[j + 1]) >= box_thr
               and np.linalg.norm(t[j + 1] - t[i]) <= move_m):
            j += 1
        if j - i + 1 >= min_len:
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--track", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--s3db", type=Path, required=True, help="for the pelvis fallback")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t-sigma", type=float, default=3.0)
    ap.add_argument("--r-sigma", type=float, default=2.0)
    ap.add_argument("--rot-split-deg", type=float, default=45.0)
    ap.add_argument("--static-min", type=int, default=10)
    ap.add_argument("--static-box-iou", type=float, default=0.8)
    ap.add_argument("--static-move-m", type=float, default=0.03)
    ap.add_argument("--window", type=int, default=16, help="MetaV3RSE forward window; first frame of each is a depth outlier")
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--low-iou", type=float, default=0.45, help="S6 mean IoU below this -> low-confidence track")
    ap.add_argument("--far-limit-m", type=float, default=2.0, help="ABSOLUTE object-pelvis distance limit (validator 05)")
    ap.add_argument("--held-max-diam", type=float, default=0.6, help="hand-depth repair only for objects smaller than this")
    ap.add_argument("--no-hand-depth-repair", action="store_true")
    ap.add_argument("--s3-human", dest="human_s3", type=Path, default=None, help="MetaV3RSE s3_human dir (depth maps)")
    ap.add_argument("--depth-scale", type=Path, default=None, help="depth_scale npz (anchoring)")
    args = ap.parse_args()

    h = np.load(args.human / "mhr_params.npz")
    T = len(h["pose"])
    e = f"episode_{args.episode:06d}"
    args.out.mkdir(parents=True, exist_ok=True)
    rep = dict(episode=args.episode, frames=T)
    mesh_src = args.object / "object_mesh.ply"
    if (args.track / "poses.npz").exists():
        tr = np.load(args.track / "poses.npz")
        P = tr["T_obj2cam"].astype(np.float64)
        trk_iou = tr["iou"]
        rep["object_source"] = "s6_foundationpose"
        tj = json.loads((args.track / "TRACK.json").read_text()) if (args.track / "TRACK.json").exists() else {}
        miou = (tj.get("iou_on_confident_masks") or {}).get("mean")
        rep["s6_mean_iou"] = miou
        if miou is not None and miou < args.low_iou:      # validator 04 re-verification: eps 1/2 unreliable tracks
            rep["low_confidence_track"] = True
            args.t_sigma, args.r_sigma = max(args.t_sigma, 5.0), max(args.r_sigma, 4.0)
    elif (args.object / "keyframe.json").exists():
        kf = json.loads((args.object / "keyframe.json").read_text())
        P = np.repeat(np.eye(4)[None], T, 0)
        P[:, :3, :3], P[:, :3, 3] = np.array(kf["R_obj2cam"]), np.array(kf["t_obj2cam"])
        trk_iou = np.full(T, np.nan)
        rep["object_source"] = "s5_keyframe_constant_FALLBACK"
    else:
        import trimesh
        pel = np.load(args.s3db / "sam3d_body.npz")["joints_cam"][:, 1]   # joint 1 = pelvis (0 is the floor-level root)
        P = np.repeat(np.eye(4)[None], T, 0)
        P[:, :3, 3] = np.median(pel, 0)
        trk_iou = np.full(T, np.nan)
        mesh_src = args.out / f"{e}_fallback_cube.ply"
        trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(mesh_src)
        rep["object_source"] = "pelvis_cube_FALLBACK"
    if len(P) != T:
        raise SystemExit(f"human has {T} frames, object {len(P)}")

    ok_pose = np.isfinite(P).all(axis=(1, 2))
    # validator 05 Q1: (a) small held objects whose tracked depth disagrees with the holding hand are moved along their
    # camera ray to the hand depth (image position kept); (b) ABSOLUTE far-from-pelvis guard (the old relative limit
    # max(2, 3*median) grew with biased tracks: 4.49 m ep8, 5.39 m ep18) -> treat as missing, interpolate.
    import trimesh as _tm
    far = np.zeros(T, bool)
    repaired = np.zeros(T, bool)
    sb_path = args.s3db / "sam3d_body.npz"
    if sb_path.exists() and ok_pose.any():
        sb = np.load(sb_path)
        Vb, kp2, Kb = sb["vertices_cam"], sb["keypoints_2d"], sb["K"]
        pel = sb["joints_cam"][:, 1]
        diam = float(np.max(np.ptp(np.asarray(_tm.load(mesh_src, process=False).vertices), axis=0)))
        rep["object_diameter_m"] = diam
        masks_rows = json.loads((args.masks / "MASKS.json").read_text())["rows"]
        dz_off = np.zeros(T)
        if not args.no_hand_depth_repair and diam < args.held_max_diam:
            for i in np.flatnonzero(ok_pose):
                t_i = P[i, :3, 3]
                if t_i[2] <= 0.05:
                    continue
                mr = masks_rows[i] if i < len(masks_rows) else {}
                if mr.get("selected") and mr.get("centroid"):
                    uv = np.array(mr["centroid"], float)
                else:
                    uv = Kb[:2, :2] @ (t_i[:2] / t_i[2]) + Kb[:2, 2]
                r_px = max(40.0, 0.75 * Kb[0, 0] * (diam / 2) / t_i[2])
                best = None
                for w in (41, 62):                       # MHR70 right / left wrist
                    d2 = np.linalg.norm(kp2[i, w] - uv)
                    if d2 < r_px and (best is None or d2 < best[0]):
                        best = (d2, w)
                if best is None:
                    continue
                vb = Vb[i]
                uvb = vb[:, :2] / np.clip(vb[:, 2:3], 1e-3, None) * Kb[0, 0] + Kb[:2, 2]
                near = np.linalg.norm(uvb - kp2[i, best[1]], axis=1) < 15
                if near.sum() < 5:
                    continue
                z_h = float(np.min(vb[near, 2]) + 0.02)    # visible hand surface, slight inset
                tol = 0.25 + diam / 2
                dz = z_h - t_i[2]
                # validator 06 R1: only PULL objects toward the camera (dz < 0). Pushing back used the wrong (hanging) hand
                # on ep3 and moved a correctly placed bowl 24-49 cm behind the visible surface.
                if dz < -tol:
                    dz_off[i] = dz                                  # full offset to the hand depth; smoothing below keeps it continuous
                    repaired[i] = True
            # continuous correction: smooth the per-frame depth offsets (sigma 5) so repaired and untouched frames do not
            # alternate (that toggling raised ACC-O on eps 3/17 in the first v2 build), then move along each frame's ray
            dz_s = smooth(dz_off[:, None], 5.0)[:, 0]
            for i in np.flatnonzero(ok_pose & (np.abs(dz_s) > 1e-4)):
                t_i = P[i, :3, 3]
                if t_i[2] > 0.05:
                    P[i, :3, 3] = t_i * ((t_i[2] + dz_s[i]) / t_i[2])
        dist = np.linalg.norm(P[:, :3, 3] - pel, axis=1)
        far = ok_pose & (dist > args.far_limit_m)
        # far, not held: re-depth along the ray to the FRONT-most anchored depth inside the mask (10th percentile); thin
        # objects' masks at 512x384 pick up background depth, which pushed e.g. the ep18 roller to 5 m
        mask_redepth = np.zeros(T, bool)
        if args.human_s3 is not None and args.depth_scale is not None and far.any():
            ds = np.load(args.depth_scale)
            sc = ds["scale"]
            of = ds["offset"] if "offset" in ds.files else np.zeros_like(sc)
            for i in np.flatnonzero(far):
                mr = masks_rows[i] if i < len(masks_rows) else {}
                f_npz = args.human_s3 / f"{i:06d}.npz"
                if not (mr.get("selected") and f_npz.exists()):
                    continue
                d = np.load(f_npz)["depth"].astype(np.float32) * float(sc[i]) + float(of[i])
                dh, dw = d.shape
                m = np.asarray(Image.open(mr["mask"]).convert("L").resize((dw, dh), Image.BILINEAR)) > 76
                vals = d[m & (d > 0.05)]
                if len(vals) < 3:
                    continue
                z = float(np.percentile(vals, 10)) + diam / 4
                t_i = P[i, :3, 3]
                if t_i[2] > 0.05:
                    P[i, :3, 3] = t_i * (z / t_i[2])
                    mask_redepth[i] = True
            dist = np.linalg.norm(P[:, :3, 3] - pel, axis=1)
            far = ok_pose & (dist > args.far_limit_m)
        rep["mask_redepth_frames"] = int(mask_redepth.sum())
        ok_pose = ok_pose & ~far
        rep["hand_depth_repaired_frames"] = int(repaired.sum())
        rep["far_from_subject_frames"] = int(far.sum())
        rep["far_limit_m"] = args.far_limit_m
    idx = np.flatnonzero(ok_pose)
    t = P[:, :3, 3].copy()
    for c in range(3):
        t[~ok_pose, c] = np.interp(np.flatnonzero(~ok_pose), idx, t[ok_pose, c])
    R = P[:, :3, :3].copy()
    for i in np.flatnonzero(~ok_pose):
        R[i] = P[idx[np.argmin(np.abs(idx - i))], :3, :3]

    min_score = args.min_score
    if (args.object / "keyframe.json").exists():      # validator 04 #8: honour S5's relaxed threshold
        min_score = min(min_score, json.loads((args.object / "keyframe.json").read_text()).get("keyframe_min_score", min_score))
    masks = json.loads((args.masks / "MASKS.json").read_text())["rows"]
    conf = np.array([bool(m.get("selected")) and m["score"] >= min_score for m in masks])

    # validator 04 #5: the first frame of every MetaV3RSE window is a 3-6 cm depth outlier -> interpolate it
    win_fix = [i for i in range(args.window, T - 1, args.window)]
    for i in win_fix:
        t[i] = 0.5 * (t[i - 1] + t[i + 1])
    spans = static_spans(masks, conf, t, args.static_min, args.static_box_iou, args.static_move_m)
    in_span = np.zeros(T, bool)
    t_h = t.copy()
    for a, b in spans:
        in_span[a:b + 1] = True
        t_h[a:b + 1] = np.median(t[a:b + 1], 0)
    # validator 04 #6: hold first, THEN smooth, so span boundaries do not step
    t_s = smooth(t_h, args.t_sigma)
    R_s = R.copy()
    # rotation: smooth inside segments split at large steps
    steps = np.array([geodesic_deg(R[i - 1], R[i]) for i in range(1, T)])
    cuts = [0] + [i for i in range(1, T) if steps[i - 1] > args.rot_split_deg] + [T]
    for a, b in zip(cuts[:-1], cuts[1:]):
        q = R_to_q(R[a:b])
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        qs = smooth(q, args.r_sigma)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        R_s[a:b] = q_to_R(qs)
    for a, b in spans:
        ii = np.arange(a, b + 1)
        best = ii[np.nanargmax(trk_iou[ii])] if np.isfinite(trk_iou[ii]).any() else ii[len(ii) // 2]
        R_s[a:b + 1] = R[best]

    np.savez_compressed(args.out / f"{e}.npz", pose=h["pose"].astype(np.float32), scales=h["scales"].astype(np.float32),
                        shape=h["shape"].astype(np.float32), object_rotation=R_s.astype(np.float64),
                        object_translation=t_s.astype(np.float64), object_scale=np.float64(1.0))
    shutil.copyfile(mesh_src, args.out / f"{e}_object.ply")
    acc = np.linalg.norm(t_s[2:] - 2 * t_s[1:-1] + t_s[:-2], axis=1) * 100
    rep.update(invalid_object_frames=int((~ok_pose).sum()), window_first_frames_interpolated=len(win_fix),
               static_frac=float(in_span.mean()), mask_min_score=min_score, t_sigma=args.t_sigma, r_sigma=args.r_sigma,
               rot_split_deg=args.rot_split_deg, rot_segments=len(cuts) - 1, static_spans=spans,
               static_frames=int(in_span.sum()), smoothing_shift_cm=float(np.linalg.norm(t_s - t, axis=1).mean() * 100),
               object_acc_cm_per_f2=dict(mean=float(acc.mean()), p95=float(np.percentile(acc, 95))),
               object_depth_m=dict(median=float(np.median(t_s[:, 2])), min=float(t_s[:, 2].min()), max=float(t_s[:, 2].max())))
    (args.out / f"ASSEMBLE_{args.episode:06d}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "static_spans"}), flush=True)


if __name__ == "__main__":
    main()
