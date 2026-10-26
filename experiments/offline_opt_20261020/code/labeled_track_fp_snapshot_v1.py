#!/usr/bin/env python3
"""S6: object 6-DoF trajectory for one V2D episode with FoundationPose (CARI4D-vendored estimater.py).

Runs at the MetaV3RSE model resolution (512x384), where the camera-frame depth lives, with K scaled accordingly.
  1. register() at the S5 keyframe with the SAM3 mask; its pose is compared with the SAM3D keyframe pose.
  2. track_one() forward to the last frame and backward to frame 0, starting from the keyframe pose.
  3. Every frame: silhouette IoU of the rendered pose vs the SAM3 mask. If IoU < --reacquire-iou on a frame with a
     confident non-empty mask, register() again there (re-acquisition) and continue tracking from it.
Output (OpenCV camera frame, metres): poses.npz with T_obj2cam[T,4,4], iou[T], mask_ok[T], events list in TRACK.json.
Must be run with cwd = CARI4D root (estimater.py resolves its weights relative to the code dir).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sam3d_mesh_contract_snapshot_v1 import raster_iou   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True, help="S5 output dir (object_mesh.ply, keyframe.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--register-iters", type=int, default=5)
    ap.add_argument("--track-iters", type=int, default=2)
    ap.add_argument("--reacquire-iou", type=float, default=0.25)
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--depth-scale", type=Path, default=None)
    args = ap.parse_args()

    import trimesh
    import nvdiffrast.torch as dr
    sys.path.insert(0, ".")
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    dscale = np.load(args.depth_scale)["scale"] if args.depth_scale else None
    kf = json.loads((args.object / "keyframe.json").read_text())
    mesh = trimesh.load(args.object / "object_mesh.ply", process=False)
    rows = json.loads((args.masks / "MASKS.json").read_text())["rows"]
    T = len(rows)
    W, H = Image.open(args.frames / "000000.jpg").size
    run = json.loads((args.human / "RUN.json").read_text())
    rw, rh = run["model_wh"]
    sx = rw / W
    f = kf["focal"] * sx
    K = np.diag([rw / W, rh / H, 1.]) @ np.asarray(kf['K_predicted'], np.float64)

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
                         debug=0, debug_dir=str(args.out / "debug"))

    def load(i):
        rgb = np.asarray(Image.open(args.frames / f"{i:06d}.jpg").convert("RGB").resize((rw, rh), Image.BILINEAR))
        m = np.asarray(Image.open(rows[i]["mask"]).resize((rw, rh), Image.NEAREST)) > 127
        d = np.load(args.human / f"{i:06d}.npz")["depth"].astype(np.float32)
        if dscale is not None:
            d = d * float(dscale[i])
        ok = bool(rows[i].get("selected")) and rows[i]["score"] >= args.min_score and m.sum() >= 20
        return rgb, d, m, ok

    poses = np.full((T, 4, 4), np.nan)
    ious = np.full(T, np.nan)
    mask_ok = np.zeros(T, bool)
    events = []
    k0 = kf["keyframe"]
    t0 = time.time()
    rgb, d, m, ok = load(k0)
    if not ok and not (kf.get('mask_confidence_fallback') and m.sum()>=20 and rows[k0]['score']>=.3):raise ValueError('registration keyframe requires confidence gate or explicit declared lower-confidence fallback')
    if ((d>0)&np.isfinite(d)&m).sum()<.5*m.sum():raise ValueError('registration keyframe lacks valid predicted depth')
    P = est.register(K=K, rgb=rgb, depth=d, ob_mask=m, iteration=args.register_iters)
    T_s5 = np.eye(4)
    T_s5[:3, :3], T_s5[:3, 3] = np.array(kf["R_obj2cam"]), np.array(kf["t_obj2cam"])
    dR = np.degrees(np.arccos(np.clip((np.trace(P[:3, :3].T @ T_s5[:3, :3]) - 1) / 2, -1, 1)))
    events.append(dict(frame=k0, kind="register_keyframe", rot_vs_sam3d_deg=float(dR),
                       trans_vs_sam3d_m=float(np.linalg.norm(P[:3, 3] - T_s5[:3, 3]))))
    P_key = P.copy()
    import torch
    T_c = np.eye(4)
    T_c[:3, 3] = np.asarray(est.model_center, np.float64).reshape(3)

    def set_last(P_orig):
        # FoundationPose tracks the pose of its internally re-centred mesh: pose_centered = P_orig @ T(+c)
        # (estimater.py:315-343 returns pose_centered @ T(-c)); validator 02 F1.
        est.pose_last = torch.as_tensor(P_orig @ T_c, device="cuda", dtype=torch.float)

    for direction, rng in (("fwd", range(k0, T)), ("bwd", range(k0, -1, -1))):
        set_last(P_key)
        for i in rng:
            rgb, d, m, ok = load(i)
            if i != k0:
                P = est.track_one(rgb=rgb, depth=d, K=K, iteration=args.track_iters)
            else:
                P = P_key
            V = mesh.vertices @ P[:3, :3].T + P[:3, 3]
            iou = raster_iou(V, mesh.faces, K, m) if m.any() else np.nan
            if ok and iou < args.reacquire_iou and i != k0:
                P2 = est.register(K=K, rgb=rgb, depth=d, ob_mask=m, iteration=args.register_iters)
                V2 = mesh.vertices @ P2[:3, :3].T + P2[:3, 3]
                iou2 = raster_iou(V2, mesh.faces, K, m)
                events.append(dict(frame=i, kind=f"reacquire_{direction}", iou_before=float(iou), iou_after=float(iou2)))
                if iou2 > iou:
                    P, iou = P2, iou2
                else:
                    set_last(P)
            poses[i], ious[i], mask_ok[i] = P, iou, ok
            if i % 100 == 0:
                print(f"  [{direction}] {i}/{T} iou={iou:.3f} {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "poses.npz", T_obj2cam=poses, iou=ious, mask_ok=mask_ok, K_model=K)
    v = ious[mask_ok & np.isfinite(ious)]
    rep = dict(frames=T, keyframe=k0, model_wh=[rw, rh], events=events,
               iou_on_confident_masks=dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None,
                                           p10=float(np.percentile(v, 10)) if len(v) else None,
                                           frac_below_0_3=float((v < 0.3).mean()) if len(v) else None),
               seconds=time.time() - t0)
    (args.out / "TRACK.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "events"}), f"events={len(events)}", flush=True)


if __name__ == "__main__":
    main()
