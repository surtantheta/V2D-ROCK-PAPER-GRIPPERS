#!/usr/bin/env python3
"""S5: object mesh + keyframe pose for one V2D episode with SAM 3D Objects.

Inputs: full-res frames, SAM3 object masks (MASKS.json), MetaV3RSE per-frame camera-frame depth (512x384) and the
episode focal. The MetaV3RSE depth is upsampled to full resolution and back-projected with the episode K, giving
a pinhole-consistent metric pointmap in the SAME camera frame as the human. It is passed to SAM 3D Objects in
PyTorch3D camera convention (x,y negated: inference_pipeline_pointmap.camera_to_pytorch3d_camera).

Keyframe candidates: confident (score>=min_score), non-border masks; the top --candidates by area, spread over
the episode. Each candidate is reconstructed and placed with the library's own recipe (validator 02 O2/O3, from
layout_post_optimization_utils.py:116-122): undo to_glb's z-up->y-up rotation (V_local = V_glb @ GLB_ROT.T), apply
the pose in ROW form p @ Rq + t (Rq = quaternion_to_matrix(q), so the column-form rotation is Rq.T), then P3D->OpenCV
with F. The candidate whose projected silhouette best matches the SAM3 mask (IoU) wins; all IoUs are recorded.
Mesh post-processing is OFF (it needs nvdiffrast, absent in this env; validator 02 O5); the mesh is decimated to
--max-faces with pyvista instead (the kit packer decimates again to its 4096 budget).

Output (camera frame = OpenCV, metres):
  object_mesh.ply   mesh in the OBJECT frame, metric (scale baked in), centred at its own origin
  keyframe.json     keyframe index, R_obj2cam, t_obj2cam, IoU table, timings
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

F = np.diag([-1.0, -1.0, 1.0])                    # P3D camera <-> OpenCV camera
GLB_ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)   # to_glb: v_glb = v @ GLB_ROT


def quat_wxyz_to_R(q):
    q = np.asarray(q, np.float64).reshape(-1)[:4]
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def decimate(V, Fc, max_faces):
    if len(Fc) <= max_faces:
        return V, Fc
    import pyvista as pv
    pd = pv.PolyData(V, np.hstack([np.full((len(Fc), 1), 3), Fc]).ravel())
    pd = pd.triangulate().decimate(1.0 - max_faces / len(Fc))
    return np.asarray(pd.points, np.float64), np.asarray(pd.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def raster_iou(V_cam, faces, K, mask):
    """Silhouette IoU of a camera-frame mesh vs a boolean mask (simple per-triangle fill)."""
    import cv2
    H, W = mask.shape
    z = V_cam[:, 2]
    if (z <= 1e-3).any():
        return 0.0
    uv = (V_cam[:, :2] / z[:, None]) * [K[0, 0], K[1, 1]] + K[:2, 2]
    sil = np.zeros((H, W), np.uint8)
    tri = uv[faces].round().astype(np.int32)
    for t in tri:                      # one triangle at a time: fillPoly over many polygons is even-odd
        cv2.fillConvexPoly(sil, t, 1)
    sil = sil.astype(bool)
    inter, union = (sil & mask).sum(), (sil | mask).sum()
    return float(inter / max(union, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--human", type=Path, required=True, help="MetaV3RSE work/s3_human/episode_X (depth per frame)")
    ap.add_argument("--focal", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sam-root", type=Path, default=Path("/rhome/tanuj/projects/sam-3d-objects"))
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-faces", type=int, default=20000)
    ap.add_argument("--depth-scale", type=Path, default=None, help="depth_scale.py npz (per-frame scale)")
    ap.add_argument("--only-frame", type=int, default=None, help="internal: reconstruct one candidate in this process")
    args = ap.parse_args()

    rows = json.loads((args.masks / "MASKS.json").read_text())["rows"]
    _ds = np.load(args.depth_scale) if args.depth_scale else None
    dscale = _ds["scale"] if _ds is not None else None
    doff = _ds["offset"] if (_ds is not None and "offset" in _ds.files) else (np.zeros_like(dscale) if dscale is not None else None)
    W, H = Image.open(args.frames / "000000.jpg").size
    K = np.array([[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], np.float64)

    scores = np.array([r["score"] for r in rows if r.get("selected")])
    min_score = args.min_score
    if not (scores >= min_score).any() and len(scores):
        min_score = float(np.percentile(scores, 90))      # validator 03: weak-mask episodes, relax and record
        print(f"[s5] no mask >= {args.min_score}; relaxed keyframe threshold to p90 score {min_score:.3f}", flush=True)
    cands = []
    for i, r in enumerate(rows):
        if not r.get("selected") or r["score"] < min_score:
            continue
        m = np.asarray(Image.open(r["mask"])) > 127
        ys, xs = np.nonzero(m)
        if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
            continue
        if not (args.human / f"{i:06d}.npz").exists():
            continue
        cands.append((r["area"], i))
    if not cands:
        raise SystemExit("no admissible keyframe candidate")
    cands.sort(reverse=True)
    picked = []
    for area, i in cands:             # largest masks, at least 30 frames apart
        if all(abs(i - j) >= 30 for _, j in picked):
            picked.append((area, i))
        if len(picked) == args.candidates:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    if args.only_frame is None:
        import subprocess
        results, best = [], None
        for area, i in picked:
            cmd = [sys.executable] + [a for a in sys.argv if True] + ["--only-frame", str(i)]
            rc = subprocess.run(cmd).returncode
            cf = args.out / f"cand_{i:06d}.npz"
            if rc != 0 or not cf.exists():
                results.append(dict(frame=i, failed=True, returncode=rc))
                print(f"[s5] candidate {i} FAILED rc={rc}", flush=True)
                continue
            c = np.load(cf, allow_pickle=True)
            rec = json.loads(str(c["rec"]))
            results.append(rec)
            if best is None or rec["iou"] > best[0]:
                best = (rec["iou"], i, c)
        if best is None:
            raise SystemExit("all keyframe candidates failed")
        iou, i, c = best
        V_obj, F_obj, R_cv, t_cv = c["V_obj"], c["F_obj"], c["R_cv"], c["t_cv"]
        centre = V_obj.mean(0)
        mesh = trimesh.Trimesh(V_obj - centre, F_obj, process=False)
        if c["colors"].size:
            mesh.visual.vertex_colors = c["colors"]
            print("[s5] vertex colours transferred", flush=True)
        mesh.export(args.out / "object_mesh.ply")
        t_cv = t_cv + R_cv @ centre       # re-centre: V_cam = R (V_c + centre) + t
        name = "glb_unrotated_rowform"
        (args.out / "keyframe.json").write_text(json.dumps(dict(
            keyframe=int(i), hypothesis=name, iou=float(iou), R_obj2cam=R_cv.tolist(), t_obj2cam=t_cv.tolist(),
            n_vertices=int(len(V_obj)), n_faces=int(len(F_obj)), extent_m=(V_obj.max(0) - V_obj.min(0)).tolist(),
            focal=args.focal, keyframe_min_score=min_score, candidates=results), indent=1))
        print(f"[s5] best frame {i} {name} IoU {iou:.3f} extent {(V_obj.max(0) - V_obj.min(0)).round(3)}", flush=True)
        return

    sys.path[:0] = [str(args.sam_root), str(args.sam_root / "notebook")]
    from inference import Inference
    model = Inference(str(args.sam_root / "checkpoints/hf/pipeline.yaml"), compile=False)
    fixed_normalized_K=np.diag([1/W,1/H,1.])@K
    def known_camera(image, pointmap=None):
        assert pointmap is not None, 'known predicted K requires supplied pointmap'
        loaded=torch.from_numpy(model._pipeline.image_to_float(image))
        points=pointmap.to(model._pipeline.device).permute(2,0,1)
        points=model._pipeline._clip_pointmap(points,loaded[...,-1])
        return {'pts_color':loaded.permute(2,0,1).contiguous()[:3], 'pointmap':points,
                'intrinsics':torch.tensor(fixed_normalized_K,device=model._pipeline.device,dtype=torch.float32)}
    model._pipeline.compute_pointmap=known_camera
    results = []
    best = None
    for area, i in [(a, j) for a, j in picked if j == args.only_frame]:
        t0 = time.time()
        img = np.asarray(Image.open(args.frames / f"{i:06d}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(rows[i]["mask"])) > 127
        d = np.load(args.human / f"{i:06d}.npz")["depth"].astype(np.float32)
        if dscale is not None:
            d = d * float(dscale[i]) + float(doff[i])
        d_full = np.asarray(Image.fromarray(d).resize((W, H), Image.BILINEAR), np.float32)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        X = (u - K[0, 2]) / K[0, 0] * d_full
        Y = (v - K[1, 2]) / K[1, 1] * d_full
        pm_p3d = torch.from_numpy(np.stack([-X, -Y, d_full], -1).astype(np.float32))
        rgba = model.merge_mask_to_rgba(img, mask)
        out = model._pipeline.run(rgba, None, seed=args.seed, stage1_only=False, with_mesh_postprocess=False,
                                  with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True,
                                  pointmap=pm_p3d)
        Rq = quat_wxyz_to_R(out["rotation"].detach().cpu().numpy())
        t_p3d = out["translation"].detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
        s = float(out["scale"].detach().cpu().numpy().reshape(-1)[0])
        glb = out["glb"]
        Vg = np.asarray(glb.vertices, np.float64)
        Fg = np.asarray(glb.faces, np.int64)
        Vg, Fg = decimate(Vg, Fg, args.max_faces)
        V_local = Vg @ GLB_ROT.T                       # undo to_glb (v_glb = v @ GLB_ROT)
        R_cv, t_cv = F @ Rq.T, F @ t_p3d               # row-form p @ Rq + t in P3D -> column form in OpenCV
        V_cam = (s * V_local) @ R_cv.T + t_cv
        iou = raster_iou(V_cam, Fg, K, mask)
        rec = dict(frame=i, iou=iou, scale=s, area=area, seconds=time.time() - t0, faces=int(len(Fg)),
                   layout_iou=float(out["iou"]) if "iou" in out and np.size(out["iou"]) == 1 else None,
                   layout_post_ran="iou" in out,
                   depth_median_in_mask=float(np.median(d_full[mask])), mesh_z_median=float(np.median(V_cam[:, 2])))
        results.append(rec)
        print(json.dumps(rec), flush=True)
        if best is None or iou > best[0]:
            best = (iou, i, "glb_unrotated_rowform", s * V_local, Fg, R_cv, t_cv, glb, s)
    iou, i, name, V_obj, F_obj, R_cv, t_cv, glb, s_best = best
    colors = np.zeros((0, 4), np.uint8)
    try:                               # validator 02 N1: carry GLB vertex colours onto the decimated mesh
        from scipy.spatial import cKDTree
        cg = np.asarray(glb.visual.vertex_colors)
        Vg_obj = s_best * (np.asarray(glb.vertices, np.float64) @ GLB_ROT.T)   # same frame as V_obj (pre-centring)
        if len(cg) == len(glb.vertices):
            colors = cg[cKDTree(Vg_obj).query(V_obj)[1]]
    except Exception as exc:
        print(f"[s5] WARNING: colour transfer failed: {exc}", flush=True)
    np.savez_compressed(args.out / f"cand_{i:06d}.npz", V_obj=V_obj, F_obj=F_obj, R_cv=R_cv, t_cv=t_cv,
                        colors=colors, rec=json.dumps(results[-1]))


if __name__ == "__main__":
    main()
