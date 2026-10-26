#!/usr/bin/env python3
"""S4b: SAM 3D Body per-frame MHR meshes -> one-identity MHR parameters (kit converter).

Input vertices are SAM 3D Body `vertices_cam` (pred_vertices + pred_cam_t): OpenCV camera frame, metres. With the
toolkit's export convention this is the Track 1 frame verts = diag(1,-1,-1) @ MHR(...) / 100, so the kit's
`mesh_to_mhr_params.convert` (which expects exactly that frame) applies directly.

Optional temporal smoothing (--smooth-sigma frames, Gaussian, 0 = off) is applied to the vertex trajectories before
fitting, to reduce per-frame jitter that ACC-H penalises. Raw and smoothed are both kept for later comparison.
Output: <out>/mhr_params.npz pose[T,136] scales[68] shape[45] + CONVERT.json (kit report + residuals).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path("/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit/tools/track1")
MODEL = "/home/tanuj/V2D_challenge/models/mhr/mhr_model.pt"


def gaussian_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    out = np.zeros_like(x)
    for j, w in enumerate(k):
        out += w * pad[j:j + len(x)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d-body", type=Path, required=True, help="work/s3db/episode_X/sam3d_body.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="isotropic sigma (frames); overridden per axis below")
    ap.add_argument("--smooth-sigma-xy", type=float, default=None)
    ap.add_argument("--smooth-sigma-z", type=float, default=None, help="camera-depth axis (validator 03: depth jitter dominates ACC-H)")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--model-batch", type=int, default=256, help="meshes per MHR call (memory; kit default may need ~14 GB)")
    args = ap.parse_args()
    sys.path.insert(0, str(KIT))
    from mesh_to_mhr_params import convert

    d = np.load(args.sam3d_body)
    V = d["vertices_cam"].astype(np.float64)
    sxy = args.smooth_sigma if args.smooth_sigma_xy is None else args.smooth_sigma_xy
    sz = args.smooth_sigma if args.smooth_sigma_z is None else args.smooth_sigma_z
    # validator 04: centroid depth is mostly noise, body-relative depth is real motion ->
    # smooth the centroid anisotropically (xy sxy, z sz) and the body-relative part isotropically with sxy
    c = V.mean(1, keepdims=True)
    cs = c.copy()
    cs[..., :2] = gaussian_smooth(c[..., :2], sxy)
    cs[..., 2:] = gaussian_smooth(c[..., 2:], sz)
    Vs = cs + gaussian_smooth(V - c, sxy)
    res = convert(Vs, MODEL, precision=args.precision, model_batch=args.model_batch)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "mhr_params.npz", pose=np.asarray(res["pose"], np.float32),
                        scales=np.asarray(res["scales"], np.float32), shape=np.asarray(res["shape"], np.float32))
    rep = res.get("report")
    rep = json.loads(rep) if isinstance(rep, str) else rep
    err = np.asarray(res.get("per_frame_vertex_error_mm", []), float)
    fin = err[np.isfinite(err)]
    out = dict(frames=int(len(V)), smooth_sigma_xy=sxy, smooth_sigma_z=sz, precision=args.precision, report=rep,
               nonfinite_error_frames=int(err.size - fin.size),
               vertex_error_mm=dict(mean=float(fin.mean()) if fin.size else None,
                                    p90=float(np.percentile(fin, 90)) if fin.size else None,
                                    max=float(fin.max()) if fin.size else None),
               smoothing_shift_mm=dict(mean=float(np.linalg.norm(Vs - V, axis=-1).mean() * 1000)))
    (args.out / "CONVERT.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "report"}), flush=True)


if __name__ == "__main__":
    main()
