#!/usr/bin/env python3
"""S3b: SAM 3D Body (MHR) per-frame human for one V2D Track 1 episode.

Prompt: the SUBJECT's bounding box, taken from the MetaV3RSE tracked person (code/smplx_to_mhr.track_person:
tallest body at the first detected frame, then nearest pelvis), projected with the episode focal and padded.
This keeps the prompt off the seated bystander seen in FORM-HOI captures. Frames where MetaV3RSE found no
person reuse the interpolated track (recorded in BBOX.json).

Outputs (camera frame = OpenCV, metres), following the V2D toolkit's export_mhr_outputs convention
(video_to_data/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py):
  mhr_model_params[T,204]  with [0:3] = native(cam_t with y,z flipped) * 10, as the toolkit writes it
  shape_params[T,45], scale_params[T,68], expr_params, pred_cam_t[T,3]
  vertices_cam[T,18439,3]  = pred_vertices + pred_cam_t (float32)
  joints_cam[T,J,3], keypoints_2d
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/tanuj/V2D_challenge")
LIB = ROOT / "video_to_data/reconstruction/modules/v2d_sam3d_body/lib"
W = ROOT / "models/sam3d_body"


def subject_boxes(human_dir: Path, K: np.ndarray, W_img: int, H_img: int, pad: float):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smplx_to_mhr import track_person
    run = json.loads((human_dir / "RUN.json").read_text())
    rows = [dict(np.load(human_dir / f"{i:06d}.npz")) if (human_dir / f"{i:06d}.npz").exists() else None
            for i in range(run["frames"])]
    verts, chosen, filled = track_person(rows, K)
    uv = verts[..., :2] / np.clip(verts[..., 2:3], 1e-3, None) * K[0, 0] + K[:2, 2]
    lo, hi = uv.min(1), uv.max(1)
    c, s = (lo + hi) / 2, (hi - lo) * (1 + pad)
    box = np.concatenate([c - s / 2, c + s / 2], 1)
    box[:, [0, 2]] = box[:, [0, 2]].clip(0, W_img - 1)
    box[:, [1, 3]] = box[:, [1, 3]].clip(0, H_img - 1)
    return box.astype(np.float32), chosen, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--human", type=Path, required=True, help="MetaV3RSE work/s3_human/episode_XXXXXX")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--focal", type=float, required=True, help="source-resolution focal (px)")
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--inference-type", default="full", choices=["full", "body"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", str(W / "torch_home"))
    sys.path.insert(0, str(LIB))
    from sam_3d_body import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    files = sorted(args.frames.glob("*.jpg"))
    if args.limit:
        files = files[: args.limit]
    W_img, H_img = Image.open(files[0]).size
    K = np.array([[args.focal, 0, W_img / 2], [0, args.focal, H_img / 2], [0, 0, 1]], np.float32)
    boxes, chosen, filled = subject_boxes(args.human, K, W_img, H_img, args.pad)

    model, cfg = load_sam_3d_body(checkpoint_path=str(W / "sam-3d-body-dinov3/model.ckpt"), device="cuda",
                                  mhr_path=str(W / "sam-3d-body-dinov3/assets/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    torch.cuda.empty_cache = lambda: None
    cam = torch.from_numpy(K)[None]

    keys = ("mhr_model_params", "shape_params", "scale_params", "expr_params", "pred_cam_t", "focal_length",
            "pred_vertices", "pred_joint_coords", "pred_keypoints_3d", "pred_keypoints_2d")
    acc = {k: [] for k in keys}
    t0 = time.time()
    for i, f in enumerate(files):
        img = np.asarray(Image.open(f).convert("RGB"))
        out = est.process_one_image(img=img, bboxes=boxes[i][None], masks=None, cam_int=cam,
                                    inference_type=args.inference_type)
        if len(out) != 1:
            raise RuntimeError(f"frame {i}: expected 1 person output, got {len(out)}")
        o = out[0]
        for k in keys:
            v = o[k]
            acc[k].append(v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, np.float32))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    arr = {k: np.stack(v) for k, v in acc.items()}
    cam_t = arr["pred_cam_t"].reshape(-1, 3)
    params = arr["mhr_model_params"].reshape(len(files), -1).copy()
    native = cam_t.copy()
    native[:, [1, 2]] *= -1
    params[:, 0:3] = native * 10          # toolkit export convention
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "sam3d_body.npz", mhr_model_params=params.astype(np.float32),
        shape_params=arr["shape_params"].reshape(len(files), -1), scale_params=arr["scale_params"].reshape(len(files), -1),
        expr_params=arr["expr_params"].reshape(len(files), -1), pred_cam_t=cam_t,
        vertices_cam=(arr["pred_vertices"].reshape(len(files), -1, 3) + cam_t[:, None]).astype(np.float32),
        joints_cam=(arr["pred_joint_coords"].reshape(len(files), -1, 3) + cam_t[:, None]).astype(np.float32),
        keypoints_2d=arr["pred_keypoints_2d"].reshape(len(files), -1, 2), boxes=boxes, K=K)
    (args.out / "BBOX.json").write_text(json.dumps(dict(
        frames=len(files), focal=args.focal, pad=args.pad, inference_type=args.inference_type,
        metav3rse_filled_frames=filled, chosen_slot_hist={str(k): chosen.count(k) for k in sorted(set(chosen))},
        params_shape=list(params.shape), seconds=time.time() - t0), indent=1))
    print(f"[s3db] DONE {args.out} {len(files)} frames {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
