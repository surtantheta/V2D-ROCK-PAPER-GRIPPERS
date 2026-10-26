#!/usr/bin/env python3
"""S3: MetaV3RSE human + scene inference on one V2D Track 1 episode (static third-person camera).

Derived from UniCon3Rpp-push-b3c6624/tools/wild_video_infer.py (validated 2026-09-23 on wild videos),
with three V2D-specific changes:
  * human/scene only: the object branch needs a RHOBIN CAD template that V2D does not provide, so the
    object fields are never built (with_object=False, exactly the source's PASS-1 view contract);
  * --mode focal runs only the first --focal-frames frames and writes FOCAL.json (pooled per camera later);
    --mode full runs every frame with a fixed --focal (pixels at the ORIGINAL resolution);
  * per frame we save the decoded SMPL-X camera-frame vertices (SMPL_Layer, person_center='head', the
    exporter's convention) and the predicted camera-frame depth/confidence (float16), not the full pointmap.

Run from the UniCon3R++ repo root with PYTHONPATH=src:src/croco and the human3r_new env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

U3PP = Path("/home/tanuj/UniCon3Rpp-push-b3c6624")
sys.path.insert(0, str(U3PP))
sys.path.insert(0, str(U3PP / "src"))
sys.path.insert(0, str(U3PP / "eval_object"))
sys.path.insert(0, str(U3PP / "tools"))

REGISTRY = "/rhome/halinh/BEHAVE/ref_hoi.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="train_spatial_gtcrop25")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["focal", "full"], required=True)
    p.add_argument("--focal", type=float, default=0.0, help="full mode: focal in ORIGINAL-resolution pixels")
    p.add_argument("--focal-frames", type=int, default=96)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--max-humans", type=int, default=3)
    args = p.parse_args()

    from wild_video_infer import model_size, intrinsics
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf  # noqa: F401
    from dust3r.datasets.behave import BEHAVE_Multi
    from dust3r.datasets.utils.transforms import ImgNorm
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.image import pad_image
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.smpl_layer import SMPL_Layer
    from eval_object_behave import _load_config, _extract_state_dict
    import roma

    torch.manual_seed(0)
    dev = "cuda"
    files = sorted(args.frames.glob("*.jpg"))
    if args.mode == "focal":
        files = files[: args.focal_frames]
    cfg = _load_config(args.config)
    model = eval(cfg.model)
    msg = model.load_state_dict(_extract_state_dict(torch.load(args.ckpt, weights_only=False, map_location="cpu")),
                                strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(f"checkpoint/config mismatch: missing={msg.missing_keys[:4]} unexpected={msg.unexpected_keys[:4]}")
    model = model.to(dev).eval()

    W, H = Image.open(files[0]).size
    RES_W, RES_H = model_size(W, H)
    tds = BEHAVE_Multi(split="val", ROOT="/rhome/tanuj/BEHAVE", source_split="val", ref=REGISTRY,
                       resolution=[[RES_W, RES_H]], num_views=1, contact_num_verts=6890,
                       transform=ImgNorm, split_mode="subject",
                       subject_index="/rhome/tanuj/unicon3rpp_runs/behave_subject_index.json")
    TPL = tds[0][0]
    print(f"[v2d] {args.frames.name}: {len(files)} frames, source {W}x{H}, model {RES_W}x{RES_H}", flush=True)

    def load(path):
        return ImgNorm(Image.open(path).convert("RGB").resize((RES_W, RES_H), Image.BILINEAR))

    def make_view(img, idx, first, K_model):
        v = {}
        for k, val in TPL.items():
            if isinstance(val, np.ndarray):
                v[k] = torch.from_numpy(np.zeros_like(val))[None].to(dev)
            elif torch.is_tensor(val):
                v[k] = torch.zeros_like(val)[None].to(dev)
            elif isinstance(val, (bool, np.bool_)):
                v[k] = torch.tensor([bool(val)], device=dev)
            else:
                v[k] = [val]
        v["img"] = img[None].to(dev)
        v["camera_intrinsics"] = torch.from_numpy(K_model)[None].to(dev)
        v["true_shape"] = torch.tensor([[RES_H, RES_W]], dtype=torch.int32, device=dev)
        v["idx"], v["instance"] = [idx], [str(idx)]
        for key, value in (("reset", first), ("is_metric", True), ("is_video", True), ("img_mask", True),
                           ("ray_mask", False), ("single_view", False), ("camera_only", False),
                           ("depth_only", False)):
            v[key] = torch.tensor([bool(value)], device=dev)
        v["img_mhmr"] = pad_image(v["img"], model.mhmr_img_res)[0:1]
        v["K_mhmr"] = resize_camera_intrinsics(v["camera_intrinsics"], RES_H, RES_W, model.mhmr_img_res)
        return v

    def run(K_model, collect):
        rows = []
        for start in range(0, len(files), args.window):
            chunk = files[start:start + args.window]
            views = [make_view(load(f), start + j, start + j == 0, K_model) for j, f in enumerate(chunk)]
            with torch.no_grad():
                out = model(views, inference=True)
            for j, pred in enumerate(out.ress):
                rows.append(collect(pred, views[j], start + j))
            del out, views
            torch.cuda.empty_cache()
            print(f"  [{args.mode}] {min(start + args.window, len(files))}/{len(files)}", flush=True)
        return rows

    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "focal":
        seed = 1.0 * RES_W
        focals = run(intrinsics(seed, RES_W, RES_H), lambda pred, view, i: float(estimate_focal_knowing_depth(
            pred["pts3d_in_self_view"].float(), torch.tensor([[RES_W // 2, RES_H // 2]], device=dev).float(),
            focal_mode="weiszfeld")[0]))
        f_model = float(np.median(focals))
        rec = dict(frames=len(files), model_wh=[RES_W, RES_H], source_wh=[W, H], seed_focal_model_px=seed,
                   focal_model_px=f_model, focal_source_px=f_model * W / RES_W,
                   focal_model_percentiles={k: float(np.percentile(focals, k)) for k in (10, 50, 90)})
        (args.out / "FOCAL.json").write_text(json.dumps(rec, indent=1))
        print(f"[v2d] FOCAL source {rec['focal_source_px']:.1f}px (model {f_model:.1f})", flush=True)
        return

    if args.focal <= 0:
        raise SystemExit("--focal required in full mode")
    f_model = args.focal * RES_W / W
    K_model = intrinsics(f_model, RES_W, RES_H)
    layers = {}

    def layer(nb):
        if nb not in layers:
            layers[nb] = SMPL_Layer(type="smplx", gender="neutral", num_betas=nb, kid=False,
                                    person_center="head").to(dev).eval()
        return layers[nb]

    def collect(pred, view, i):
        row = dict(frame=i)
        pts = pred["pts3d_in_self_view"][0].float()
        row["depth"] = pts[..., 2].cpu().numpy().astype(np.float16)
        row["conf"] = pred["conf_self"][0].float().cpu().numpy().astype(np.float16)
        nh = 0
        if "smpl_rotmat" in pred and torch.is_tensor(pred["smpl_rotmat"]) and pred["smpl_rotmat"].shape[1] > 0:
            rot = pred["smpl_rotmat"][0].float()            # [H,J,3,3]
            nh = int(rot.shape[0])
            k = min(nh, args.max_humans)
            shape = pred["smpl_shape"][0, :k].float()
            transl = pred["smpl_transl"][0, :k].float()
            pose = roma.rotmat_to_rotvec(rot[:k])
            if pose.shape[1] < 53:
                pose = torch.cat([pose, pose.new_zeros((k, 53 - pose.shape[1], 3))], 1)
            pose = pose[:, :53]
            nb = 11 if shape.shape[-1] >= 11 else 10
            expr = pred["smpl_expression"][0, :k].float() if "smpl_expression" in pred else None
            with torch.no_grad():
                o = layer(nb)(pose, shape[:, :nb], transl, None, None,
                              K=torch.from_numpy(K_model)[None].expand(k, 3, 3).to(dev), expression=expr)
            row["verts"] = o["smpl_v3d"].cpu().numpy().astype(np.float32)          # [k,10475,3] camera frame, m
            row["pelvis"] = o["smpl_transl_pelvis"][:, 0].cpu().numpy().astype(np.float32)
            row["smpl_rotmat"] = rot[:k].cpu().numpy()
            row["smpl_shape"] = shape.cpu().numpy()
            row["smpl_transl"] = transl.cpu().numpy()
            for key in ("smpl_scores", "smpl_conf"):
                if key in pred and torch.is_tensor(pred[key]):
                    row[key] = pred[key][0].detach().float().cpu().numpy()
        row["n_humans"] = nh
        np.savez_compressed(args.out / f"{i:06d}.npz", **row)
        return dict(frame=i, n_humans=nh)

    rows = run(K_model, collect)
    meta = dict(frames=len(files), window=args.window, model_wh=[RES_W, RES_H], source_wh=[W, H], focal_source_px=args.focal,
                focal_model_px=f_model, ckpt=args.ckpt, config=args.config,
                n_humans_hist={str(n): sum(r["n_humans"] == n for r in rows) for n in sorted({r["n_humans"] for r in rows})},
                frames_without_human=[r["frame"] for r in rows if r["n_humans"] == 0])
    (args.out / "RUN.json").write_text(json.dumps(meta, indent=1))
    print(f"[v2d] DONE {args.out} humans={meta['n_humans_hist']}", flush=True)


if __name__ == "__main__":
    main()
