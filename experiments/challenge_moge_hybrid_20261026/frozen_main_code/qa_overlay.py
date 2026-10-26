#!/usr/bin/env python3
"""QA: project the SUBMISSION arrays (as the host will decode them) onto the video frames.

Reads <submission_dir>/episode_XXXXXX.npz + episode_XXXXXX_object.ply exactly as the kit packer does, decodes the
human with the kit TorchScript MHR model in the host frame (diag(1,-1,-1) @ MHR / 100, kit MESH_TO_MHR_PARAMS.md and
validator 01 recipe), places the object as s R p + t, and projects both with the episode focal (work/FOCALS.json or
--focal). Human points green, object points red, on N evenly spaced scored frames. Also writes per-frame numbers:
fraction of projected object points inside the SAM3 mask, human/object depth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

ROOT = Path("/home/tanuj/V2D_challenge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--focal", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    e = f"episode_{args.episode:06d}"
    d = np.load(args.submission / f"{e}.npz")
    mesh = trimesh.load(args.submission / f"{e}_object.ply", process=False)
    T = len(d["pose"])
    f = args.focal or json.loads((ROOT / "work/FOCALS.json").read_text())["episode_focal_px"][str(args.episode)]
    frames = ROOT / "work/frames" / e
    W, H = Image.open(frames / "000000.jpg").size
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ts = torch.jit.load(str(ROOT / "models/mhr/mhr_model.pt"), map_location=dev).eval()
    rep = json.loads((ROOT / "logs/track1_verify_report.json").read_text())["sample_submission"]["per_episode"][str(args.episode)]
    first = int(rep["first"])
    last = max(f for f in [int(rep["last"]) if int(rep["last"]) != 999999 else T - 1, first])
    # always include the first scored frame (the scorer fits its Sim(3) there), then spread over the scored range
    idx = np.unique(np.concatenate([[first], np.linspace(first, min(last, T - 1), args.n - 1).round().astype(int)]))
    with torch.no_grad():
        p = torch.from_numpy(np.concatenate([d["pose"][idx], np.repeat(d["scales"][None], len(idx), 0)], 1)).float().to(dev)
        z = torch.from_numpy(d["shape"][None]).float().to(dev).expand(len(idx), -1).contiguous()
        v, _ = ts(z, p, torch.zeros(len(idx), 72, device=dev), True)
    Vh = v.float().cpu().numpy() * np.array([1, -1, -1]) / 100.0
    Vo_local = np.asarray(mesh.vertices) * float(d["object_scale"])
    mc = ROOT / "work/MASK_CHOICE.json"
    mv = json.loads(mc.read_text()).get(str(args.episode), "v1") if mc.exists() else "v1"
    masks = json.loads((ROOT / "work" / ("masks" if mv == "v1" else "masks_v2") / e / "MASKS.json").read_text())["rows"]

    def proj(P):
        return P[:, :2] / np.clip(P[:, 2:3], 1e-3, None) * K[0, 0] + K[:2, 2]

    tiles, rows = [], []
    for k, i in enumerate(idx):
        Vo = Vo_local @ d["object_rotation"][i].T + d["object_translation"][i]
        img = Image.open(frames / f"{i:06d}.jpg").convert("RGB")
        dr = ImageDraw.Draw(img)
        for (u, vv) in proj(Vh[k])[::4]:
            dr.point((u, vv), fill=(0, 255, 0))
        uvo = proj(Vo)
        for (u, vv) in uvo[:: max(1, len(uvo) // 3000)]:
            dr.point((u, vv), fill=(255, 0, 0))
        m = np.asarray(Image.open(masks[i]["mask"])) > 127
        ui = np.clip(uvo.round().astype(int), 0, [W - 1, H - 1])
        inside = float(m[ui[:, 1], ui[:, 0]].mean()) if m.any() else None
        rows.append(dict(frame=int(i), human_z=float(Vh[k][:, 2].mean()), object_z=float(Vo[:, 2].mean()),
                         object_points_in_mask=inside))
        tiles.append(img.resize((384, 288)))
    cols = 4
    sheet = Image.new("RGB", (384 * cols, 288 * ((len(tiles) + cols - 1) // cols)))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 384, (k // cols) * 288))
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out / f"overlay_{e}.jpg")
    (args.out / f"overlay_{e}.json").write_text(json.dumps(dict(focal=f, rows=rows), indent=1))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
