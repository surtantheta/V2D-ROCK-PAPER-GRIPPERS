#!/usr/bin/env python3
"""Mask QA without GT: per episode and mask version, confident-frame count, empty count, median area, and the
NEAR-SUBJECT rate = fraction of confident masks whose centroid lies inside the subject box (SAM 3D Body prompt box)
enlarged x2 (validator 04 used this to expose ep09's background-desk masks). Also writes a contact sheet per
episode/version (8 confident frames, subject box green, mask red) for the visual pass.
Usage: mask_qa.py --episodes 0 1 ... --out work/qa/masks   (reads work/masks and work/masks_v2)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W = Path("/home/tanuj/V2D_challenge/work")


def qa(ep, version, min_score, sheet_dir):
    d = W / ("masks" if version == "v1" else "masks_v2") / f"episode_{ep:06d}"
    if not (d / "MASKS.json").exists():
        return None
    rows = json.loads((d / "MASKS.json").read_text())["rows"]
    boxes = np.load(W / f"s3db/episode_{ep:06d}/sam3d_body.npz")["boxes"]
    conf, near, areas = [], 0, []
    for i, r in enumerate(rows):
        if r.get("selected") and r["score"] >= min_score and r.get("centroid"):
            conf.append(i)
            areas.append(r["area"])
            x0, y0, x1, y1 = boxes[i]
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
            u, v = r["centroid"]
            near += abs(u - cx) <= w and abs(v - cy) <= h
    rep = dict(episode=ep, version=version, frames=len(rows), confident=len(conf),
               empty=sum(not r.get("selected") for r in rows),
               near_subject=near / max(len(conf), 1), median_area=float(np.median(areas)) if areas else 0.0)
    pick = [conf[k] for k in np.linspace(0, len(conf) - 1, min(8, len(conf))).round().astype(int)] if conf else []
    tiles = []
    for i in pick:
        img = Image.open(W / f"frames/episode_{ep:06d}/{i:06d}.jpg").convert("RGB")
        a = np.asarray(img).astype(np.float32)
        m = np.asarray(Image.open(rows[i]["mask"])) > 127
        a[m] = 0.4 * a[m] + 0.6 * np.array([255, 0, 0])
        img = Image.fromarray(a.astype(np.uint8))
        ImageDraw.Draw(img).rectangle(list(boxes[i]), outline=(0, 255, 0), width=4)
        tiles.append(img.resize((384, 288)))
    if tiles:
        sheet = Image.new("RGB", (384 * 4, 288 * ((len(tiles) + 3) // 4)))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % 4) * 384, (k // 4) * 288))
        sheet_dir.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_dir / f"ep{ep:02d}_{version}.jpg")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=W / "qa/masks")
    ap.add_argument("--min-score", type=float, default=0.5)
    args = ap.parse_args()
    reps = [r for ep in args.episodes for v in ("v1", "v2") if (r := qa(ep, v, args.min_score, args.out))]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "MASK_QA.json").write_text(json.dumps(reps, indent=1))
    for r in reps:
        print(f"ep{r['episode']:02d} {r['version']} conf={r['confident']:4d}/{r['frames']} empty={r['empty']:4d} "
              f"near={r['near_subject']:.2f} area={r['median_area']:.4f}", flush=True)


if __name__ == "__main__":
    main()
