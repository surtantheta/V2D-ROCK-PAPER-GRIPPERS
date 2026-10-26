#!/usr/bin/env python3
"""Decompose the labeled BEHAVE object error for our pipeline vs a branch arm (scorer-only diagnostic, GT used ONLY here).

Per sequence, on the human-aligned predictions, using the scorer's own surface sampling and GT join:
  size ratio      RMS radius of the scaled predicted surface / RMS radius of the GT surface
  centroid err    mean |pred surface centroid - GT surface centroid| (cm), and its camera-z part
  CD              scorer CD (as reported)
  CD centred      CD after moving the predicted centroid onto the GT centroid per frame (= shape + size + rotation)
usage: labeled_diagnose.py <ours_aligned_PREDICTIONS.json> <branch_aligned_PREDICTIONS.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

BR = Path("/home/tanuj/V2D_challenge/experiments/offline_opt_20261020")
sys.path.insert(0, str(BR / "code"))
import labeled_object_scorer as S                                  # noqa: E402

REQ = json.loads((BR / "benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json").read_text())
REFS = {x["sequence"]: x for x in json.loads((BR / "benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json").read_text())}


def gt_surface(ref):
    simplified, _ = S.load_surface(ref["object_mesh_simplified"])
    v, f = S.load_surface(ref["object_mesh_original"], simplified.mean(0))
    return S.surface_samples(v, f, 4096)


def analyse(path):
    pred = json.loads(Path(path).read_text())
    out = {}
    for item in pred["sequences"]:
        seq = item["sequence"]; ref = REFS[seq]
        req = next(x for x in REQ["BEHAVE"] if x["sequence"] == seq)
        lookup = {f: i for i, f in enumerate(req["video_frame_indices"])}
        gt = np.load(ref["gt_npz"], allow_pickle=False)
        gp, gc = gt_surface(ref)
        v, f = S.load_surface(item["mesh_path"])
        pp, pc = S.surface_samples(v, f, 4096)
        s = float(item["object_scale"])
        rows = []
        for fr in item["frames"]:
            row = ref["GT_row_indices"][lookup[fr["video_frame_index"]]]
            r = np.array(fr["R_obj2cam"]); t = np.array(fr["t_obj2cam_m"])
            T = gt["pose_abs"][row]; rg, tg = T[:3, :3], T[:3, 3]
            P = s * pp @ r.T + t; G = gp @ rg.T + tg
            dc = P.mean(0) - G.mean(0)
            rows.append((S.surface_cd_sum_cm(P, G), S.surface_cd_sum_cm(P - dc, G), np.linalg.norm(dc) * 100, abs(dc[2]) * 100))
        rows = np.array(rows)
        size = np.sqrt(((s * (pp - pp.mean(0))) ** 2).sum(1).mean()) / np.sqrt(((gp - gp.mean(0)) ** 2).sum(1).mean())
        out[seq] = dict(size_ratio=float(size), CD=float(rows[:, 0].mean()), CD_centred=float(rows[:, 1].mean()),
                        centroid_err_cm=float(rows[:, 2].mean()), centroid_z_err_cm=float(rows[:, 3].mean()))
    return out


if __name__ == "__main__":
    a, b = analyse(sys.argv[1]), analyse(sys.argv[2])
    for seq in a:
        print(seq)
        for tag, d in (("ours", a[seq]), ("branch", b[seq])):
            print(f"  {tag:7s} " + " ".join(f"{k}={v:.2f}" for k, v in d.items()))
    for tag, d in (("ours", a), ("branch", b)):
        print(tag, "mean", " ".join(f"{k}={np.mean([x[k] for x in d.values()]):.2f}" for k in next(iter(d.values()))))
