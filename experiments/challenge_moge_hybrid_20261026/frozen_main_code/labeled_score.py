#!/usr/bin/env python3
"""Score OUR pipeline's labeled-clip outputs (work_labeled/submission, pseudo-episodes 900-911) with the parallel
branch's validated labeled_object_scorer, under the same two alignments the branch reported:
  raw_camera     metric OpenCV camera poses as predicted
  human_aligned  one first-frame HUMAN-only Sim3 (our submitted MHR human at frame 0 -> GT SMPL-H mapped to MHR),
                 computed by patched copies of the branch's own sidecar scripts (only the predicted-body path changes)

  step 1 (--stage export): decode our submitted human at frame 0 (TorchScript MHR, kit flip) and write
         <W>/human_first/<name>/body_000000.npz, plus <W>/predictions/<dataset>_PREDICTIONS.json
  step 2 (--stage score):  apply <W>/align_<dataset>/TRANSFORMS.json and run the scorer for both alignments
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

W = Path("/home/tanuj/V2D_challenge/work_labeled")
BR = Path("/home/tanuj/V2D_challenge/experiments/offline_opt_20261020")
REQ = BR / "benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json"
REFS = dict(BEHAVE=BR / "benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json",
            InterCap=BR / "benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json")
PY = "/home/tanuj/.conda/envs/cari4d/bin/python"
FLIP = np.array([1.0, -1.0, -1.0])
SUB, TAG = "submission", ""     # --sub/--tag: score a variant (e.g. variants/s6_raw) into predictions_<tag>/scores_<tag>


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name_of(dataset, seq):
    return seq.replace("/", "_") if dataset == "InterCap" else seq


def export():
    import torch
    ts = torch.jit.load("/home/tanuj/V2D_challenge/models/mhr/mhr_model.pt", map_location="cpu").eval()
    eps = json.loads((W / "LABELED_EPISODES.json").read_text())
    preds = {"BEHAVE": [], "InterCap": []}
    for ep, info in eps.items():
        e = f"episode_{int(ep):06d}"; ds = info["dataset"]; seq = info["sequence"]
        npz = W / SUB / f"{e}.npz"
        if not npz.exists():
            print("MISSING", ep, seq, flush=True); continue
        d = np.load(npz)
        p = np.concatenate([d["pose"][:1], d["scales"][None]], 1).astype(np.float32)
        with torch.no_grad():
            v, _ = ts(torch.from_numpy(d["shape"].reshape(1, 45).astype(np.float32)), torch.from_numpy(p), torch.zeros(1, 72), True)
        verts = v[0].double().numpy() / 100 * FLIP
        hf = W / "human_first" / name_of(ds, seq); hf.mkdir(parents=True, exist_ok=True)
        np.savez(hf / "body_000000.npz", vertices_cam=verts)
        mesh = W / SUB / f"{e}_object.ply"
        R, t = d["object_rotation"], d["object_translation"]
        assert len(R) == len(info["frame_ids"])
        preds[ds].append(dict(sequence=seq, camera=1, mesh_path=str(mesh), mesh_sha256=sha(mesh), mesh_units="metres",
                              object_scale=float(d["object_scale"]),
                              pose_convention="X_cam = object_scale * (V_local @ R_obj2cam.T) + t_obj2cam_m",
                              origin_convention="SAM 3D Objects mesh origin as exported by S5; no correspondence to native GT mesh origin",
                              origin_correspondence_to_gt="unverified",
                              source=dict(pipeline="V2D main S1-S7 (v2.1 code)", episode=int(ep), npz=str(npz), npz_sha256=sha(npz),
                                          crop_xyxy=info["crop_xyxy"], GT_inputs=False),
                              frames=[dict(video_frame_index=int(f), R_obj2cam=R[i].tolist(), t_obj2cam_m=t[i].tolist())
                                      for i, f in enumerate(info["frame_ids"])]))
    (W / ("predictions" + TAG)).mkdir(exist_ok=True)
    for ds, rows in preds.items():
        (W / ("predictions" + TAG) / f"{ds}_PREDICTIONS.json").write_text(json.dumps(
            dict(schema="opencv-metric-object-predictions-v1", sequences=rows), indent=1, allow_nan=False))
        print(ds, len(rows), "sequences exported", flush=True)


def score(datasets):
    sys.path.insert(0, str(BR / "code"))
    from score_human_aligned_surrogate import compose
    out = W / ("scores" + TAG); out.mkdir(exist_ok=True)
    summary = {}
    for ds in datasets:
        pred_path = W / ("predictions" + TAG) / f"{ds}_PREDICTIONS.json"
        data = json.loads(pred_path.read_text())
        tr = json.loads((W / f"align_{ds}" / "TRANSFORMS.json").read_text())
        lookup = {x["sequence"]: x for x in tr["records"]}
        aligned = copy.deepcopy(data)
        for s in aligned["sequences"]:
            compose(s, lookup[s["sequence"]]["primary_transform"])
        ap = out / f"{ds}_human_aligned_PREDICTIONS.json"
        ap.write_text(json.dumps(aligned, indent=1, allow_nan=False))
        for alignment, p in [("raw_camera", pred_path), ("human_aligned", ap)]:
            res = out / f"{ds}_{alignment}_SCORES.json"
            argv = [PY, str(BR / "code/labeled_object_scorer.py"), "--prediction", str(p), "--dataset", ds,
                    "--requests", str(REQ), "--references", str(REFS[ds]), "--out", str(res)]
            if ds == "InterCap":
                argv += ["--intercap-source-fps", "30"]
            subprocess.run(argv, check=True)
            summary[f"{ds}/{alignment}"] = json.loads(res.read_text())
            print(ds, alignment, {k: v for k, v in summary[f"{ds}/{alignment}"].items() if k != "per_sequence"}, flush=True)
        summary[f"{ds}/human_fit_cm"] = {r["sequence"]: r["primary_human_fit_mean_vertex_residual_cm"] for r in tr["records"]}
    (out / f"SUMMARY_{'_'.join(datasets)}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["export", "score"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["BEHAVE", "InterCap"])
    ap.add_argument("--sub", default="submission"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    SUB, TAG = a.sub, (f"_{a.tag}" if a.tag else "")
    export() if a.stage == "export" else score(a.datasets)
