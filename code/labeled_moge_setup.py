#!/usr/bin/env python3
"""MoGe-depth variant of the labeled BEHAVE run (work_labeled_moge): identical frames, masks and prompts as work_labeled,
but our focal is replaced by the branch's RGB-only MoGe focal (K_fixed_source, 1076.3 px) and, after s3full, the
MetaV3RSE depth in s3_human/*.npz is replaced by the branch's MoGe depth (same 512x384 grid, same local frame order).
Everything downstream (depth_scale to the SAM 3D Body human, S5, S6, S7) is unchanged.
  --stage setup   copy frames/masks/LABELED_EPISODES, write FOCALS.json
  --stage inject  overwrite 'depth' in s3_human/episode_X/*.npz (keeps a .metav3rse backup of the original array)"""
import argparse, json, shutil
from pathlib import Path
import numpy as np

SRC = Path("/home/tanuj/V2D_challenge/work_labeled"); W = Path("/home/tanuj/V2D_challenge/work_labeled_moge")
MOGE = Path("/home/tanuj/V2D_challenge/experiments/offline_opt_20261020/labeled_predictions_v1/depth")
EPS = [900, 901, 902, 903]

ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["setup", "inject"], required=True); a = ap.parse_args()
info = json.loads((SRC / "LABELED_EPISODES.json").read_text())
if a.stage == "setup":
    W.mkdir(exist_ok=True)
    (W / "LABELED_EPISODES.json").write_text(json.dumps({str(e): info[str(e)] for e in EPS}, indent=1))
    f = {}
    for e in EPS:
        E = f"episode_{e:06d}"
        for d in ("frames", "masks"):
            if not (W / d / E).exists():
                shutil.copytree(SRC / d / E, W / d / E)
        run = json.loads((MOGE / info[str(e)]["sequence"] / "RUN.json").read_text())
        f[str(e)] = float(run["K_fixed_source"][0][0])
    (W / "FOCALS.json").write_text(json.dumps(dict(method="branch RGB-only MoGe K_fixed_source (via SAM3D compute_pointmap)", episode_focal_px=f), indent=1))
    # masks are referenced by absolute path in MASKS.json -> repoint to the copies
    for e in EPS:
        p = W / f"masks/episode_{e:06d}/MASKS.json"; s = p.read_text().replace(str(SRC), str(W)); p.write_text(s)
    print("focals", f)
else:
    for e in EPS:
        E = f"episode_{e:06d}"; seq = info[str(e)]["sequence"]
        for i in range(info[str(e)]["frames"]):
            p = W / f"s3_human/{E}/{i:06d}.npz"; d = dict(np.load(p))
            m = np.load(MOGE / seq / f"{i:06d}.npz")["depth"].astype(np.float32)
            assert m.shape == d["depth"].shape
            d["depth_metav3rse"] = d["depth"]; d["depth"] = m
            np.savez_compressed(p, **d)
        print("injected MoGe depth", e, seq, info[str(e)]["frames"])
