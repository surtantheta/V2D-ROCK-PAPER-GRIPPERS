#!/usr/bin/env python3
"""Gate before packing the 30-episode submission: every episode must have a FRESH S7 output (S7 .ok newer than the
npz's upstream inputs, no .failed marker for any stage), finite arrays covering the episode length, and its
ASSEMBLE json. Prints a per-episode table (object source, low-confidence flag, static fraction) and exits 1 on any
blocking problem. No Kaggle access.
"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path("/home/tanuj/V2D_challenge/work")
SUB = W / (sys.argv[1] if len(sys.argv) > 1 else "submission_v1")
lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"]
           for l in open("/home/tanuj/V2D_challenge/data/video_to_data_challenge/track_1/meta/episodes.jsonl")}
bad = []
for ep in range(30):
    e = f"episode_{ep:06d}"
    st = W / "STATUS"
    fails = sorted(p.name for p in st.glob(f"*_{ep}.failed"))
    npz, ply, asm = SUB / f"{e}.npz", SUB / f"{e}_object.ply", SUB / f"ASSEMBLE_{ep:06d}.json"
    row = dict(ep=ep, failed_markers=fails)
    if not (npz.exists() and ply.exists() and asm.exists()):
        row["problem"] = "missing S7 output"
    else:
        d = np.load(npz)
        a = json.loads(asm.read_text())
        T = lengths[ep]
        finite = all(np.isfinite(d[k]).all() for k in d.files)
        okshape = d["pose"].shape == (T, 136) and d["object_rotation"].shape == (T, 3, 3) and d["object_translation"].shape == (T, 3)
        s7ok = st / f"s7_{ep}.ok"
        fresh = s7ok.exists() and s7ok.stat().st_mtime >= npz.stat().st_mtime - 5
        row.update(source=a.get("object_source"), low_conf=a.get("low_confidence_track", False),
                   static=round(a.get("static_frac", 0), 2), finite=finite, shape_ok=okshape, s7_marker_fresh=fresh)
        if not (finite and okshape and fresh):
            row["problem"] = "invalid arrays or stale S7"
    if fails:
        row["problem"] = row.get("problem", "") + " failed-stage-markers"
    if row.get("problem"):
        bad.append(ep)
    print(json.dumps(row), flush=True)
print("BLOCKING:", bad if bad else "none")
sys.exit(1 if bad else 0)
