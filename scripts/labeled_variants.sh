#!/usr/bin/env bash
# Variants of OUR labeled outputs, scored like the main run: raw S6 track, branch fitter from S6, branch fitter from v2.1 (hybrid)
# usage: labeled_variants.sh BEHAVE|InterCap
set -e
DS=$1; if [ "$DS" = BEHAVE ]; then EPS="900 901 902 903"; else EPS="904 905 906 907 908 909 910 911"; fi
PY=/home/tanuj/.conda/envs/cari4d/bin/python; WL=/home/tanuj/V2D_challenge/work_labeled
cd /home/tanuj/V2D_challenge/code
$PY - "$EPS" <<'PYEOF'
import sys, shutil, numpy as np
from pathlib import Path
W = Path("/home/tanuj/V2D_challenge/work_labeled"); out = W / "variants/s6_raw"; out.mkdir(parents=True, exist_ok=True)
for ep in sys.argv[1].split():
    e = f"episode_{int(ep):06d}"; d = dict(np.load(W / f"submission/{e}.npz")); P = np.load(W / f"s6/{e}/poses.npz")["T_obj2cam"].astype(float)
    ok = np.isfinite(P).all(axis=(1, 2)) & (P[:, 2, 3] > .05); idx = np.flatnonzero(ok); t = P[:, :3, 3].copy(); R = P[:, :3, :3].copy()
    for c in range(3): t[~ok, c] = np.interp(np.flatnonzero(~ok), idx, t[ok, c])
    for i in np.flatnonzero(~ok): R[i] = P[idx[np.argmin(abs(idx - i))], :3, :3]
    d["object_translation"], d["object_rotation"] = t, R
    np.savez_compressed(out / f"{e}.npz", **d); shutil.copyfile(W / f"submission/{e}_object.ply", out / f"{e}_object.ply")
    print("s6_raw", ep, "invalid", int((~ok).sum()))
PYEOF
for EP in $EPS; do
  for B in s6 v21; do $PY branch_fit_episode.py --episode $EP --work $WL --v21 $WL/submission --out $WL/branchfit_$B --baseline $B; done
done
$PY labeled_score.py --stage score --datasets $DS >/dev/null 2>&1 || true
for V in variants/s6_raw branchfit_s6/identity_temporal branchfit_s6/shared_scale_contact branchfit_v21/identity_temporal branchfit_v21/fixedscale_geometry branchfit_v21/shared_scale branchfit_v21/shared_scale_contact branchfit_s6/fixedscale_geometry branchfit_s6/shared_scale; do
  T=$(echo $V | tr '/' '_')
  /home/tanuj/.conda/envs/v2d/bin/python labeled_score.py --stage export --sub $V --tag $T > /dev/null
  $PY labeled_score.py --stage score --datasets $DS --sub $V --tag $T | grep -E "human_aligned|raw_camera" | sed "s/^/$T /" | cut -c1-60
done
