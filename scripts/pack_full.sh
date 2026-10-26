#!/usr/bin/env bash
# Full 30-episode pack with the kit packer. COMMIT defaults to a format-valid placeholder (validation only, NOT for upload).
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/home/tanuj/.conda/envs/v2d/bin/python
K=/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit
SUB=/home/tanuj/V2D_challenge/work/${SUBDIR:-submission_v1}
OUT=/home/tanuj/V2D_challenge/work/pack_${TAG:-v1_test}; mkdir -p $OUT
COMMIT=${COMMIT:-https://github.com/placeholder/placeholder/commit/0000000000000000000000000000000000000000}
$PY -I /home/tanuj/V2D_challenge/code/pack_gate.py ${SUBDIR:-submission_v1}
cd $K
$PY eval_reconstruction.py --episodes $SUB --sample data/track_1_sample_submission.parquet --commit "$COMMIT" --out $OUT/track1.parquet
$PY -I - "$OUT/track1.parquet" "$K/data/track_1_sample_submission.parquet" <<'PY'
import sys, pandas as pd, numpy as np
d = pd.read_parquet(sys.argv[1]); s = pd.read_parquet(sys.argv[2])
v = d[["x","y","z"]].to_numpy()
print("packed rows", len(d), "sample rows", len(s), "same ids", bool((d["row_id"].values == s["row_id"].values).all()),
      "finite", bool(np.isfinite(v).all()), "commit", d["code_commit_url"].iloc[0])
PY
sha256sum $OUT/track1.parquet
