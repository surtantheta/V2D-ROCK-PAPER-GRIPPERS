#!/usr/bin/env bash
# Pack test on a subset of episodes: sample filtered to those episodes, format-valid placeholder commit (NOT uploaded).
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/home/tanuj/.conda/envs/v2d/bin/python
K=/home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit
OUT=/home/tanuj/V2D_challenge/work/pack_test; mkdir -p $OUT
$PY -I - "$K/data/track_1_sample_submission.parquet" "$OUT/sample_subset.parquet" $EPS <<'PY'
import sys, pandas as pd
src, dst, *eps = sys.argv[1:]
s = pd.read_parquet(src)
keep = s["row_id"].str.split("_").str[1].astype(int).isin([int(e) for e in eps])
s[keep].to_parquet(dst, index=False); print("subset rows", int(keep.sum()), "of", len(s))
PY
cd $K
$PY eval_reconstruction.py --episodes /home/tanuj/V2D_challenge/work/submission_v0 --sample $OUT/sample_subset.parquet \
   --commit https://github.com/placeholder/placeholder/commit/0000000000000000000000000000000000000000 --out $OUT/track1_subset.parquet
$PY -I - "$OUT/track1_subset.parquet" <<'PY'
import sys, pandas as pd, numpy as np
d = pd.read_parquet(sys.argv[1]); v = d[["x","y","z"]].to_numpy()
print("packed rows", len(d), "finite", bool(np.isfinite(v).all()), "commit", d["code_commit_url"].iloc[0])
PY
