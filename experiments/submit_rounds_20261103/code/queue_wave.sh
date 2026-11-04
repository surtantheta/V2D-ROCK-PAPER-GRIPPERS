#!/bin/bash
# Queue a chain of round uploads that starts at a given time (PBS -a), one job after another (afterany).
# Usage: queue_wave.sh <start YYYYMMDDhhmm> <plan file>; plan lines: TAG|parquet|commit sha40|note (no commas in note)
# Every parquet's directory must hold a VALIDATION.json with status PASS_ROUND_TRIP and the parquet's sha256.
set -euo pipefail
START=$1; PLAN=$2
R=/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/submit_rounds_20261103
PY=/home/tanuj/.conda/envs/v2d/bin/python
prev=""
while IFS='|' read -r TAG IN COMMIT NOTE; do
  [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ] && continue
  test -f "$R/SOURCE_RECEIPT_${COMMIT:0:8}.json"
  test ! -e "$R/rounds/$TAG/KAGGLE_RECEIPT.json"
  case "$NOTE" in *,*) echo "comma in note: $TAG"; exit 1;; esac
  $PY - "$IN" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); v = json.loads((p.parent / 'VALIDATION.json').read_text())
assert v['status'] == 'PASS_ROUND_TRIP', v['status']
assert v['parquet_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest(), 'validation is for another file'
PY
  if [ -z "$prev" ]; then dep=(-a "$START"); else dep=(-W "depend=afterany:$prev"); fi
  prev=$(qsub "${dep[@]}" -l select=1:ncpus=2:mem=12gb:host=cvml07 -v IN="$IN",TAG="$TAG",COMMIT="$COMMIT",NOTE="$NOTE" "$R/upload.pbs")
  echo "$TAG $prev"
done < "$PLAN"
