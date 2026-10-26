#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
for ep in ${EPS//_/ }; do
  /home/tanuj/.conda/envs/v2d/bin/python /home/tanuj/V2D_challenge/code/qa_overlay.py --episode $ep --submission /home/tanuj/V2D_challenge/work/${SUBDIR:-submission_v1} --out /home/tanuj/V2D_challenge/work/qa/overlays_${QTAG:-v1} --n 8 > /dev/null 2>&1 && echo "ok $ep" || echo "FAIL $ep"
done
