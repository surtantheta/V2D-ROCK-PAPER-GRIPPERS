#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
K=/home/tanuj/.conda/envs/v2d/bin/kaggle
for c in cd-h cd-o acc-h acc-o pen; do echo "== v2d-challenge-track1-$c"; $K competitions leaderboard -c v2d-challenge-track1-$c -s 2>&1 | head -25; done
