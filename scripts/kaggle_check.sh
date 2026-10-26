#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
K=/home/tanuj/.conda/envs/v2d/bin/kaggle
echo "== kaggle version"; $K --version 2>&1 | tail -1
echo "== competitions matching v2d-challenge-track1"; $K competitions list -s v2d-challenge-track1 2>&1 | tail -n 8
for c in cd-h cd-o acc-h acc-o pen; do
  echo "== submissions v2d-challenge-track1-$c (read-only; 403 => rules not accepted)"
  $K competitions submissions -c v2d-challenge-track1-$c 2>&1 | tail -n 3
done
