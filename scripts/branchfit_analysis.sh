#!/usr/bin/env bash
# usage: branchfit_analysis.sh <batch 0|1|2> <subdir> <tag>   (batch b = episodes 10b..10b+9)
EPS=$(seq -s, $((10*$1)) $((10*$1+9)))
cd /home/tanuj/V2D_challenge/code && /home/tanuj/.conda/envs/v2d/bin/python -I branchfit_analysis.py "$EPS" 80 "$2" "$3"
