#!/usr/bin/env bash
set -e
PY=/home/tanuj/.conda/envs/cari4d/bin/python
cd /home/tanuj/V2D_challenge/code
/home/tanuj/.conda/envs/v2d/bin/python labeled_score.py --stage export
rm -rf /home/tanuj/V2D_challenge/work_labeled/align_BEHAVE
$PY labeled_align_behave.py
$PY labeled_score.py --stage score --datasets BEHAVE
