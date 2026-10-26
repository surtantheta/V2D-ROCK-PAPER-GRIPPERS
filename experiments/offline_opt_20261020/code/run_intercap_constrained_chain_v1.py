#!/usr/bin/env python3
"""Inference-only scaling and transfer; scoring is a separate dependent job."""
from pathlib import Path
import subprocess

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def main():
    for script in ['prepare_intercap_scaled_predictions_v1.py', 'run_constrained_intercap_v1.py']:
        subprocess.run([PY, str(ROOT/'code'/script)], check=True)


if __name__ == '__main__': main()
