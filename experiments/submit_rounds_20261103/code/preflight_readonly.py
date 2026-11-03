#!/usr/bin/env python3
"""Read-only Kaggle identity, daily allowance and submission history for all five Track 1 boards."""
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('pf', '/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/joint_kaggle_20261031/code/preflight_v1.py')
pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
r = pf.collect(require_available=False)
Path(sys.argv[1]).write_text(json.dumps(r, indent=1, default=str))
print(json.dumps({k: r['metrics'][k] for k in r['metrics']}, default=str, indent=1))
for k, rows in r['submissions_last_seven_days'].items():
    print(k, [(x['date_utc'][:16], x['public_score'], (x['description'] or '')[:60]) for x in rows])
