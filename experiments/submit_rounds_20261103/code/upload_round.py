#!/usr/bin/env python3
"""One full Track 1 round: stamp the published commit URL, validate officially, upload to all five boards.

The input Parquet's x/y/z must be unchanged by the stamp; the commit must have a verified source receipt.
The official uploader reserves its receipt before uploading and is never retried automatically; a round
whose receipt already exists is refused.
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/submit_rounds_20261103')
KIT = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/data/assets/submission_kit/v2d_submission_kit')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('PBS_JOBID'), 'uploads run in PBS'
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--commit', required=True, help='40-hex commit with a SOURCE_RECEIPT_<sha8>.json')
    ap.add_argument('--note', required=True); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    receipt_src = json.loads((R / f'SOURCE_RECEIPT_{a.commit[:8]}.json').read_text())
    assert receipt_src['commit_sha'] == a.commit and receipt_src['remote_file_hashes_verified'] > 0
    url = receipt_src['commit_url']
    out_dir = R / 'rounds' / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f'track1_{a.tag}.parquet'
    receipt = out_dir / 'KAGGLE_RECEIPT.json'
    assert not receipt.exists(), 'this round already has a receipt; inspect it instead of re-uploading'
    src = pd.read_parquet(a.parquet)
    df = src.copy(); df['code_commit_url'] = url
    df.to_parquet(final, index=False)
    back = pd.read_parquet(final)
    assert back[['x', 'y', 'z']].equals(src[['x', 'y', 'z']]) and back['row_id'].equals(src['row_id'])
    spec = importlib.util.spec_from_file_location('v2d_submit_official', KIT / 'submit/v2d_submit.py')
    off = importlib.util.module_from_spec(spec); sys.modules[spec.name] = off; spec.loader.exec_module(off)
    off.validate_submission(final, 'track_1', json.loads((KIT / 'config/tracks.json').read_text()))
    message = f'Rock Paper Grippers - {a.note} - {a.commit[:8]} {sha(final)[:8]}'
    meta = {'tag': a.tag, 'input': a.parquet, 'input_sha256': sha(a.parquet), 'file': str(final),
            'file_sha256': sha(final), 'commit_url': url, 'message': message, 'PBS_JOBID': os.environ['PBS_JOBID']}
    (out_dir / 'ROUND.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    cmd = [sys.executable, '-I', '-B', '-u', str(KIT / 'submit/v2d_submit.py'), str(final), '--track', 'track_1',
           '-m', message]
    cmd += ['--dry-run'] if a.dry_run else ['--receipt', str(receipt), '--wait']
    rc = subprocess.run(cmd, cwd=KIT).returncode
    if receipt.exists():
        rec = json.loads(receipt.read_text())
        meta['scores'] = {k: e.get('public_score') for k, e in rec['submissions'].items()}
        meta['states'] = {k: e.get('state') for k, e in rec['submissions'].items()}
        meta['refs'] = {k: e.get('ref') for k, e in rec['submissions'].items()}
    meta['uploader_return_code'] = rc
    (out_dir / 'ROUND.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta.get('scores')), flush=True)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
