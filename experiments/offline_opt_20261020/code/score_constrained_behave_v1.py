#!/usr/bin/env python3
"""Separate CPU scoring of all frozen trajectory arms, after inference finishes."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runtime = ROOT/'constrained_behave_v1'; receipt = runtime/'COMPLETE.json'
    complete = json.loads(receipt.read_text()); assert complete['full_frames'] == 384 and len(complete['predictions']) == 5
    results = []
    for row in complete['predictions']:
        pred = Path(row['prediction']); assert sha(pred) == row['prediction_sha256']
        output = pred.with_name(pred.name.replace('_PREDICTIONS', '_SCORES'))
        subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(pred),
            '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
            '--references', str(ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'),
            '--out', str(output)], check=True)
        results.append(dict(arm=row['arm'], scores=str(output), scores_sha256=sha(output)))
    subprocess.run([PY, str(ROOT/'code/score_human_aligned_surrogate.py'), '--transforms',
        str(ROOT/'human_alignment_sidecars_v1/TRANSFORMS.json'), '--source', str(runtime),
        '--out', str(runtime/'human_aligned')], check=True)
    (runtime/'SCORING_COMPLETE.json').write_text(json.dumps(dict(inference_receipt_sha256=sha(receipt),
        raw_camera_scores=results, human_aligned_scores=str(runtime/'human_aligned'), all_arms_retained=True,
        exact_hosted_metrics=False), indent=2))


if __name__ == '__main__': main()
