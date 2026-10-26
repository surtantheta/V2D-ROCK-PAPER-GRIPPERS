#!/usr/bin/env python3
"""CPU scoring after direct tracking, with and without human-only alignment."""
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def main():
    runtime = ROOT/'labeled_association_tracking_v1'
    receipt = runtime/'TRACKING_COMPLETE.json'; data = json.loads(receipt.read_text())
    assert data['full_coverage'] == 384 and not data['pose_imputation']
    prediction = runtime/'comparison/foundationpose_association_PREDICTIONS.json'
    subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(prediction),
        '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
        '--references', str(ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'),
        '--out', str(prediction.with_name(prediction.name.replace('_PREDICTIONS', '_SCORES')))], check=True)
    subprocess.run([PY, str(ROOT/'code/score_human_aligned_surrogate.py'), '--transforms',
        str(ROOT/'human_alignment_sidecars_v1/TRANSFORMS.json'), '--source', str(runtime/'comparison'),
        '--out', str(runtime/'human_aligned_scores')], check=True)
    (runtime/'COMPLETE.json').write_text(json.dumps(dict(tracking_receipt=str(receipt),
        camera_scores=str(prediction.with_name(prediction.name.replace('_PREDICTIONS', '_SCORES'))),
        human_aligned_scores=str(runtime/'human_aligned_scores'), full_coverage=384,
        exact_hosted_metrics=False, pose_imputation=False), indent=2))


if __name__ == '__main__': main()
