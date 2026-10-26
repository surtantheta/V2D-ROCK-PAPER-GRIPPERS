#!/usr/bin/env python3
"""Run frozen image-only pose repair, then separately evaluate every arm."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    runtime = ROOT/'labeled_predictions_v2'
    source = runtime/'human_anchored_comparison'
    output = runtime/'temporal_continuity_comparison'
    audit = runtime/'MASK_CONTINUITY_AUDIT.json'
    adapter = ROOT/'code/mask_continuity_ablation.py'
    scorer = ROOT/'code/labeled_object_scorer.py'
    request = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    refs = ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'
    originals = sorted(source.glob('*_PREDICTIONS.json'))
    assert len(originals) == 6
    for p in originals:
        data = json.loads(p.read_text())['sequences']
        assert len(data) == 4 and sum(len(s['frames']) for s in data) == 384
    subprocess.run([PY, str(adapter), '--source', str(source), '--out', str(output), '--audit', str(audit)], check=True)
    reports = []
    for p in sorted(output.glob('*_PREDICTIONS.json')):
        scores = p.with_name(p.name.replace('_PREDICTIONS', '_SCORES'))
        subprocess.run([PY, str(scorer), '--prediction', str(p), '--requests', str(request),
                        '--references', str(refs), '--out', str(scores)], check=True)
        reports.append(dict(prediction=str(p), prediction_sha256=sha(p), scores=str(scores), scores_sha256=sha(scores)))
    (output/'COMPLETE.json').write_text(json.dumps(dict(records=reports,
        source_sha256={str(p): sha(p) for p in [Path(__file__), adapter, scorer, audit, request, refs, *originals]},
        inference_GT_inputs=False, exact_hosted_metrics=False, post_failure_engineering_ablation=True), indent=2))


if __name__ == '__main__': main()
