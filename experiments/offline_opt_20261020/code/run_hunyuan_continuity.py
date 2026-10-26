#!/usr/bin/env python3
"""Apply the SAME frozen identity gate to both generators and image selection."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runtime = ROOT/'labeled_hunyuan_placement_v1'
    source = runtime/'comparison'; output = runtime/'temporal_comparison'
    audit = ROOT/'labeled_predictions_v2/MASK_CONTINUITY_AUDIT.json'
    assert (source/'COMPLETE.json').is_file()
    subprocess.run([PY, str(ROOT/'code/mask_continuity_ablation.py'), '--source', str(source),
                    '--out', str(output), '--audit', str(audit)], check=True)
    scores = []
    for p in sorted(output.glob('*_PREDICTIONS.json')):
        target = p.with_name(p.name.replace('_PREDICTIONS', '_SCORES'))
        subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(p),
            '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
            '--references', str(ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'),
            '--out', str(target)], check=True)
        scores.append(dict(prediction=str(p), prediction_sha256=sha(p), scores=str(target), scores_sha256=sha(target)))
    assert len(scores) == 6
    (output/'COMPLETE.json').write_text(json.dumps(dict(records=scores, audit_sha256=sha(audit),
        source_completion_sha256=sha(source/'COMPLETE.json'), all_frames_retained=True,
        inference_GT_inputs=False, exact_hosted_metrics=False), indent=2))


if __name__ == '__main__': main()
