#!/usr/bin/env python3
"""Score all fresh InterCap arms with native GT; never feed GT into inference."""
import hashlib
import json
from pathlib import Path
import subprocess

from score_human_aligned_surrogate import compose, self_test

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(prediction, output):
    subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(prediction),
        '--dataset', 'InterCap', '--intercap-source-fps', '30',
        '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
        '--references', str(ROOT/'benchmarks/labeled_dev_v2/scorer/INTERCAP_LABEL_REFERENCES_VERIFIED.json'),
        '--out', str(output)], check=True)
    data = json.loads(output.read_text())
    assert data['complete_coverage'] and len(data['per_sequence']) == 8
    assert sum(x['scored_frames'] for x in data['per_sequence']) == 246


def main():
    self_test(); output = ROOT/'intercap_results_v1'; output.mkdir(exist_ok=False)
    receipt = ROOT/'constrained_intercap_v1/COMPLETE.json'; completed = json.loads(receipt.read_text())
    assert completed['full_frames'] == 246 and len(completed['predictions']) == 5
    predictions = {'foundationpose_raw': ROOT/'labeled_intercap_v1/baseline/foundationpose_raw_PREDICTIONS.json'}
    for row in completed['predictions']:
        p = Path(row['prediction']); assert sha(p) == row['prediction_sha256']; predictions[row['arm']] = p
    assert len(predictions) == 6
    transform_path = ROOT/'intercap_human_alignment_sidecars_v1/TRANSFORMS.json'
    transforms = json.loads(transform_path.read_text()); lookup = {x['sequence']: x for x in transforms['records']}
    assert len(lookup) == 8 and not transforms['native_object_GT_fields_accessed']
    reports = []
    for folder in ['raw_camera', 'human_aligned', 'human_aligned/raw_scorer']: (output/folder).mkdir(parents=True, exist_ok=True)
    for arm, prediction in predictions.items():
        data = json.loads(prediction.read_text()); sequences = data['sequences']
        assert len(sequences) == len({x['sequence'] for x in sequences}) == 8
        assert set(x['sequence'] for x in sequences) == set(lookup)
        raw_score = output/'raw_camera'/f'{arm}_SCORES.json'; score(prediction, raw_score)
        for sequence in sequences:
            record = lookup[sequence['sequence']]
            assert min(x['video_frame_index'] for x in sequence['frames']) == 0
            compose(sequence, record['primary_transform'])
            sequence['source']['scorer_only_human_alignment'] = dict(transform_receipt_sha256=sha(transform_path),
                original_prediction_sha256=sha(prediction), first_frame_GT_human_used=True,
                object_GT_used_to_fit_transform=False, same_transform_for_all_frames=True,
                production_inference_output=False)
        aligned_pred = output/'human_aligned'/f'{arm}_PREDICTIONS.json'
        aligned_pred.write_text(json.dumps(data, indent=2, allow_nan=False))
        raw_aligned_score = output/'human_aligned/raw_scorer'/f'{arm}_SCORES.json'; score(aligned_pred, raw_aligned_score)
        adjusted = json.loads(raw_aligned_score.read_text())
        adjusted.update(alignment='one first-frame human-only Sim3; native InterCap SMPLH mapped into MHR',
            human_alignment_present=True, exact_hosted_metrics=False,
            human_alignment_receipt_sha256=sha(transform_path), original_raw_scorer_sha256=sha(raw_aligned_score),
            limitation='engineering development; MHR-mapped human GT surrogate; nominal30fps/stride5; object origins not certified corresponding')
        final_score = output/'human_aligned'/f'{arm}_SCORES.json'; final_score.write_text(json.dumps(adjusted, indent=2))
        reports.append(dict(arm=arm, prediction=str(prediction), prediction_sha256=sha(prediction),
            raw_camera_scores=str(raw_score), raw_camera_scores_sha256=sha(raw_score),
            human_aligned_prediction=str(aligned_pred), human_aligned_prediction_sha256=sha(aligned_pred),
            human_aligned_scores=str(final_score), human_aligned_scores_sha256=sha(final_score)))
    (output/'COMPLETE.json').write_text(json.dumps(dict(records=reports, exact_hosted_metrics=False,
        transform_receipt_sha256=sha(transform_path), inference_receipt_sha256=sha(receipt),
        all_arms_retained=True, frames_per_arm=246, takes_per_arm=8,
        inference_native_GT_or_CAD=False, report_reserve_included=False), indent=2))


if __name__ == '__main__': main()
