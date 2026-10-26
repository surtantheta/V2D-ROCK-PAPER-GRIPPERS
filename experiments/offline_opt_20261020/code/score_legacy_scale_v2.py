#!/usr/bin/env python3
"""Scorer-only evaluation of four frozen fits from matched legacy trajectories."""
import copy
import json
from pathlib import Path
import subprocess

from run_identity_scale_v2 import configuration, sha
from run_legacy_scale_v2 import ARMS
from score_human_aligned_surrogate import compose, self_test

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def main():
    self_test()
    inference = ROOT/'identity_scale_v2/legacy_optimized/COMPLETE.json'
    completed = json.loads(inference.read_text())
    assert completed['full_frames_per_arm'] == 630 and completed['arms'] == ARMS
    assert not completed['GT_inputs'] and not completed['reserve_included']
    output = ROOT/'identity_scale_v2/legacy_scores'; output.mkdir(exist_ok=False)
    reports = []
    sources = {str(p): sha(p) for p in [Path(__file__), inference,
        ROOT/'code/labeled_object_scorer.py', ROOT/'code/score_human_aligned_surrogate.py']}
    assert {x['dataset'] for x in completed['records']} == {'BEHAVE', 'InterCap'}
    for group in completed['records']:
        dataset = group['dataset']; cfg = configuration(dataset)
        assert group['frames_per_arm'] == cfg['frames'] and group['takes_per_arm'] == cfg['count']
        assert [x['arm'] for x in group['records']] == ARMS
        prefix = '' if dataset == 'BEHAVE' else 'intercap_'
        transforms_path = ROOT/f'{prefix}human_alignment_sidecars_v1/TRANSFORMS.json'
        references = ROOT/'benchmarks/labeled_dev_v2/scorer'/(
            'BEHAVE_LABEL_REFERENCES_PINNED.json' if dataset == 'BEHAVE' else 'INTERCAP_LABEL_REFERENCES_VERIFIED.json')
        sources.update({str(p): sha(p) for p in [transforms_path, references]})
        transforms = json.loads(transforms_path.read_text())
        assert transforms['GT_human_only_for_scoring'] and transforms['single_transform_per_clip']
        assert not transforms['native_object_GT_fields_accessed'] and not transforms['ICP_used']
        lookup = {x['sequence']: x for x in transforms['records']}
        assert len(lookup) == cfg['count']
        for record in group['records']:
            path = Path(record['prediction']); assert sha(path) == record['prediction_sha256']
            data = json.loads(path.read_text()); names = [x['sequence'] for x in data['sequences']]
            assert len(names) == len(set(names)) == cfg['count'] and set(names) == set(lookup)
            assert sum(len(x['frames']) for x in data['sequences']) == cfg['frames']
            for alignment in ['raw_camera', 'human_aligned']:
                folder = output/dataset/alignment; folder.mkdir(parents=True, exist_ok=True)
                if alignment == 'human_aligned':
                    data = copy.deepcopy(data)
                    for sequence in data['sequences']:
                        transform = lookup[sequence['sequence']]
                        first = min(x['video_frame_index'] for x in sequence['frames'])
                        assert first == (transform['RGB_frame'] if dataset == 'BEHAVE' else 0)
                        compose(sequence, transform['primary_transform'])
                        sequence['source']['scorer_only_human_alignment'] = dict(
                            original_prediction_sha256=sha(path), transform_receipt_sha256=sha(transforms_path),
                            object_GT_used_to_fit_transform=False, same_transform_for_all_frames=True,
                            production_inference_output=False)
                    prediction = folder/f'{record["arm"]}_PREDICTIONS.json'
                    prediction.write_text(json.dumps(data, indent=2, allow_nan=False))
                else:
                    prediction = path
                raw = folder/'raw_scorer'/f'{record["arm"]}_SCORES.json'
                raw.parent.mkdir(exist_ok=True)
                argv = [PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(prediction),
                    '--dataset', dataset, '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
                    '--references', str(references), '--out', str(raw)]
                if dataset == 'InterCap':
                    argv += ['--intercap-source-fps', '30']
                subprocess.run(argv, check=True)
                score = json.loads(raw.read_text())
                assert score['complete_coverage'] and len(score['per_sequence']) == cfg['count']
                assert sum(x['scored_frames'] for x in score['per_sequence']) == cfg['frames']
                score.update(alignment=alignment, human_alignment_present=alignment == 'human_aligned',
                    exact_hosted_metrics=False, full_cohort_retained=True,
                    limitation='Engineering development; mapped-human GT alignment surrogate; nominal InterCap source fps.',
                    original_raw_scorer_sha256=sha(raw))
                if alignment == 'human_aligned':
                    score['human_alignment_receipt_sha256'] = sha(transforms_path)
                final = folder/f'{record["arm"]}_SCORES.json'
                final.write_text(json.dumps(score, indent=2, allow_nan=False))
                reports.append(dict(dataset=dataset, arm=record['arm'], alignment=alignment,
                    original_prediction=str(path), original_prediction_sha256=sha(path),
                    scores=str(final), scores_sha256=sha(final)))
    assert len(reports) == 16
    (output/'COMPLETE.json').write_text(json.dumps(dict(records=reports, source_sha256=sources,
        inference_receipt_sha256=sha(inference), inference_GT_inputs=False,
        report_reserve_included=False, exact_hosted_metrics=False), indent=2))


if __name__ == '__main__':
    main()
