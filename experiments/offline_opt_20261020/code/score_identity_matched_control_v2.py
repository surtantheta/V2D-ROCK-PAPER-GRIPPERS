#!/usr/bin/env python3
"""Scorer-only same-hardware legacy replay diagnostic; frozen v2 arms unchanged."""
import copy
import json
from pathlib import Path
import subprocess

from run_identity_scale_v2 import configuration, sha
from score_human_aligned_surrogate import compose, self_test

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def main():
    self_test(); runtime = ROOT/'identity_scale_v2/matched_control'
    receipt = runtime/'COMPLETE.json'; completed = json.loads(receipt.read_text())
    assert len(completed['records']) == 2
    output = ROOT/'identity_scale_v2/matched_control_scores'; output.mkdir(exist_ok=False)
    reports = []
    for dataset in ['BEHAVE', 'InterCap']:
        cfg = configuration(dataset)
        record = next(x for x in completed['records'] if x['dataset'] == dataset)
        path = Path(record['prediction']); assert sha(path) == record['prediction_sha256']
        data = json.loads(path.read_text())
        assert len(data['sequences']) == cfg['count']
        assert sum(len(x['frames']) for x in data['sequences']) == cfg['frames']
        prefix = '' if dataset == 'BEHAVE' else 'intercap_'
        transform_path = ROOT/f'{prefix}human_alignment_sidecars_v1/TRANSFORMS.json'
        transforms = json.loads(transform_path.read_text())
        assert transforms['GT_human_only_for_scoring'] and transforms['single_transform_per_clip']
        assert not transforms['native_object_GT_fields_accessed'] and not transforms['ICP_used']
        lookup = {x['sequence']: x for x in transforms['records']}
        assert len(lookup) == cfg['count'] and set(lookup) == {x['sequence'] for x in data['sequences']}
        references = ROOT/'benchmarks/labeled_dev_v2/scorer'/(
            'BEHAVE_LABEL_REFERENCES_PINNED.json' if dataset == 'BEHAVE' else 'INTERCAP_LABEL_REFERENCES_VERIFIED.json')
        for alignment in ['raw_camera', 'human_aligned']:
            folder = output/dataset/alignment; folder.mkdir(parents=True)
            if alignment == 'human_aligned':
                aligned = copy.deepcopy(data)
                for sequence in aligned['sequences']:
                    transform = lookup[sequence['sequence']]
                    assert min(x['video_frame_index'] for x in sequence['frames']) == (
                        transform['RGB_frame'] if dataset == 'BEHAVE' else 0)
                    compose(sequence, transform['primary_transform'])
                    sequence['source']['scorer_only_human_alignment'] = dict(
                        transform_receipt_sha256=sha(transform_path), original_prediction_sha256=sha(path),
                        object_GT_used_to_fit_transform=False, same_transform_for_all_frames=True,
                        production_inference_output=False)
                prediction = folder/'legacy_matched_PREDICTIONS.json'
                prediction.write_text(json.dumps(aligned, indent=2, allow_nan=False))
            else:
                prediction = path
            raw = folder/'raw_scorer/SCORES.json'; raw.parent.mkdir()
            argv = [PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(prediction),
                '--dataset', dataset, '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
                '--references', str(references), '--out', str(raw)]
            if dataset == 'InterCap':
                argv += ['--intercap-source-fps', '30']
            subprocess.run(argv, check=True)
            scores = json.loads(raw.read_text())
            assert scores['complete_coverage'] and len(scores['per_sequence']) == cfg['count']
            assert sum(x['scored_frames'] for x in scores['per_sequence']) == cfg['frames']
            scores.update(alignment=alignment, human_alignment_present=alignment == 'human_aligned',
                exact_hosted_metrics=False, original_raw_scorer_sha256=sha(raw),
                diagnostic='same-hardware legacy rerun, separate from frozen six optimizer arms')
            result = folder/'legacy_matched_SCORES.json'
            result.write_text(json.dumps(scores, indent=2, allow_nan=False))
            reports.append(dict(dataset=dataset, alignment=alignment, scores=str(result), scores_sha256=sha(result),
                                original_prediction=str(path), original_prediction_sha256=sha(path)))
    assert len(reports) == 4
    (output/'COMPLETE.json').write_text(json.dumps(dict(records=reports,
        inference_receipt_sha256=sha(receipt), scorer_only=True, exact_hosted_metrics=False,
        source_sha256={str(Path(__file__)): sha(__file__)}), indent=2))


if __name__ == '__main__':
    main()
