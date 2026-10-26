#!/usr/bin/env python3
"""SCORER ONLY: one first-frame human-derived Sim3 per complete trajectory.

The human mapping/GT decode is a separately validated receipt. This wrapper
never estimates a transform from object geometry, object labels or later frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PY = '/home/tanuj/.conda/envs/cari4d/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compose(sequence, transform):
    scale = float(transform['scale']); rotation = np.array(transform['R']); shift = np.array(transform['t'])
    assert scale > 0 and np.isfinite(scale)
    assert rotation.shape == (3, 3) and shift.shape == (3,) and np.isfinite(shift).all()
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(rotation), 1., atol=1e-6)
    sequence['object_scale'] *= scale
    for frame in sequence['frames']:
        frame['R_obj2cam'] = (rotation @ np.array(frame['R_obj2cam'])).tolist()
        frame['t_obj2cam_m'] = (scale * rotation @ np.array(frame['t_obj2cam_m']) + shift).tolist()


def self_test():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(123); v = rng.normal(size=(30, 3))
    r = Rotation.from_euler('xyz', [.2, .3, -.1]).as_matrix(); t = [.1, .2, 2.]
    rh = Rotation.from_euler('xyz', [-.1, .4, .3]).as_matrix(); th = [.2, -.3, .4]
    seq = dict(object_scale=2., frames=[dict(R_obj2cam=r.tolist(), t_obj2cam_m=t)])
    before = 2. * v @ r.T + t
    compose(seq, dict(scale=1.3, R=rh.tolist(), t=th))
    f = seq['frames'][0]; actual = seq['object_scale'] * v @ np.array(f['R_obj2cam']).T + f['t_obj2cam_m']
    assert np.allclose(actual, 1.3 * before @ rh.T + th, atol=1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--transforms', type=Path, required=True)
    ap.add_argument('--source', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True); args = ap.parse_args(); self_test()
    receipt = json.loads(args.transforms.read_text())
    assert receipt['GT_human_only_for_scoring'] and receipt['single_transform_per_clip']
    assert not receipt['native_object_GT_fields_accessed'] and not receipt['ICP_used']
    lookup = {x['sequence']: x for x in receipt['records']}
    assert len(lookup) == 4
    args.out.mkdir(exist_ok=False); reports = []
    for directory in args.source:
        for original in sorted(directory.glob('*_PREDICTIONS.json')):
            data = json.loads(original.read_text()); assert len(data['sequences']) == 4
            assert sum(len(s['frames']) for s in data['sequences']) == 384
            for seq in data['sequences']:
                record = lookup[seq['sequence']]
                assert record['RGB_frame'] == min(x['video_frame_index'] for x in seq['frames'])
                compose(seq, record['primary_transform'])
                seq['source']['scorer_only_human_alignment'] = dict(
                    transform_receipt_sha256=sha(args.transforms), original_prediction_sha256=sha(original),
                    first_frame_GT_human_used=True, object_GT_used_to_fit_transform=False,
                    same_transform_for_all_frames=True, production_inference_output=False)
            label = directory.parent.name + '_' + directory.name + '_' + original.name
            pred = args.out/label; pred.write_text(json.dumps(data, indent=2, allow_nan=False))
            raw = args.out/'raw_scorer'/label.replace('_PREDICTIONS', '_SCORES'); raw.parent.mkdir(exist_ok=True)
            subprocess.run([PY, str(ROOT/'code/labeled_object_scorer.py'), '--prediction', str(pred),
                '--requests', str(ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'),
                '--references', str(ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'),
                '--out', str(raw)], check=True)
            score = json.loads(raw.read_text())
            score.update(alignment='single fixed Sim3 from predicted first-frame MHR human to native GT SMPLH mapped into MHR; scorer-only surrogate',
                human_alignment_present=True, exact_hosted_metrics=False,
                human_alignment_receipt=str(args.transforms), human_alignment_receipt_sha256=sha(args.transforms),
                original_raw_scorer_sha256=sha(raw), limitation='Converted BEHAVE human GT, not native challenge MHR GT; engineering development cohort.')
            output = args.out/label.replace('_PREDICTIONS', '_SCORES'); output.write_text(json.dumps(score, indent=2))
            reports.append(dict(original=str(original), original_sha256=sha(original), prediction=str(pred),
                                prediction_sha256=sha(pred), scores=str(output), scores_sha256=sha(output)))
    assert reports
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(records=reports,
        transform_receipt_sha256=sha(args.transforms), script_sha256=sha(Path(__file__)),
        scorer_only=True, exact_hosted_metrics=False), indent=2))


if __name__ == '__main__': main()
