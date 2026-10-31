#!/usr/bin/env python3
"""Rebuild the challenge transport inputs from the published Pi3X parent run.

Use --check-only to replay input discovery and every hash without rewriting the
already frozen manifest. This reads predictions and prediction-only calibration,
not GT. Runtime roots intentionally match run_challenge_joint_v1.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / 'physical_refinement_20261031'
OLD = Path('/home/tanuj/V2D_challenge')
BASELINE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def build():
    legacy = OLD / 'experiments/identity_fusion_20261029'
    kit = OLD / 'data/assets/submission_kit/v2d_submission_kit'
    transport_inputs = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/INPUTS.json')
    parent = json.loads(transport_inputs.read_text())
    metadata_file = OLD / 'experiments/pi3x_kaggle_20261029/CHALLENGE_EPISODES.json'
    metadata = json.loads(metadata_file.read_text())
    paths = [
        ROOT / 'code/run_challenge_joint_v1.py',
        PRIOR / 'code/joint_runner_v3.py',
        PRIOR / 'code/joint_core_v1.py',
        PRIOR / 'code/temporal_v2.py',
        PRIOR / 'code/test_joint_core_v1.py',
        PRIOR / 'JOINT_CORE_TEST_V1.json',
        PRIOR / 'JOINT_INPUTS_V3.json',
        ROOT.parent / 'validation_20261031/GENUINE_JOINT_FULL_V3_OUTPUT_REVIEW.json',
        legacy / 'code/evaluate_penetration.py',
        legacy / 'code/run_real_mesh_fit_v3.py',
        legacy / 'code/real_mesh_optimizer.py',
        kit / 'metric_code/track_1/PEN.py',
        kit / 'v2dlb/mesh_budget.py',
        OLD / 'models/mhr/mhr_model.pt',
        OLD / 'models/mhr/mhr_faces_lod1.npy',
        OLD / 'experiments/post_feedback_20261026/human_penetration_audit/run_v2_cari4d/HAND_VERTEX_IDS.npy',
        transport_inputs,
        metadata_file,
        OLD / 'work/MASK_CHOICE.json',
    ]
    sources = {str(p): sha(p) for p in paths}
    case_pins, cases = {}, []
    for case in parent['cases']:
        ep = case['episode']
        assert ep == len(cases)
        n = case['total_frames']
        meta = metadata[str(ep)]
        assert meta['frame_ids'] == list(range(n)) and len(meta['times']) == n
        case['paths']['baseline'] = str(BASELINE / f'episode_{ep:06d}.npz')
        case['paths']['mesh'] = str(BASELINE / f'episode_{ep:06d}_object.ply')
        paths = [Path(x) for x in case['paths'].values()]
        paths.append(Path(case['paths']['poses']).parent / 'TRACK.json')
        for sample in case['samples']:
            frame = sample['frame']
            assert frame['native_frame_id'] == frame['local_index']
            assert abs(frame['time_seconds'] - frame['local_index'] / 30) < 1e-6
            paths.extend([Path(sample['mask']), Path(frame['rgb'])])
        case_pins[str(ep)] = {str(p): sha(p) for p in paths}
        cases.append(case)
    assert len(cases) == 30 and sum(c['total_frames'] for c in cases) == 16563
    return dict(
        schema='challenge30-genuine-joint-v1', cases=cases, metadata=metadata,
        source_sha256=sources, case_sha256=case_pins, baseline=str(BASELINE),
        native_frame_policy='allframes0..N-1,30fps',
        source_policy_sha256=sha(PRIOR / 'JOINT_INPUTS_V3.json'),
        optimizer='unchanged joint_core_v1.Config defaults λ64,C10,2mmmargin,20mmbound; originalV3ImageDepthGuard',
        all_5840_hand_points_all_frames=True, GT_used=False, GT_read=False,
        output='/rhome/tanuj/v2d_joint_kaggle_20261031',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    candidate = build()
    target = ROOT / 'FIT_INPUTS_V1.json'
    encoded = json.dumps(candidate, indent=2) + '\n'
    if args.check_only:
        assert candidate == json.loads(target.read_text()), 'Replay changed the frozen manifest'
        assert encoded.encode() == target.read_bytes(), 'Replay serialization differs'
        status = 'PASS_EXACT_FROZEN_INPUT_REPLAY'
    else:
        assert not target.exists(), 'Preserve frozen inputs; use a new version for changes'
        target.write_text(encoded)
        status = 'FROZEN'
    print(json.dumps(dict(status=status, manifest=str(target), sha256=sha(target), cases=30,
                          frames=16563, global_pins=len(candidate['source_sha256']),
                          case_pins=sum(len(x) for x in candidate['case_sha256'].values()), GT_read=False)), flush=True)


if __name__ == '__main__':
    main()
