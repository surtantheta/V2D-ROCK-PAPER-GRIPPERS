#!/usr/bin/env python3
"""Challenge transport of the independently exercised genuine-joint V3 fit.

No numerical optimizer or observation-guard changes. The fit function is the
original source with its local-labeled metadata read replaced by a frozen
challenge-native metadata lookup. Its baseline directory points to the actual
30-clip Pi3X submission producer. All frames, not just scored frames, are fit.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LARGE = Path('/rhome/tanuj/v2d_joint_kaggle_20261031')
PRIOR = Path('/rhome/tanuj/home_moved_20261029/V2D_challenge/experiments/physical_refinement_20261031')
OLD = Path('/home/tanuj/V2D_challenge')
BASELINE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(pins):
    for path, expected in pins.items():
        assert sha(path) == expected, ('input/source changed', path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules(manifest):
    verify(manifest['source_sha256'])
    sys.path.insert(0, str(PRIOR / 'code'))
    # The same successful read-only Python3.10 dependency overlay as prior PEN.
    sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
    runner = load(PRIOR / 'code/joint_runner_v3.py', 'challenge_genuine_joint_prior_v3')
    runner.util.LOCAL = BASELINE
    runner.CHALLENGE_METADATA = manifest['metadata']
    original = inspect.getsource(runner.fit_case)
    old = "meta=json.loads((OLD/'work_labeled/LABELED_EPISODES.json').read_text())[str(ep)]"
    new = "meta=CHALLENGE_METADATA[str(ep)]"
    assert original.count(old) == 1
    transported = original.replace(old, new)
    ast.parse(transported)
    exec(compile(transported, '<challenge-native-metadata-transport>', 'exec'), runner.__dict__)
    receipt = dict(original_fit_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   transported_fit_function_sha256=hashlib.sha256(transported.encode()).hexdigest(),
                   replacements=[dict(old=old, new=new)],
                   baseline_directory=str(BASELINE), numerical_optimizer_changed=False,
                   image_depth_guard_changed=False)
    return runner, receipt


def fit_episode(ep, manifest):
    assert os.environ.get('PBS_JOBID'), 'Production requires PBS'
    case = manifest['cases'][ep]
    assert case['episode'] == ep
    verify(manifest['case_sha256'][str(ep)])
    runner, transport = modules(manifest)
    util = runner.util
    tests = json.loads((PRIOR / 'JOINT_CORE_TEST_V1.json').read_text())
    assert tests['status'] == 'PASS'
    verify(tests['source_sha256'])
    local_audit = json.loads((PRIOR.parent / 'validation_20261031/GENUINE_JOINT_FULL_V3_OUTPUT_REVIEW.json').read_text())
    assert local_audit['status'] == 'PASS' and local_audit['cases'] == 12 and local_audit['frames'] == 630
    if ep != 0:
        review_path = ROOT / 'validation/SMOKE_OUTPUT_REVIEW.json'
        review = json.loads(review_path.read_text())
        assert review['status'] == 'PASS_CHALLENGE_SMOKE'
        assert review['episode'] == 0
        assert review['producer_completion_sha256'] == sha(LARGE / 'episodes/episode_000000/COMPLETE.json')
        assert review['input_manifest_sha256'] == sha(ROOT / 'FIT_INPUTS_V1.json')
    import torch
    import numba
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(2)
    pe = util.module(util.PRIOR / 'code/evaluate_penetration.py', 'challenge_joint_pe')
    pen = util.module(pe.PEN, 'challenge_joint_official_pen')
    base = util.module(util.PRIOR / 'code/run_real_mesh_fit_v3.py', 'challenge_joint_decode')
    sys.path.insert(0, str(pe.KIT))
    from v2dlb.mesh_budget import budget_mesh
    hand = np.load(pe.HAND, allow_pickle=False)
    assert len(hand) == 5840
    faces = np.load(OLD / 'models/mhr/mhr_faces_lod1.npy', allow_pickle=False).astype(np.int64)
    model = torch.jit.load(str(OLD / 'models/mhr/mhr_model.pt'), map_location='cpu').eval()
    name = f'episode_{ep:06d}'
    data = dict(np.load(BASELINE / f'{name}.npz', allow_pickle=False))
    n = len(data['pose'])
    metadata = manifest['metadata'][str(ep)]
    assert metadata['frame_ids'] == list(range(n))
    assert np.allclose(metadata['times'], np.arange(n) / 30, atol=1e-10, rtol=0)
    assert case['total_frames'] == n
    output = LARGE / 'episodes' / name
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'TRANSPORT.json', transport)
    row = runner.fit_case(case, data, model, hand, faces, pen, base, budget_mesh, output)
    verify(manifest['source_sha256'])
    verify(manifest['case_sha256'][str(ep)])
    pins = {str(p): sha(p) for p in (output / 'submission').iterdir() if p.is_file()}
    auxiliary = {str(p): sha(p) for p in output.iterdir() if p.is_file()}
    dump(output / 'COMPLETE.json', dict(status='CHALLENGE_JOINT_EPISODE_COMPLETE', episode=ep,
         cases=1, frames=n, records=[row], submission=str(output / 'submission'),
         output_sha256=pins, auxiliary_output_sha256=auxiliary,
         source_sha256=manifest['source_sha256'], input_manifest_sha256=sha(ROOT / 'FIT_INPUTS_V1.json'),
         case_input_sha256=manifest['case_sha256'][str(ep)],
         native_frame_ids=list(range(n)), all_5840_hand_points_all_frames=True,
         human_bitexact=True, mesh_bitexact=True, rotation_scale_bitexact=True,
         GT_used_for_inference=False, GT_read=False, PBS_JOBID=os.environ['PBS_JOBID']))


def consolidate(manifest):
    assert os.environ.get('PBS_JOBID'), 'Production requires PBS'
    verify(manifest['source_sha256'])
    target = LARGE / 'submission'
    target.mkdir(parents=True, exist_ok=False)
    rows, outputs, auxiliary = [], {}, {}
    for ep in range(30):
        stem = f'episode_{ep:06d}'
        folder = LARGE / 'episodes' / stem
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        assert receipt['status'] == 'CHALLENGE_JOINT_EPISODE_COMPLETE' and receipt['episode'] == ep
        assert receipt['input_manifest_sha256'] == sha(ROOT / 'FIT_INPUTS_V1.json')
        verify(receipt['output_sha256'])
        verify(receipt['auxiliary_output_sha256'])
        assert receipt['frames'] == manifest['cases'][ep]['total_frames']
        before = dict(np.load(BASELINE / f'{stem}.npz', allow_pickle=False))
        after = dict(np.load(folder / 'submission' / f'{stem}.npz', allow_pickle=False))
        assert set(before) == set(after)
        for key in before:
            assert np.isfinite(after[key]).all()
            if key != 'object_translation':
                assert before[key].dtype == after[key].dtype and before[key].tobytes() == after[key].tobytes(), (ep, key)
        correction = after['object_translation'] - before['object_translation']
        assert correction.shape == (receipt['frames'], 3)
        assert float(np.linalg.norm(correction, axis=1).max()) <= .020000000001
        assert float(after['object_scale']) > 0
        rotation = after['object_rotation']
        assert np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0)
        assert np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
        for suffix in ['.npz', '_object.ply']:
            source = folder / 'submission' / (stem + suffix)
            path = target / source.name
            shutil.copyfile(source, path)
            outputs[str(path)] = sha(path)
        assert sha(target / f'{stem}_object.ply') == sha(BASELINE / f'{stem}_object.ply')
        rows.append(receipt['records'][0])
        auxiliary[str(folder / 'COMPLETE.json')] = sha(folder / 'COMPLETE.json')
    assert len(rows) == 30 and sum(r['frames'] for r in rows) == 16563
    assert len(outputs) == 60
    dump(LARGE / 'COMPLETE.json', dict(status='FULL30_GENUINE_JOINT_SUBMISSION_COMPLETE', cases=30,
         frames=16563, scored_frames=9877, records=rows, submission=str(target), output_sha256=outputs,
         auxiliary_output_sha256=auxiliary, source_sha256=manifest['source_sha256'],
         input_manifest=str(ROOT / 'FIT_INPUTS_V1.json'), input_manifest_sha256=sha(ROOT / 'FIT_INPUTS_V1.json'),
         human_bitexact=True, MAIN_human_bitexact=True, mesh_bitexact=True,
         rotation_scale_bitexact=True, GT_used_for_inference=False, GT_read=False,
         all_5840_hand_points_all_frames=True, CPU_job=os.environ['PBS_JOBID']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int)
    parser.add_argument('--consolidate', action='store_true')
    args = parser.parse_args()
    assert args.consolidate != (args.episode is not None)
    manifest = json.loads((ROOT / 'FIT_INPUTS_V1.json').read_text())
    if args.consolidate:
        consolidate(manifest)
    else:
        assert 0 <= args.episode < 30
        fit_episode(args.episode, manifest)
