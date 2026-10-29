#!/usr/bin/env python3
"""Challenge-only transport of the frozen old-mask Pi3X calibration/mesh control.

Numerical helpers, body visibility, fitting, solver and expansion come directly
from the already exercised sources. Function copies only replace cohort paths,
metadata and completeness checks; the explicit missing-calibration guard admits
zero metric observations, preserving the existing mask-only solver behavior.
V2 additionally projects stored object rotation roundoff to SO(3), under a
strict 1e-4 input-error gate, before the unchanged solver; all corrections are
recorded. Single-frame calibrated Sim3 rotations remain unchanged.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

R = Path('/home/tanuj/V2D_challenge')
K = R/'experiments/pi3x_kaggle_20261029'
PREV = R/'experiments/vggt_fusion_20261028'
CONTROL = R/'experiments/identity_fusion_20261029/code'
LARGE = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029')
OUT = LARGE/'cpu_v2'
METADATA = K/'CHALLENGE_EPISODES.json'
EXPECTED_CONTROL = {
    str(CONTROL/'run_real_mesh_fit_v3.py'): 'b5fe1d0758c6e7939158918a9a64b59e6eba3196c7762d5b528a180c394f3f50',
    str(CONTROL/'real_mesh_optimizer.py'): '116e9cab6335102a323de01b8b17868ff463d7b7f5eabcb8c83db20bb34a6f3b',
}
CALIBRATION_TRANSPORT = [
    ("metadata=json.loads((W/'LABELED_EPISODES.json').read_text())", "metadata=json.loads(CHALLENGE_METADATA.read_text())"),
    ("        if case['dataset']=='challenge':continue\n", ''),
    ('controls=control_diagnostics(j,pins)', 'controls=[]  # no duplicate/reverse diagnostic runs in production'),
    ('assert len(records)==36', 'assert len(records)==30'),
]
OPTIMIZER_TRANSPORT = [
    ("metadata=json.loads((MAIN/'LABELED_EPISODES.json').read_text())", "metadata=json.loads(CHALLENGE_METADATA.read_text())"),
    ("case['calibration']['frames'][p]['calibrated_body_residual_cm']", "case['calibration']['frames'][p].get('calibrated_body_residual_cm')"),
    ('if len(records)==12:assert total==630', "assert len(records)==30 and total==sum(c['total_frames'] for c in j['cases'])"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    for p, expected in EXPECTED_CONTROL.items():
        assert sha(p) == expected, ('Original control changed', p)
    sys.path.insert(0, str(CONTROL))
    sys.path.insert(0, str(PREV/'admin'))
    diag = load('challenge30_geometry_calibration_v1', PREV/'admin/evaluate_geometry_pilot.py')
    base = load('challenge30_old_control_v1', CONTROL/'run_real_mesh_fit_v3.py')
    surface = load('challenge30_frontmost_surface_v3', PREV/'admin/body_surface_raster_v3.py')
    return diag, base, surface


def transported(module, name, replacements):
    original = inspect.getsource(getattr(module, name))
    changed = original
    for old, new in replacements:
        assert changed.count(old) == 1, ('transport source changed', name, old)
        changed = changed.replace(old, new)
    ast.parse(changed)
    exec(compile(changed, f'<challenge30-transport:{name}>', 'exec'), module.__dict__)
    return dict(name=name, original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                transported_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                replacements=[dict(old=a, new=b) for a, b in replacements])


def freeze():
    target = K/'CPU_INPUTS_V2.json'
    assert not target.exists(), 'Use a new version for any changed input'
    manifest = json.loads((K/'PILOT_INPUTS.json').read_text())
    metadata = json.loads(METADATA.read_text())
    assert [c['episode'] for c in manifest['cases']] == list(range(30))
    pins = dict(manifest['input_sha256'])
    paths = [Path(__file__), K/'code/rotation_precision_v2.py', K/'PILOT_INPUTS.json', K/'SOURCE_FREEZE.json', METADATA,
             PREV/'admin/evaluate_geometry_pilot.py', PREV/'admin/body_surface_raster_v3.py',
             CONTROL/'run_real_mesh_fit_v3.py', CONTROL/'real_mesh_optimizer.py',
             R/'models/mhr/mhr_model.pt', R/'models/mhr/mhr_faces_lod1.npy']
    for case in manifest['cases']:
        ep = case['episode']; name = f'episode_{ep:06d}'
        paths.extend([Path(case['fixed_MAIN_human_npz']), R/f'work/submission_v2/{name}_object.ply',
                      R/f'work/s6/{name}/poses.npz', R/f'work/s6/{name}/TRACK.json',
                      R/f'work/s3db/{name}/sam3d_body.npz'])
        m = metadata[str(ep)]
        assert case['total_frames'] == len(m['times']) == len(m['frame_ids'])
        assert len(case['frames']) == 16
        for fr in case['frames']:
            i = fr['local_index']
            assert fr['native_frame_id'] == m['frame_ids'][i]
            assert abs(fr['time_seconds']-m['times'][i]) < 1e-8
    pins.update({str(p): sha(p) for p in paths})
    dump(target, dict(schema='challenge30-old-mask-pi3x-cpu-input-v2', sha256=pins,
                     cases=list(range(30)), provider='pi3x', observations_per_case=16,
                     object_solver_sample_positions=[0, 2, 4, 6, 9, 11, 13, 15],
                     canonical_mesh='literal_MAIN', human='literal_MAIN',
                     masks='episode-specific original MAIN choices',
                     new_GT_read=False, numerical_policy='original old_pi3x_v3 control plus bounded stored-rotation precision repair',
                     rotation_precision_repair='nearestSO3, orthogonality and determinant errors <=1e-4, positive determinant; reject other inputs',
                     unsupported_calibration='empty metric cloud, no invented calibration; existing silhouette-only solve keeps scale fixed'))
    print('CPU_INPUTS_FROZEN', target, len(pins), flush=True)


def run():
    assert os.environ.get('PBS_JOBID'), 'Calibration and optimization require PBS'
    frozen = json.loads((K/'CPU_INPUTS_V2.json').read_text())
    diag, base, surface = modules()
    diag.verify(frozen['sha256'])
    manifest = json.loads((K/'PILOT_INPUTS.json').read_text())
    assert manifest['models'][0]['name'] == 'pi3x' and len(manifest['models']) == 1
    assert [c['episode'] for c in manifest['cases']] == list(range(30))
    assert not OUT.exists(), 'Preserve partial and completed runs; use a new version'
    OUT.mkdir(parents=True)
    diag.W = R/'work'
    diag.OUT = OUT/'calibration'
    diag.OUT.mkdir()
    diag.CHALLENGE_METADATA = METADATA
    diag.PROTOCOL = dict(copy.deepcopy(diag.PROTOCOL), cases=list(range(30)), models=['pi3x'],
        body_visibility='frontmost complete predicted-MAIN triangle depth; perspective-correct Z; torso face majority; same existing reliability thresholds',
        calibration_version=3, native_GT_geometry_read=False,
        score='No native GT scoring in challenge production')
    faces = np.load(R/'models/mhr/mhr_faces_lod1.npy').astype(np.int64)
    def pairs(vertices, joints, local, confidence, A, Krgb, objmask):
        return surface.visible_pairs(vertices, joints, local, confidence, A, Krgb, objmask,
            faces, diag.PROTOCOL['torso_joint_positions'], diag.PROTOCOL['body_max_pixels_per_frame'])
    diag.body_pairs = pairs
    transport = [transported(diag, 'predicted_stage', CALIBRATION_TRANSPORT)]
    calibration_freeze = dict(frozen, protocol=diag.PROTOCOL,
        pilot_source_freeze_sha256=sha(K/'SOURCE_FREEZE.json'))
    dump(OUT/'CALIBRATION_SOURCE_BEFORE_FIT.json', dict(frozen=calibration_freeze,
        function_transports=transport, visibility_preflight=surface.preflight(), GT_read=False))
    cp = diag.predicted_stage(calibration_freeze, manifest)
    cal = json.loads(cp.read_text())
    assert len(cal['records']) == 30 and not cal['native_GT_values_read']
    fit = OUT/'fit'
    fit.mkdir()
    source = dict(cal['source_sha256'])
    source[str(cp)] = sha(cp)
    cases = []
    for case, record in zip(manifest['cases'], cal['records']):
        ep=case['episode']; name=f'episode_{ep:06d}'
        assert record['episode'] == ep and record['model'] == 'pi3x'
        modelroot=Path(manifest['large_output_root'])/'pi3x'/f'{name}_normal'
        paths=dict(baseline=case['fixed_MAIN_human_npz'], mesh=str(R/f'work/submission_v2/{name}_object.ply'),
                   poses=str(R/f'work/s6/{name}/poses.npz'), geometry=str(modelroot/'GEOMETRY.npz'))
        source.update({p:sha(p) for p in paths.values()})
        samples=[]
        for p in base.take(16,8):
            fr=case['frames'][int(p)]
            samples.append(dict(position=int(p), frame=fr, mask=fr['mask'], selected=fr['mask_selected'], state='visible'))
        cases.append(dict(episode=ep,dataset=case['dataset'],sequence=case['sequence'],
                          total_frames=case['total_frames'],samples=samples,paths=paths,calibration=record))
    config = dict(base.Config().__dict__)
    inp = fit/'INPUTS.json'
    base.dump(inp, dict(protocol=base.PROTOCOL, model='pi3x', mask_treatment='old_mask_control',
        cases=cases, config=config, source_sha256=source, output=str(fit), frames_requested=8,
        masks_manifest=None, created_before_new_fullmesh_GT_scoring=True))
    base.MAIN=R/'work'
    base.CHALLENGE_METADATA=METADATA
    original_sample_cloud=base.sample_cloud
    def available_cloud(local,A,mask_path,rgb_wh,calibration,budget=256):
        if calibration is None:
            return np.zeros((0,3)),dict(mask_pixels=None,eroded_pixels=None,erosion_fallback=None,
                finite_available_points=0,source_mask_wh=list(rgb_wh),metric_unavailable_reason='unsupported_predicted_human_calibration')
        return original_sample_cloud(local,A,mask_path,rgb_wh,calibration,budget)
    base.sample_cloud=available_cloud
    transport.append(transported(base,'run',OPTIMIZER_TRANSPORT))
    dump(OUT/'TRANSPORTS_BEFORE_OPTIMIZATION.json', dict(function_transports=transport,
        missing_calibration_guard='No metric points; keep old solver scale-lock and image evidence behavior',
        GT_read=False, input_manifest=str(inp), input_manifest_sha256=sha(inp)))
    precision = load('challenge30_rotation_precision_v2', K/'code/rotation_precision_v2.py')
    original_proper = base.proper
    precision_records = []
    rotation_episode = {}
    for case in cases:
        with np.load(case['paths']['baseline']) as baseline:
            key = hashlib.sha256(np.ascontiguousarray(baseline['object_rotation'], dtype=np.float64).tobytes()).hexdigest()
            rotation_episode[key] = case['episode']
    def precision_proper(value):
        array = np.asarray(value, dtype=np.float64)
        # The calibrated single-frame Sim3 rotation remains literally unchanged.
        # Full stored object trajectories receive the explicit bounded projection.
        if array.ndim == 2:
            return original_proper(array)
        repaired, diagnostic = precision.project(array)
        key = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        assert key in rotation_episode, 'Unexpected matrix batch at precision admission'
        diagnostic['episode'] = rotation_episode[key]
        diagnostic['input_rotation_sha256'] = key
        precision_records.append(diagnostic)
        dump(OUT/'ROTATION_PRECISION_REPAIRS.json', dict(records=precision_records,
            tolerance=precision.TOLERANCE, human_mesh_scale_translation_unchanged_by_projection=True,
            GT_used=False))
        return original_proper(repaired)
    base.proper = precision_proper
    base.run(argparse.Namespace(inputs=str(inp)))
    assert sorted(r['episode'] for r in precision_records) == list(range(30))
    result=json.loads((fit/'COMPLETE.json').read_text())
    assert result['cases']==30 and len([p for p in (fit/'submission').iterdir()])==60
    diag.verify(frozen['sha256'])
    base.dump(OUT/'COMPLETE.json',dict(status='CHALLENGE30_OLD_MASK_PI3X_COMPLETE',
        cases=30,frames=result['frames'],submission=result['submission'],
        source_sha256=frozen['sha256'],output_sha256=result['output_sha256'],
        producer_completion=str(fit/'COMPLETE.json'),producer_completion_sha256=sha(fit/'COMPLETE.json'),
        calibration=str(cp),calibration_sha256=sha(cp),
        rotation_precision_receipt=str(OUT/'ROTATION_PRECISION_REPAIRS.json'),
        rotation_precision_receipt_sha256=sha(OUT/'ROTATION_PRECISION_REPAIRS.json'),human_bitexact=True,mesh_byteexact=True,
        GT_read=False,GT_used_for_inference=False,CPU_job=os.environ['PBS_JOBID']))
    print('CHALLENGE30_CPU_COMPLETE',OUT/'COMPLETE.json',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true')
    ap.add_argument('--import-only',action='store_true')
    args=ap.parse_args()
    assert sum([args.freeze,args.run,args.import_only])==1
    if args.import_only:
        modules()
        load('challenge30_rotation_precision_v2', K/'code/rotation_precision_v2.py')
        import torch
        assert not torch.cuda.is_initialized()
        print('CPU_ADAPTER_IMPORT_PASS_CUDA_UNINITIALIZED',flush=True)
    elif args.freeze:freeze()
    else:run()


if __name__=='__main__':main()
