"""Exercise the actual exceptional branch under rigid rotation and scale."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / 'physical_refinement_20261031'
OLD = Path('/home/tanuj/V2D_challenge')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert os.environ.get('PBS_JOBID')
    import numba
    numba.set_num_threads(2)
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(PRIOR / 'code'))
    sys.path.insert(0, str(ROOT / 'code'))
    sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
    sys.path.insert(0, str(OLD / 'data/assets/submission_kit/v2d_submission_kit'))
    import joint_core_v1 as core
    from v2dlb.mesh_budget import budget_mesh
    repair = load(ROOT / 'code/official_closest_repair_v2.py', 'actual_closest_branch_repair_v2')
    pen_path = OLD / 'data/assets/submission_kit/v2d_submission_kit/metric_code/track_1/PEN.py'
    pen = load(pen_path, 'actual_closest_branch_official_pen')
    point_tests = json.loads((ROOT / 'validation/OFFICIAL_CLOSEST_REPAIR_TEST_V2.json').read_text())
    assert point_tests['original_fastpath_bitexact']
    failed = json.loads((ROOT / 'validation/KERNEL_POINT_PROBE_V1.json').read_text())['records'][0]
    q = np.asarray(failed['local_point'])
    v, f = budget_mesh(Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission/episode_000012_object.ply'), 4096, 4096)
    rotation = Rotation.from_rotvec([.3, -.2, .1]).as_matrix()[None]
    scale = 1.7
    translation = np.array([[.6, -.4, 3.2]])
    hands = (scale * (q @ rotation[0].T) + translation[0])[None, None]
    old = core.SignedMesh(v, f, rotation, scale, hands, pen)
    signed = old.signed(translation, .022)
    normalizers, _ = core.frozen_normalizers(signed, core.Config())
    failed_before = False
    try:
        old.linearization(translation, np.zeros_like(translation), signed, normalizers, core.Config())
    except AssertionError as error:
        assert error.args[0][0] == 'distance-gradient mismatch'
        failed_before = True
    assert failed_before
    proof, events = repair.install(core, pen)
    new = core.SignedMesh(v, f, rotation, scale, hands, pen)
    a, b, w, joined = new.linearization(translation, np.zeros_like(translation), signed, normalizers, core.Config())
    assert len(events) == 1 and a.shape == (1, 3)
    derivative = []
    step = 1e-7
    for axis in range(3):
        plus = translation.copy(); plus[0, axis] += step
        minus = translation.copy(); minus[0, axis] -= step
        derivative.append(float(-(new.signed(plus, .022)[0, 0] - new.signed(minus, .022)[0, 0]) / (2 * step)))
    gradient_error = float(np.max(np.abs(np.asarray(derivative) - a.toarray()[0])))
    assert gradient_error < 1e-5
    paths = [Path(__file__), ROOT / 'code/official_closest_repair_v2.py', PRIOR / 'code/joint_core_v1.py', pen_path,
             ROOT / 'validation/KERNEL_POINT_PROBE_V1.json', ROOT / 'validation/OFFICIAL_CLOSEST_REPAIR_TEST_V2.json']
    receipt = dict(status='PASS_REAL_PAN_RARE_BRANCH_ROTATED_SCALED_GRADIENT',
                   old_branch_reproduced_failure=True, exceptional_branch_events=events,
                   actual_rotated_scaled_pose_gradient_max_error=gradient_error,
                   original_fastpath_bitexact=point_tests['original_fastpath_bitexact'],
                   all14_real_oracle_queries_exact=True, original_tolerance_unchanged=True,
                   proof=proof, sha256={str(p): sha(p) for p in paths}, GT_loaded=False,
                   CPU_job=os.environ['PBS_JOBID'])
    out = ROOT / 'validation/OFFICIAL_CLOSEST_BRANCH_TEST_V2.json'
    assert not out.exists()
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['proof', 'sha256']}), flush=True)


if __name__ == '__main__':
    main()
