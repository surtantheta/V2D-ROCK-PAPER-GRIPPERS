"""Actual pan-point oracle replay plus fast-path and geometric regressions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/tanuj/V2D_challenge')
PRIOR = ROOT.parent / 'physical_refinement_20261031'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert os.environ.get('PBS_JOBID')
    import torch
    import numba
    torch.set_num_threads(2)
    numba.set_num_threads(2)
    sys.path.insert(0, str(ROOT / 'code'))
    sys.path.insert(0, str(PRIOR / 'code'))
    sys.path.append('/home/tanuj/.conda/envs/grail/lib/python3.10/site-packages')
    import joint_core_v1 as core
    repair = load(ROOT / 'code/official_closest_repair_v2.py', 'tested_official_point_repair_v2')
    pen = load(OLD / 'data/assets/submission_kit/v2d_submission_kit/metric_code/track_1/PEN.py', 'tested_repair_official_pen')
    sys.path.insert(0, str(OLD / 'data/assets/submission_kit/v2d_submission_kit'))
    from v2dlb.mesh_budget import budget_mesh
    kernel, proof = repair.official_point_kernel(pen)
    probe = json.loads((ROOT / 'validation/KERNEL_POINT_PROBE_V1.json').read_text())
    records = []
    for row in probe['records']:
        ep = row['episode']
        v, f = budget_mesh(Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029/cpu_v2/fit/submission') / f'episode_{ep:06d}_object.ply', 4096, 4096)
        tri = np.asarray(v, float)[np.asarray(f, np.int64)]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        tri = np.ascontiguousarray(tri[area > 0])
        v0, v1, v2 = [np.ascontiguousarray(tri[:, k]) for k in range(3)]
        q = np.asarray(row['local_point'], float)
        closest, squared, face = kernel(q, v0, v1, v2)
        oracle = float(pen._PENETRATION_KERNELS[1](q[None], v0, v1, v2)[0])
        norm = float(np.linalg.norm(q - closest))
        assert abs(norm - oracle) < 1e-12
        finite_differences = []
        gradient = (q - closest) / norm
        for step in [1e-8, 1e-7, 1e-6]:
            points = np.concatenate([q[None] + step * np.eye(3), q[None] - step * np.eye(3)])
            values = pen._PENETRATION_KERNELS[1](points, v0, v1, v2)
            finite = (values[:3] - values[3:]) / (2 * step)
            finite_differences.append(dict(step=step, gradient=finite.tolist(), max_error=float(np.max(np.abs(finite - gradient)))))
        records.append(dict(episode=ep, frame=row['frame'], delta=row['translation_delta_m'],
                            oracle_distance_model_units=oracle, repaired_distance_model_units=norm,
                            original_distance_discrepancy_scene_m=row['distance_mismatch_m'],
                            selected_official_triangle=int(face), selected_triangle_area=float(area[area > 0][face] / 2),
                            selected_triangle=tri[face].tolist(), closest_point=closest.tolist(),
                            nearest_point_gradient=gradient.tolist(), finite_difference=finite_differences))
    # Nonproblematic box keeps original path and exact Jacobian output.
    import trimesh
    box = trimesh.creation.box(extents=[2, 2, 2])
    hands = np.array([[[.9, 0, 0], [1.0005, 0, 0]]])
    t0 = np.zeros((1, 3)); rotations = np.eye(3)[None]
    old_class = core.SignedMesh
    old = old_class(box.vertices, box.faces, rotations, 1., hands, pen)
    signed = old.signed(t0, .022); normalizers, _ = core.frozen_normalizers(signed, core.Config())
    old_values = old.linearization(t0, t0, signed, normalizers, core.Config())
    install_proof, events = repair.install(core, pen)
    new = core.SignedMesh(box.vertices, box.faces, rotations, 1., hands, pen)
    new_values = new.linearization(t0, t0, signed, normalizers, core.Config())
    assert len(events) == 0
    for a, b in zip(old_values[:3], new_values[:3]):
        if hasattr(a, 'toarray'):
            assert np.array_equal(a.toarray(), b.toarray())
        else:
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert old_values[3] == new_values[3]
    result = dict(status='ORACLE_REPLAY_AND_FASTPATH_PASS_GRADIENT_EVIDENCE_RECORDED', records=records,
                  original_fastpath_bitexact=True, original_threshold_unchanged=True,
                  point_kernel=proof, installation=install_proof, GT_loaded=False, PBS_JOBID=os.environ['PBS_JOBID'])
    out = ROOT / 'validation/OFFICIAL_CLOSEST_REPAIR_TEST_V2.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], cases=len(records),
                          max_replay_error=max(abs(x['oracle_distance_model_units'] - x['repaired_distance_model_units']) for x in records),
                          first_point=records[0])), flush=True)


if __name__ == '__main__':
    main()
