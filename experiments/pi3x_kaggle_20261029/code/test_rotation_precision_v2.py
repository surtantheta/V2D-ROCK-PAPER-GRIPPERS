"""Actual 30-clip floating-point admission and geometric displacement audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

K=Path('/home/tanuj/V2D_challenge/experiments/pi3x_kaggle_20261029')
R=Path('/home/tanuj/V2D_challenge')
spec=importlib.util.spec_from_file_location('precision_under_test',K/'code/rotation_precision_v2.py')
precision=importlib.util.module_from_spec(spec);sys.modules[spec.name]=precision;spec.loader.exec_module(precision)
spec=importlib.util.spec_from_file_location('original_control_math',R/'experiments/identity_fusion_20261029/code/real_mesh_optimizer.py')
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)


def main():
    rows=[]
    for ep in range(30):
        stem=f'episode_{ep:06d}'
        data=np.load(R/f'work/submission_v2/{stem}.npz')
        original=data['object_rotation'];before=hashlib.sha256(original.tobytes()).hexdigest()
        repaired,diagnostic=precision.project(original)
        core.proper(repaired)
        assert hashlib.sha256(original.tobytes()).hexdigest()==before
        vertices=np.asarray(trimesh.load(R/f'work/submission_v2/{stem}_object.ply',process=False).vertices,float)
        scale=float(data['object_scale'])
        delta=repaired-original
        # This bound covers EVERY frame/vertex, including unmeasured vertices.
        bound=scale*float(np.linalg.norm(delta,axis=(1,2)).max())*float(np.linalg.norm(vertices,axis=1).max())
        keep=np.unique(np.rint(np.linspace(0,len(vertices)-1,min(128,len(vertices)))).astype(int))
        measured=scale*float(np.linalg.norm(np.einsum('fij,vj->fvi',delta,vertices[keep]),axis=-1).max())
        assert measured<=bound+1e-12 and bound<1e-4
        rows.append(dict(episode=ep,frames=len(original),all_full_frames_proper_after=True,
            input_bytes_preserved=True,max_Frobenius_correction=diagnostic['max_Frobenius_correction'],
            max_orth_before=max(diagnostic['orthogonality_error_before']),
            max_orth_after=max(diagnostic['orthogonality_error_after']),
            deterministic_vertices_tested=len(keep),max_measured_vertex_displacement_m=measured,
            rigorous_all_frame_all_vertex_displacement_bound_m=bound))
    for name,bad in [('reflection',np.diag([1.,1.,-1.])),('gross_scale',np.eye(3)*1.01),
                     ('nonfinite',np.eye(3)*np.nan),('wrong_shape',np.eye(4))]:
        try:precision.project(bad)
        except ValueError:pass
        else:raise AssertionError(('Invalid rotation admitted',name))
    _,exact=precision.project(np.eye(3))
    assert exact['max_Frobenius_correction']==0
    result=dict(status='ACTUAL30_ROTATION_PRECISION_AND_GEOMETRY_BOUND_PASS',rows=rows,
        cases=30,frames=sum(r['frames'] for r in rows),tolerance=precision.TOLERANCE,
        max_all_frame_all_vertex_bound_m=max(r['rigorous_all_frame_all_vertex_displacement_bound_m'] for r in rows),
        reflection_gross_nonfinite_shape_rejected=True,new_model_or_calibration_fit=False,
        input_MAIN_changed=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
