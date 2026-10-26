#!/usr/bin/env python3
"""CPU-only independent official-decoder/SVD check of frozen human sidecars."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')


def main():
    for name, value in {'bool': bool, 'int': int, 'float': float,
                        'complex': complex, 'object': object, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    sys.path.insert(0, '/rhome/tanuj/projects/behave-dataset/libsmpl')
    import torch
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer
    torch.set_num_threads(2)
    sidecar = ROOT/'human_alignment_sidecars_v1/TRANSFORMS.json'
    report = json.loads(sidecar.read_text())
    mapping = np.load(ROOT.parents[1]/'third_party/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz')
    rows = []
    with tempfile.TemporaryDirectory(prefix='v2d_independent_human_') as temp:
        models = {}
        for record in report['records']:
            gender = record['gender']
            if gender not in models:
                Path(temp, f'SMPLH_{gender}.pkl').symlink_to(
                    f'/rhome/tanuj/CONTHO_base_data/human_models/smplh/SMPLH_{gender.upper()}.pkl')
                models[gender] = SMPL_Layer(center_idx=0, gender=gender,
                    num_betas=10, model_root=temp, hands=True).cpu().eval()
            model = models[gender]
            native = np.load(record['native_human_path'], allow_pickle=False)
            row = record['GT_row']
            with torch.inference_mode():
                vertices = model(torch.from_numpy(native['poses'][row:row+1]),
                    th_betas=torch.from_numpy(native['betas'][row:row+1]),
                    th_trans=torch.from_numpy(native['trans'][row:row+1]))[0][0].numpy().astype(np.float64)
            triangles = vertices[model.th_faces.numpy()[mapping['triangle_ids']]]
            target = (triangles*mapping['baryc_coords'][:, :, None]).sum(axis=1)
            predicted = np.load(record['predicted_body_path'])['vertices_cam'].astype(np.float64)
            # Separate SVD implementation; neither builder nor kit fitter imported.
            p = predicted-predicted.mean(0); g = target-target.mean(0)
            u, singular, vh = np.linalg.svd(p.T@g)
            correction = np.diag([1., 1., np.linalg.det(u@vh)])
            rotation = u@correction@vh
            scale = float(np.trace(np.diag(singular)@correction)/(p*p).sum())
            translation = target.mean(0)-scale*predicted.mean(0)@rotation
            frozen = record['primary_transform']
            errors = dict(scale=abs(scale-frozen['scale']),
                R_max=float(abs(rotation.T-np.array(frozen['R'])).max()),
                t_max_m=float(abs(translation-frozen['t']).max()))
            assert max(errors.values()) < 1e-6
            rows.append(dict(sequence=record['sequence'], native_first_frame=record['GT_frame'],
                errors_vs_frozen=errors, mean_human_fit_cm=float(100*np.linalg.norm(
                    scale*predicted@rotation+translation-target, axis=1).mean())))
    camera = Path('/rhome/tanuj/BEHAVE_raw/calibs/Date01/config/1/config.json')
    config = json.loads(camera.read_text())
    assert np.array_equal(np.array(config['rotation']).reshape(3, 3), np.eye(3))
    assert np.array_equal(config['translation'], np.zeros(3))
    output = dict(method='Independent official BEHAVE SMPL_Layer decode, direct barycentric sums, separate NumPy SVD',
        records=rows, camera1_identity_config_path=str(camera),
        camera1_identity_config_sha256=hashlib.sha256(camera.read_bytes()).hexdigest(),
        camera1_R_identity_t_zero=True, object_GT_used=False,
        sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
    path = ROOT.parents[1]/'validation/OPT_HUMAN_ALIGNMENT_INDEPENDENT_FIT_20261020.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
