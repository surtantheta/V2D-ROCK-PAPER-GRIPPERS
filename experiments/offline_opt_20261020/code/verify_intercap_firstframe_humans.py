#!/usr/bin/env python3
"""CPU GT-human-only decoder validation for frozen eight development takes."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    for name, value in {'bool': bool, 'int': int, 'float': float,
                        'complex': complex, 'object': object, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    import torch
    import smplx
    torch.set_num_threads(2)
    source = ROOT/'intercap_native_contract_v1/FIRST_FRAME_NATIVE_HUMANS.json'
    refs = json.loads(source.read_text())['records']
    assets = Path('/rhome/tanuj/CONTHO_base_data/human_models/smplh')
    neutral = smplx.SMPLH(str(assets), gender='neutral', use_pca=False, flat_hand_mean=False)
    regressor = neutral.J_regressor.numpy()
    # Exact CONTHO templates.py 52 joint rows +21 surface vertex rows.
    extra_vertices = [3216, 3226, 3387, 6617, 6624, 6787, 2746, 2319, 2445,
        2556, 2673, 6191, 5782, 5905, 6016, 6133, 332, 2800, 6260, 583, 4071]
    outputs = []; vertex_arrays = {}; pins = {str(source): sha(source), str(Path(__file__)): sha(Path(__file__))}
    for gender in sorted({r['smpl_param']['gender'] for r in refs}):
        selected = [r for r in refs if r['smpl_param']['gender'] == gender]
        pose = torch.tensor([r['smpl_param']['pose'] for r in selected], dtype=torch.float32)
        betas = torch.tensor([r['smpl_param']['shape'] for r in selected], dtype=torch.float32)
        trans = torch.tensor([r['smpl_param']['trans'] for r in selected], dtype=torch.float32)
        for flat in [False, True]:
            model = smplx.SMPLH(str(assets), gender=gender, use_pca=False,
                flat_hand_mean=flat, batch_size=len(selected)).cpu().eval()
            assert np.array_equal(model.faces, neutral.faces)
            with torch.inference_mode():
                vertices = model(global_orient=pose[:, :3], body_pose=pose[:, 3:66],
                    left_hand_pose=pose[:, 66:111], right_hand_pose=pose[:, 111:],
                    betas=betas, transl=trans).vertices.numpy()
            joints = np.concatenate([regressor@vertices, vertices[:, extra_vertices]], axis=1)
            for i, row in enumerate(selected):
                error = np.linalg.norm(joints[i]-np.asarray(row['h3d_keypoints']), axis=1)
                outputs.append(dict(take=row['take'], gender=gender, flat_hand_mean=flat,
                    mean_joint_error_m=float(error.mean()), max_joint_error_m=float(error.max())))
                if not flat:
                    assert error.max() < 1e-5, 'native human decoder/translation convention mismatch'
                    vertex_arrays[row['take'].replace('/', '_')] = vertices[i]
        path = assets/f'SMPLH_{gender.upper()}.pkl'; pins[str(path)] = sha(path)
    path = assets/'SMPLH_NEUTRAL.pkl'; pins[str(path)] = sha(path)
    assert len(vertex_arrays) == 8
    verts = ROOT/'intercap_native_contract_v1/FIRST_FRAME_GT_SMPLH_VERTICES.npz'
    assert not verts.exists()
    np.savez_compressed(verts, **vertex_arrays)
    pins[str(verts)] = sha(verts)
    output = dict(records=outputs, primary_flat_hand_mean=False,
        model='SMPLH6890, gender-specific; use_pca=False; common neutral CONTHO joint regressor',
        coordinate_contract='vertices_cam = SMPLH(pose156,betas10,gender,nonflat_hand_mean).vertices + native_trans_metres',
        GT_human_only=True, native_object_fields_used=False,
        predicted_human_required_for_alignment=True, first_frame_source_join_verified=True,
        MHR_mapping='official SMPL6890-to-MHR18439 barycentric map; surrogate alignment, not native hosted MHR GT',
        source_sha256=pins)
    path = ROOT/'intercap_native_contract_v1/HUMAN_DECODER_CHECKS.json'
    path.write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
