#!/usr/bin/env python3
"""Scorer-only first-frame HUMAN Sim3; native object labels/CAD never opened.

GT SMPLH -> official SMPL-to-MHR barycentric map -> exact kit human Sim3.
This is a BEHAVE correspondence surrogate, not native challenge MHR GT.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PROJECT = ROOT.parents[1]
KIT = PROJECT/'data/assets/submission_kit/v2d_submission_kit'
MAPS = PROJECT/'third_party/MHR/tools/mhr_smpl_conversion/assets'
BEHAVE_CODE = Path('/rhome/tanuj/projects/behave-dataset')
MODELS = Path('/rhome/tanuj/CONTHO_base_data/human_models/smplh')
SMPL_MODELS = Path('/home/tanuj/UniCon3Rpp-push-b3c6624/src/models/smpl')
NATIVE = Path('/rhome/tanuj/BEHAVE_raw')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mapping(path, face_count, target_vertices):
    data = np.load(path, allow_pickle=False)
    ids = data['triangle_ids']; weights = data['baryc_coords']
    assert ids.shape == (target_vertices,) and weights.shape == (target_vertices, 3)
    assert ids.min() >= 0 and ids.max() < face_count
    assert np.isfinite(weights).all() and weights.min() > -1e-10
    assert np.max(abs(weights.sum(1)-1)) < 1e-12
    return ids, weights


def transfer(vertices, faces, asset):
    ids, weights = asset
    return np.einsum('vkd,vk->vd', vertices[faces[ids]], weights)


def output_transform(transform):
    scale, row_rotation, translation = transform
    return {'scale': float(scale), 'R': np.asarray(row_rotation).T.tolist(),
            't': np.asarray(translation).tolist(),
            'convention': 'X_GT = scale * (X_pred @ R.T) + t'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'human_alignment_sidecars_v1')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('output must be new')
    # Compatibility is confined to this CPU process; no installed package is edited.
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    sys.path[:0] = [str(KIT), str(BEHAVE_CODE/'libsmpl')]
    import smplx
    import torch
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer
    from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform

    torch.set_num_threads(2)
    requests_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    requests = json.loads(requests_path.read_text())['BEHAVE']
    assert len(requests) == 4 and len({x['sequence'] for x in requests}) == 4
    faces_path = PROJECT/'models/mhr/mhr_faces_lod1.npy'
    mhr_faces = np.load(faces_path, allow_pickle=False)
    assert mhr_faces.shape == (36874, 3) and mhr_faces.max() == 18438
    reverse_path = MAPS/'mhr2smpl_mapping.npz'
    reverse = mapping(reverse_path, len(mhr_faces), 6890)
    cases = []; sources = {}
    def pin(path):
        sources[str(Path(path).resolve())] = sha(path)
    for path in [Path(__file__), requests_path, faces_path, reverse_path,
                 MAPS/'smpl2mhr_mapping.npz', KIT/'v2dlb/mhr_metrics.py',
                 BEHAVE_CODE/'tools/smpl_params2mesh.py',
                 BEHAVE_CODE/'libsmpl/smplpytorch/pytorch/smpl_layer.py',
                 BEHAVE_CODE/'libsmpl/smplpytorch/native/webuser/serialization.py',
                 BEHAVE_CODE/'data/kinect_transform.py']:
        pin(path)
    for item in requests:
        assert item['camera'] == 1
        seq = item['sequence']; first_frame = min(item['video_frame_indices'])
        assert item['video_frame_indices'][0] == first_frame
        native_path = NATIVE/seq/'smpl_fit_all.npz'
        info_path = NATIVE/seq/'info.json'
        native = np.load(native_path, allow_pickle=False)
        # Deliberately only body-specific NPZ; no object_fit_all or pose_abs read.
        gender = str(native['gender'].item())
        assert gender in ['male', 'female']
        assert json.loads(info_path.read_text())['gender'] == gender
        frame_times = native['frame_times']
        times = np.asarray([float(str(x).split('/')[-1].lstrip('t')) for x in frame_times])
        timestamp = float(item['timestamps_seconds'][0])
        row = int(np.argmin(abs(times-timestamp)))
        join_error = float(abs(times[row]-timestamp))
        assert join_error < .005, 'first scored RGB/body GT timestamp join failed'
        human_path = ROOT/'labeled_predictions_v1/human_scale'/seq/'body_000000.npz'
        human = np.load(human_path, allow_pickle=False)
        predicted = np.asarray(human['vertices_cam'], np.float64)
        assert predicted.shape == (18439, 3) and np.isfinite(predicted).all()
        assert (predicted[:, 2] > 0).all()
        cases.append({'sequence': seq, 'gender': gender, 'predicted': predicted,
                      'pose': native['poses'][row], 'betas': native['betas'][row],
                      'translation': native['trans'][row], 'GT_row': row,
                      'GT_frame': str(frame_times[row]), 'GT_time_seconds': float(times[row]),
                      'RGB_frame': first_frame, 'RGB_time_seconds': timestamp,
                      'timestamp_join_error_seconds': join_error,
                      'predicted_body_path': str(human_path),
                      'predicted_body_sha256': sha(human_path),
                      'native_human_path': str(native_path)})
        for path in [native_path, info_path, human_path]: pin(path)
    records = []; parity = []
    with tempfile.TemporaryDirectory(prefix='v2d_human_decoder_') as temp:
        for gender in sorted({x['gender'] for x in cases}):
            selected = [x for x in cases if x['gender'] == gender]
            model_path = MODELS/f'SMPLH_{gender.upper()}.pkl'
            Path(temp, f'SMPLH_{gender}.pkl').symlink_to(model_path)
            official = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                                  model_root=temp, hands=True).cpu().eval()
            model = smplx.SMPLH(str(MODELS), gender=gender, ext='pkl', use_pca=False,
                                flat_hand_mean=True, num_betas=10,
                                batch_size=len(selected)).cpu().eval()
            smpl_path = SMPL_MODELS/f'SMPL_{gender.upper()}.pkl'
            topology = smplx.SMPL(str(smpl_path), gender=gender, num_betas=10).cpu().eval()
            smpl_faces = np.asarray(model.faces, np.int64)
            assert np.array_equal(smpl_faces, official.th_faces.numpy())
            assert np.array_equal(smpl_faces, topology.faces), 'SMPL/SMPLH vertex-ID topology parity failed'
            forward = mapping(MAPS/'smpl2mhr_mapping.npz', len(smpl_faces), 18439)
            poses = torch.from_numpy(np.stack([x['pose'] for x in selected]))
            betas = torch.from_numpy(np.stack([x['betas'] for x in selected]))
            translations = torch.from_numpy(np.stack([x['translation'] for x in selected]))
            with torch.inference_mode():
                reference = official(poses, th_betas=betas, th_trans=translations)[0].numpy()
                actual = model(global_orient=poses[:, :3], body_pose=poses[:, 3:66],
                               left_hand_pose=poses[:, 66:111], right_hand_pose=poses[:, 111:156],
                               betas=betas, transl=translations).vertices.numpy()
            maximum = float(np.linalg.norm(reference-actual, axis=-1).max())
            assert maximum < 1e-5, 'native SMPLH decoder parity failed'
            parity.append({'gender': gender, 'frames': len(selected),
                           'official_decoder_max_vertex_error_m': maximum,
                           'flat_hand_mean': True, 'use_pca': False,
                           'SMPL_SMPLH_faces_exact': True})
            pin(model_path); pin(smpl_path)
            for index, case in enumerate(selected):
                predicted = case['predicted']; native_vertices = actual[index].astype(np.float64)
                target_mhr = transfer(native_vertices, smpl_faces, forward)
                mapped_prediction = transfer(predicted, mhr_faces, reverse)
                transform = fit_similarity_transform(predicted, target_mhr)
                alternative = fit_similarity_transform(mapped_prediction, native_vertices)
                residual = np.linalg.norm(apply_similarity_transform(predicted, transform)-target_mhr, axis=1)
                # Affine-equivariance test uses the actual per-vertex official mapping.
                angle = .37; c, s = np.cos(angle), np.sin(angle)
                q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
                offset = np.array([.13, -.07, .2]); factor = 1.73
                mapped_scaled = transfer(factor*predicted@q+offset, mhr_faces, reverse)
                error = float(abs(mapped_scaled-(factor*mapped_prediction@q+offset)).max())
                mapped_target_scaled = transfer(factor*native_vertices@q+offset, smpl_faces, forward)
                forward_error = float(abs(mapped_target_scaled-(factor*target_mhr@q+offset)).max())
                assert max(error, forward_error) < 1e-11
                known = fit_similarity_transform(predicted, factor*predicted@q+offset)
                known_error = float(abs(apply_similarity_transform(predicted, known)-(factor*predicted@q+offset)).max())
                assert known_error < 1e-10
                record = {k: v for k, v in case.items() if k not in ['predicted', 'pose', 'betas', 'translation']}
                record.update(primary_transform=output_transform(transform),
                              reverse_topology_diagnostic_transform=output_transform(alternative),
                              primary_correspondence_count=18439,
                              primary_human_fit_mean_vertex_residual_cm=float(100*residual.mean()),
                              primary_human_fit_p95_vertex_residual_cm=float(100*np.percentile(residual, 95)),
                              mapping_affine_equivariance_max_error_m=max(error, forward_error),
                              kit_known_Sim3_max_error_m=known_error,
                              GT_body_units='metres; native camera1/world frame',
                              predicted_body_units='metres; OpenCV camera frame',
                              native_object_GT_fields_accessed=False)
                records.append(record)
                print(case['sequence'], 'scale', record['primary_transform']['scale'],
                      'human fit cm', record['primary_human_fit_mean_vertex_residual_cm'], flush=True)
    report = {'schema': 'human-only-first-frame-sim3-sidecars-v1', 'records': records,
              'transform_convention': 'X_GT = scale * (X_pred @ R.T) + t',
              'primary_policy': 'first scored frame; GT SMPLH6890 to MHR18439 via official template barycentric map; exact kit fit_similarity_transform on all matched MHR vertices',
              'fit_function': str(KIT/'v2dlb/mhr_metrics.py')+':fit_similarity_transform',
              'object_pose_composition': 'object_scale_new=s*object_scale_old; R_obj_new=R_align@R_obj_old; t_new=s*(R_align@t_old)+t_align',
              'single_transform_per_clip': True, 'GT_human_only_for_scoring': True,
              'native_object_GT_fields_accessed': False, 'ICP_used': False,
              'exact_hosted_alignment': False,
              'surrogate_warning': 'BEHAVE GT is SMPLH, mapped to MHR topology; challenge native MHR GT correspondences are not available here. This sidecar never becomes an inference input.',
              'decoder_parity': parity, 'source_sha256': sources}
    args.out.mkdir(parents=True)
    (args.out/'TRANSFORMS.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('saved', args.out/'TRANSFORMS.json', flush=True)


if __name__ == '__main__':
    main()
