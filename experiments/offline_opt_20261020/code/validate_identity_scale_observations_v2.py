#!/usr/bin/env python3
"""Synthetic integration checks for camera/gauge/occlusion/contact observations."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import trimesh

from run_identity_scale_v2 import observe, export_validated, sha, ROOT


def main():
    with tempfile.TemporaryDirectory(prefix='v2d-observation-contract-') as temp:
        folder = Path(temp); depthdir = folder/'depth'; depthdir.mkdir()
        k = np.array([[16., 0, 8.], [0, 16., 8.], [0, 0, 1.]])
        tetra = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]) + [.4, .3, .2]
        vertices = np.vstack([tetra, np.tile(tetra[0], (4092, 1))])
        faces = np.tile([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], (1024, 1))
        meshpath = folder/'mesh.ply'
        trimesh.Trimesh(vertices, faces, process=False).export(meshpath)
        run = dict(model_wh=[16, 16], source_wh=[16, 16], K_fixed_source=k.tolist())
        (depthdir/'RUN.json').write_text(json.dumps(run))
        rows = []
        for i in range(2):
            obj = folder/f'object{i}.png'; person = folder/f'person{i}.png'
            Image.fromarray(np.ones((16, 16), np.uint8)*255).save(obj)
            Image.fromarray(np.ones((16, 16), np.uint8)*(255 if i == 0 else 0)).save(person)
            np.savez_compressed(depthdir/f'{i:06d}.npz', depth=np.ones((16, 16))*3.)
            rows.append(dict(selected=True, score=.9, mask=str(obj), mask_sha256=sha(obj),
                             person_mask=str(person), person_mask_sha256=sha(person)))
        manifest = folder/'MASKS.json'; manifest.write_text(json.dumps(dict(rows=rows)))
        poses = np.tile(np.eye(4), (2, 1, 1)); poses[:, 2, 3] = 2.
        posespath = folder/'poses.npz'; np.savez_compressed(posespath, T_obj2cam=poses, iou=np.ones(2))
        tips = folder/'tips.npz'
        np.savez_compressed(tips, fingertips_cam_m=np.tile([0., 0., 1.], (10, 1)),
                            tip_names=np.array([f'{side}{i}' for side in ['l', 'r'] for i in range(5)]), K=k)
        sequence = dict(mesh_path=str(meshpath), mesh_sha256=sha(meshpath), object_scale=2.,
            source=dict(human_anchor=dict(alpha=2.), identity_mask_manifest=str(manifest),
                        poses_path=str(posespath), poses_sha256=sha(posespath)),
            frames=[dict(R_obj2cam=np.eye(3).tolist(), t_obj2cam_m=[0., 0., 4.], video_frame_index=i) for i in range(2)])
        contact = dict(frames=[dict(local_frame_index=0, video_frame_index=0,
                                   tips_path=str(tips), tips_sha256=sha(tips))])
        sources = {}
        obs, audit = observe(sequence, dict(source_paths=['synthetic0', 'synthetic1']),
                             depthdir, contact, [0., .1], 1, sources)
        assert len(obs['clouds'][0]) == 0 and len(obs['image_observations'][0]['maskuv']) == 128
        assert len(obs['clouds'][1]) == 196 and np.all(obs['clouds'][1][:, 2] == 6.)
        assert len(obs['contacts'][0]) == 2
        assert all(np.all(hand[:, 2] == 1.) for hand in obs['contacts'][0])
        np.testing.assert_array_equal(obs['baseline_translation'], [[0, 0, 4], [0, 0, 4]])
        expected_mu = 2.*np.average(np.asarray(trimesh.load(meshpath, process=False).triangles).mean(1),
            axis=0, weights=trimesh.load(meshpath, process=False).area_faces)
        np.testing.assert_allclose(obs['mu_metric'], expected_mu, atol=1e-12)
        vertices_metric = np.asarray(trimesh.load(meshpath, process=False).vertices)*2.
        for candidate, ratio, diagnostics in [
                (np.full((2, 3), np.nan), 1., {}),
                (obs['baseline_translation'].copy(), 1., dict(final_objective=dict(total=float('nan')))),
                (np.tile([0., 0., -2.], (2, 1)), 1., {}),
                (obs['baseline_translation'].copy(), float('inf'), {})]:
            t, scale, checked = export_validated(candidate, ratio, diagnostics, obs, vertices_metric)
            assert checked['export_validation']['whole_case_identity_fallback'] and scale == 1.
            np.testing.assert_array_equal(t, obs['baseline_translation'])
            json.dumps(checked, allow_nan=False)
        # A local mesh origin is arbitrary: physical vertices, not t_z, decide validity.
        shifted_vertices = vertices_metric + [0., 0., 5.]
        negative_origin = obs['baseline_translation'] - [0., 0., 5.]
        shifted_obs = dict(obs, mu_metric=obs['mu_metric']+[0., 0., 5.])
        t, scale, checked = export_validated(negative_origin, 1., {}, shifted_obs, shifted_vertices)
        assert not checked['export_validation']['whole_case_identity_fallback']
        assert np.all(t[:, 2] < 0)
        result = dict(positive_object_coverage_survives_person_overlap=True,
            depth_alpha_once=True, baseline_translation_alpha_once=True,
            body_tips_not_scaled=True, no_initial_3d_depth_admission=True,
            no_initial_3d_contact_admission=True, exact_area_centroid_in_same_metric_gauge=True,
            nonfinite_and_camera_invalid_candidates_fallback_without_case_drop=True,
            nonfinite_candidate_diagnostics_json_safe=True, negative_origin_valid_for_front_camera_mesh=True,
            synthetic_sources=len(sources), source_sha256={str(Path(__file__)): sha(__file__),
                str(ROOT/'code/run_identity_scale_v2.py'): sha(ROOT/'code/run_identity_scale_v2.py')})
        out = ROOT/'identity_scale_v2/OBSERVATION_SYNTHETICS_FINAL.json'
        if out.exists():
            assert json.loads(out.read_text()) == result, 'immutable synthetic receipt changed'
        else:
            out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
