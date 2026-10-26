#!/usr/bin/env python3
"""Scorer-only development audit. Decodes selected native IDs, never reserve records."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SCORER = ROOT/'benchmarks/labeled_dev_v2/scorer'
CODE = Path('/home/tanuj/intercap_contact_sources/CONTHO_RELEASE')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def project(points, camera):
    return points[:, :2]/points[:, 2:3]*camera['focal']+camera['princpt']


def main():
    from labeled_object_scorer import selected_native_records
    reference_path = SCORER/'INTERCAP_LABEL_REFERENCES.json'
    request_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    ref = json.loads(reference_path.read_text())
    requests = {r['take']: r for r in json.loads(request_path.read_text())['InterCap']}
    takes = ref['development_samples']
    assert len(takes) == len(requests) == 8
    assert {r['take'] for r in takes} == set(requests)
    assert ref['reserve_samples_included'] is False
    native = Path(takes[0]['GT_annotation_source'])
    assert all(r['GT_annotation_source'] == str(native) for r in takes)
    assert sha(native) == ref['native_annotations_sha256']
    image_ids = [i for r in takes for i in r['image_ids']]
    annotation_ids = [i for r in takes for i in r['annotation_ids']]
    assert len(image_ids) == len(set(image_ids)) == len(annotation_ids) == len(set(annotation_ids)) == 246
    images, annotations = selected_native_records(native, image_ids, annotation_ids)
    out = ROOT/'intercap_native_contract_v1'
    assert not out.exists()
    out.mkdir()
    sources = {str(p.resolve()): sha(p) for p in [Path(__file__), reference_path,
        request_path, native, ROOT/'code/labeled_object_scorer.py',
        CODE/'data/InterCap/dataset.py', CODE/'lib/utils/aug_utils.py', CODE/'lib/models/templates.py']}
    records = []; panels = []; first_bodies = []
    for take in takes:
        req = requests[take['take']]
        assert take['camera'] == req['camera'] == 1 and req['source_frame_cadence'] == 5
        frames = req['native_frame_numbers']
        assert len(frames) == len(take['file_names']) == len(take['annotation_ids']) == len(take['image_ids'])
        assert np.array_equal(np.diff(frames), np.full(len(frames)-1, 5))
        mesh = trimesh.load(take['scorer_mesh'], force='mesh', process=False)
        vertices = np.asarray(mesh.vertices, np.float64)
        assert sha(take['scorer_mesh']) == take['scorer_mesh_sha256']
        assert np.isfinite(vertices).all() and .01 < np.ptp(vertices, axis=0).max() < 5
        sources[take['scorer_mesh']] = sha(take['scorer_mesh'])
        residuals = []; depths = []; bounds = []; inside_fractions = []
        for i, (aid, iid, name, frame) in enumerate(zip(take['annotation_ids'], take['image_ids'], take['file_names'], frames)):
            image = images[iid]; annotation = annotations[aid]
            assert annotation['image_id'] == iid and image['file_name'] == name
            assert name.startswith(take['take']+'/Frames_Cam1/color/') and int(Path(name).stem) == frame
            assert req['rgb_paths'][i].endswith('/'+name) and annotation['obj_param']['name'] == take['take'].split('/')[1]
            assert set(image['cam_param']) == {'focal', 'princpt'}
            camera = image['cam_param']; assert np.isfinite(list(camera.values())).all()
            assert min(camera['focal']) > 0 and (image['width'], image['height']) == (1920, 1080)
            rgb = Path(req['rgb_paths'][i]); assert sha(rgb) == req['rgb_sha256'][i]
            sources[str(rgb)] = req['rgb_sha256'][i]
            pose = annotation['obj_param']; rotation = Rotation.from_rotvec(pose['pose']).as_matrix()
            translation = np.asarray(pose['trans'], np.float64)
            points = vertices@rotation.T+translation
            assert np.isfinite(points).all() and points[:, 2].min() > 0
            uv = project(points, camera)
            inside = (uv[:, 0] >= 0)&(uv[:, 0] < image['width'])&(uv[:, 1] >= 0)&(uv[:, 1] < image['height'])
            depths.append(float(points[:, 2].min())); inside_fractions.append(float(inside.mean()))
            bounds.append([*uv.min(0).tolist(), *uv.max(0).tolist()])
            human3d = np.asarray(annotation['h3d_keypoints']); human2d = np.asarray(annotation['h2d_keypoints'])
            assert human3d.shape == (73, 3) and human2d.shape == (73, 2) and (human3d[:, 2] > 0).all()
            residuals.extend(np.linalg.norm(project(human3d, camera)-human2d, axis=1).tolist())
            if i == 0:
                first_bodies.append(dict(take=take['take'], annotation_id=aid,
                    image_id=iid, camera=camera, native_frame=frame,
                    smpl_param=annotation['smpl_param'], h3d_keypoints=annotation['h3d_keypoints']))
            if i in {0, len(frames)//2, len(frames)-1}:
                raw = Image.open(rgb).convert('RGB'); assert raw.size == (1920, 1080)
                painted = raw.copy(); draw = ImageDraw.Draw(painted)
                for x, y in uv[::max(1, len(uv)//1600)]:
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 230, 0))
                for x, y in human2d:
                    draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 255, 255))
                low = uv.min(0)-70; high = uv.max(0)+70
                center = (low+high)/2; span = np.maximum(high-low, [320, 300])
                box = tuple(np.r_[np.maximum(center-span/2, [0, 0]), np.minimum(center+span/2, [1920, 1080])].astype(int))
                pair = Image.new('RGB', (640, 322), 'black')
                for k, im in enumerate([raw, painted]):
                    tile = ImageOps.contain(im.crop(box), (320, 300)); pair.paste(tile, (k*320, 22))
                ImageDraw.Draw(pair).text((5, 4), f"{take['take']} frame{frame} raw | native mesh/yellow, human/cyan", fill='white')
                panels.append(pair)
        maximum = float(max(residuals))
        assert maximum < .01, 'native camera3D-to-2D reprojection mismatch'
        records.append(dict(take=take['take'], frames=len(frames),
            native_object_dimensions_m=np.ptp(vertices, axis=0).tolist(),
            native_object_vertex_mean_retained_m=vertices.mean(0).tolist(),
            minimum_object_vertex_Z_m=min(depths),
            minimum_projected_object_vertex_fraction_inside_RGB=min(inside_fractions),
            projected_object_bboxes_px=bounds,
            human73_reprojection_mean_px=float(np.mean(residuals)),
            human73_reprojection_max_px=maximum,
            all_ID_camera_frame_object_joins_passed=True,
            object_transform='X_camera = V_native @ Rodrigues(axis_angle_radians).T + translation_metres',
            timestamp_basis='native frame numbers / declared30fps; nominal not measured hardware time',
            source_frame_cadence=5, nominal_retained_dt_seconds=5/30,
            exact_hosted_ACC_compatible=False))
    for group in range(2):
        canvas = Image.new('RGB', (1920, 1288), 'black')
        for j, panel in enumerate(panels[group*12:(group+1)*12]):
            canvas.paste(panel, ((j%3)*640, (j//3)*322))
        path = out/f'projection_QA_group{group+1}.jpg'; canvas.save(path, quality=93)
    (out/'FIRST_FRAME_NATIVE_HUMANS.json').write_text(json.dumps(dict(
        schema='intercap-development-only-firstframe-human-reference-v1',
        records=first_bodies, scorer_only=True, native_object_fields_included=False), indent=2))
    (out/'CHECKS.json').write_text(json.dumps(dict(schema='intercap-native-contract-checks-v1',
        records=records, decoded_images=len(images), decoded_annotations=len(annotations),
        reserve_payloads_decoded=0, source_sha256=sources,
        projection_visual_QA_required_before_publish=True), indent=2))
    print(json.dumps({'out':str(out), 'records':records}, indent=2))


if __name__ == '__main__':
    main()
