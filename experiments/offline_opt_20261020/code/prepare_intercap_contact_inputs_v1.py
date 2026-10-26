#!/usr/bin/env python3
"""Frozen MHR -> named SMPLH fingertip proxy for InterCap RGB predictions only."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PROJECT = ROOT.parents[1]
REFERENCE_SHA = 'c97d7d6667227f6d8cc8847f6345a4bccb12934616d133d9a79dd7ae86200cea'
EXPECTED_NAMES = [side+finger for side in ['l', 'r'] for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']]
EXPECTED_IDS = [2746, 2319, 2445, 2556, 2673, 6191, 5782, 5905, 6016, 6133]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'labeled_intercap_contact_inputs_v1')
    args = ap.parse_args()
    runtime = ROOT/'labeled_intercap_v1'
    paths = dict(inputs=runtime/'RUN_INPUTS.json', fp=runtime/'FP_RUNTIME.json',
        human=runtime/'human_scale/RESULT.json', reference=ROOT/'labeled_contact_inputs_v1/CONTACT_INPUTS.json',
        mapping=PROJECT/'third_party/MHR/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz',
        faces=PROJECT/'models/mhr/mhr_faces_lod1.npy',
        vertex_ids=Path('/home/tanuj/.conda/envs/cari4d/lib/python3.10/site-packages/smplx/vertex_ids.py'),
        body_models=Path('/home/tanuj/.conda/envs/cari4d/lib/python3.10/site-packages/smplx/body_models.py'),
        conversion=PROJECT/'third_party/MHR/tools/mhr_smpl_conversion/conversion.py')
    assert all(p.is_file() for p in paths.values()), 'require completed human RESULT and FP_RUNTIME plus frozen mapping sources'
    assert sha(paths['reference']) == REFERENCE_SHA
    sources = {str(p): sha(p) for p in [Path(__file__), *paths.values()]}
    reference = json.loads(paths['reference'].read_text())
    assert reference['tip_names'] == EXPECTED_NAMES and reference['SMPLH_vertex_ids'] == EXPECTED_IDS
    assert reference['MHR_faces_match_torchscript'] and not reference['GT_inputs']
    for key in ['mapping', 'faces', 'vertex_ids', 'body_models', 'conversion']:
        assert sha(paths[key]) == reference['source_sha256'][str(paths[key])], key
    inputs = json.loads(paths['inputs'].read_text()); fp = json.loads(paths['fp'].read_text())
    human = json.loads(paths['human'].read_text())
    assert not inputs['GT_fields_included'] and not inputs['reserve_takes_included']
    assert not fp['GT_inputs_provided'] and not fp['reserve_takes_included'] and fp['complete_take_coverage']
    assert not human['GT_inputs_provided'] and not human['native_object_geometry_provided']
    assert human['gates']['authoritative_model_head_topology']['kit_faces_exactly_equal']
    items = {x['take']: x for x in inputs['items']}; fpitems = {x['take']: x for x in fp['records']}
    humans = {x['sequence']: x for x in human['records']}
    assert len(items) == len(fpitems) == len(humans) == 8 and set(items) == set(fpitems)
    assert sum(len(x['source_paths']) for x in items.values()) == 246
    faces = np.load(paths['faces'], allow_pickle=False)
    mapping = np.load(paths['mapping'], allow_pickle=False)
    triangle_ids = mapping['triangle_ids']; weights = mapping['baryc_coords']
    assert faces.shape == (36874, 3) and faces.min() >= 0 and faces.max() == 18438
    assert triangle_ids.shape == (6890,) and weights.shape == (6890, 3)
    assert triangle_ids.min() >= 0 and triangle_ids.max() < len(faces)
    assert np.isfinite(weights).all() and np.max(abs(weights.sum(1)-1)) < 1e-12
    labels = runpy.run_path(str(paths['vertex_ids']))['vertex_ids']['smplh']
    ids = np.array([labels[x] for x in EXPECTED_NAMES]); assert ids.tolist() == EXPECTED_IDS
    indices = faces[triangle_ids[ids]]; bary = weights[ids]
    def interpolate(vertices): return np.sum(vertices[indices]*bary[:, :, None], axis=1)
    rng = np.random.default_rng(20261020); v = rng.normal(size=(18439, 3))
    rr = Rotation.from_rotvec([.2, -.3, .1]).as_matrix(); ss = 1.23; tt = np.array([.3, -.2, 2.])
    affine_error = float(np.max(abs(interpolate(ss*v@rr.T+tt)-(ss*interpolate(v)@rr.T+tt))))
    assert affine_error < 1e-12
    args.out.mkdir(exist_ok=False); records = []; all_tiles = []; expected_total = 0
    for take, item in items.items():
        disk = item['name']; fpi = fpitems[take]
        assert fpi['name'] == disk and fpi['sequence'] == item['sequence'] == take
        assert fpi['native_frame_numbers'] == item['native_frame_numbers']
        assert fpi['source_paths'] == item['source_paths']
        assert np.all(np.diff(item['native_frame_numbers']) == 5)
        assert disk in humans and humans[disk]['sequence'] == disk
        mp = Path(fpi['masks'])/'MASKS.json'; sources[str(mp)] = sha(mp)
        rows = json.loads(mp.read_text())['rows']; n = len(rows)
        assert n == len(item['source_paths'])
        folder = args.out/disk; folder.mkdir(); frames = []; missing = []; tiles = []
        depthrun = Path(fpi['depth'])/'RUN.json'; sources[str(depthrun)] = sha(depthrun)
        source_k = np.array(json.loads(depthrun.read_text())['K_fixed_source'])
        for i in range(0, n, 8):
            expected_total += 1
            body = runtime/'human_scale'/disk/f'body_{i:06d}.npz'
            if not body.exists():
                status = next((x.get('status') for x in humans[disk].get('frames', []) if x['frame'] == i), humans[disk]['status'])
                missing.append(dict(local_frame_index=i, reason='missing_predicted_body_file', human_status=status)); continue
            sources[str(body)] = sha(body); data = np.load(body, allow_pickle=False)
            vertices = np.array(data['vertices_cam'], dtype=np.float64); K = np.array(data['K'], dtype=np.float64)
            assert vertices.shape == (18439, 3) and K.shape == (3, 3)
            assert np.max(abs(K-source_k)) < .001 and K[0, 0] > 0 and K[1, 1] > 0
            tips = interpolate(vertices)
            if not np.isfinite(tips).all() or np.any(tips[:, 2] <= .001):
                missing.append(dict(local_frame_index=i, reason='nonfinite_or_nonpositive_predicted_tip_depth')); continue
            uvw = tips@K.T; uv = uvw[:, :2]/uvw[:, 2:]
            row = rows[i]; rgb = Path(item['source_paths'][i]); person = Path(row['person_mask']); mask = Path(row['mask'])
            for p in [rgb, person, mask]: sources[str(p)] = sha(p)
            assert sha(rgb) == item['source_sha256'][i] == row['source_rgb_sha256']
            assert sha(person) == row['person_mask_sha256'] and sha(mask) == row['mask_sha256']
            vid = item['native_frame_numbers'][i]; assert vid == int(rgb.stem)
            dest = folder/f'tips_{i:06d}.npz'
            np.savez_compressed(dest, fingertips_cam_m=tips, K=K, uv_source_px=uv,
                smplh_vertex_ids=ids, tip_names=np.array(EXPECTED_NAMES), mhr_triangle_vertices=indices,
                barycentric_weights=bary, local_frame_index=np.array(i), video_frame_index=np.array(vid))
            im = Image.open(rgb).convert('RGB'); W, H = im.size; pm = np.asarray(Image.open(person)) > 127
            pix = np.floor(uv).astype(int); inside = (pix[:, 0]>=0)&(pix[:, 0]<W)&(pix[:, 1]>=0)&(pix[:, 1]<H)
            onperson = np.zeros(10, bool); onperson[inside] = pm[pix[inside, 1], pix[inside, 0]]
            frames.append(dict(local_frame_index=i, video_frame_index=vid, tips_path=str(dest), tips_sha256=sha(dest),
                body_path=str(body), body_sha256=sha(body), K_units='source-RGB pixels',
                tips_units='metres; OpenCV camera x-right/y-down/z-forward; no additional alpha',
                object_mask_selected=bool(row['selected']), object_mask_score=row['score'],
                object_mask_path=str(mask), object_mask_sha256=sha(mask), RGB_path=str(rgb), RGB_sha256=sha(rgb),
                projected_tips_in_frame=int(inside.sum()), projected_tips_in_person_mask=int(onperson.sum()),
                no_body_interpolation=True))
            arr = np.array(im); maskpix = np.asarray(Image.open(mask)) > 127
            arr[maskpix] = (arr[maskpix]*.65+np.array([255, 0, 0])*.35).astype('uint8')
            im = Image.fromarray(arr); draw = ImageDraw.Draw(im)
            for label, (u, vv) in zip(EXPECTED_NAMES, uv):
                color = 'cyan' if label[0] == 'l' else 'yellow'
                draw.ellipse((u-7, vv-7, u+7, vv+7), outline=color, width=3)
                draw.text((u+9, vv), label, fill=color)
            yy, xx = np.where(pm)
            crop = im.crop((max(0, int(xx.min())-80), max(0, int(yy.min())-80), min(W, int(xx.max())+80), min(H, int(yy.max())+80))) if len(xx) else im.copy()
            crop.thumbnail((360, 480)); im.thumbnail((360, 480))
            tile = Image.new('RGB', (720, 510), 'white'); tile.paste(im, (0, 0)); tile.paste(crop, (360, 0))
            ImageDraw.Draw(tile).text((4, 486), f'{take} local{i}/native{vid}; selected={row["selected"]}; in-person={onperson.sum()}/10', fill='black')
            tiles.append(tile); all_tiles.append(tile)
        canvas = Image.new('RGB', (4*720, 510), 'white')
        for j, tile in enumerate(tiles): canvas.paste(tile, (j*720, 0))
        canvas.save(args.out/(disk+'_PROJECTION_QA.jpg'))
        records.append(dict(sequence=take, disk_name=disk, name=disk, frames=frames, sparse_stride_retained_frames=8,
            source_frame_cadence=5, nominal_source_fps=30., expected_sparse_frames=list(range(0, n, 8)),
            count=len(frames), missing_body_frames=missing, mask_manifest=str(mp), mask_manifest_sha256=sha(mp)))
    assert len(records) == 8 and expected_total == 32
    for part in range(2):
        canvas = Image.new('RGB', (4*720, 4*510), 'white')
        for j, tile in enumerate(all_tiles[part*16:(part+1)*16]): canvas.paste(tile, ((j%4)*720, (j//4)*510))
        canvas.save(args.out/f'PROJECTION_QA_OVERVIEW_{part}.jpg')
    report = dict(schema='sparse-predicted-MHR-mapped-SMPLH-fingertips-v1', records=records,
        tip_names=EXPECTED_NAMES, SMPLH_vertex_ids=EXPECTED_IDS, source_sha256=sources,
        frozen_BEHAVE_contact_protocol_sha256=REFERENCE_SHA,
        topology='MHR18439/36874 -> official SMPL6890 barycentric map, official SMPLH fingertip labels',
        units='OpenCV camera metres already in predicted vertices_cam; NO extra alpha, /100 or coordinate flip',
        official512_hand_PEN_samples_available=False, contact_proxy_only=True, contact_GT_or_labels_used=False,
        GT_inputs=False, native_object_geometry_inputs=False, bodyframes_interpolated=False,
        expected_sparse_frames=expected_total, generated_sparse_frames=sum(r['count'] for r in records),
        affine_equivariance_max_abs_error_m=affine_error, MHR_faces_match_torchscript_via_frozen_reference=True,
        gate_policy='Unchanged downstream frozen protocol: confidence>=.8, selected mask, 2D<=4modelpixels, initial3D<=.15m, sparseobservedframes only; no forced contact.',
        reserve_takes_included=False, exact_hosted_metrics=False)
    manifest = args.out/'CONTACT_INPUTS.json'; manifest.write_text(json.dumps(report, indent=2, allow_nan=False))
    receipt = dict(manifest=str(manifest), manifest_sha256=sha(manifest), source_sha256=sources,
        output_tip_sha256={f['tips_path']: f['tips_sha256'] for r in records for f in r['frames']},
        contact_proxy_only=True, GT_inputs=False, reserve_takes_included=False)
    (args.out/'CONTACT_INPUTS_RECEIPT.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(dict(manifest=str(manifest), takes=8, expected_sparse_frames=32,
        generated_sparse_frames=report['generated_sparse_frames'], affine_equivariance_error=affine_error), indent=2))


if __name__ == '__main__': main()
