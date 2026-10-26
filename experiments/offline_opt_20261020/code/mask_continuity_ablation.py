#!/usr/bin/env python3
"""Image-only association-jump gate, followed by bounded pose interpolation.

This is a diagnostic temporal prior. It does not inspect GT, depth or scores.
Rejected spans remain in the predictions and in subsequent scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
PROTOCOL = dict(median_window=31, normalized_centroid_threshold=2.,
                area_ratio_low=.5, area_ratio_high=2., max_interior_gap=16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected_spans(flags):
    ids = np.flatnonzero(flags)
    return np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(ids) else []


def interpolate(frames, flags):
    result = json.loads(json.dumps(frames))
    repaired = []
    times = np.array([x['video_frame_index'] for x in frames], float)
    assert np.all(np.diff(times) > 0)
    for span in rejected_spans(flags):
        a, b = int(span[0]) - 1, int(span[-1]) + 1
        if a < 0 or b >= len(frames) or len(span) > PROTOCOL['max_interior_gap']:
            continue
        ends = [frames[a], frames[b]]
        rotations = Rotation.from_matrix(np.array([x['R_obj2cam'] for x in ends]))
        rr = Slerp(times[[a, b]], rotations)(times[span]).as_matrix()
        fraction = (times[span] - times[a]) / (times[b] - times[a])
        ta, tb = [np.array(x['t_obj2cam_m']) for x in ends]
        tt = ta[None] * (1 - fraction[:, None]) + tb[None] * fraction[:, None]
        for j, index in enumerate(span):
            result[int(index)]['R_obj2cam'] = rr[j].tolist()
            result[int(index)]['t_obj2cam_m'] = tt[j].tolist()
        repaired.extend(int(x) for x in span)
    return result, repaired


def self_test():
    frames = [dict(video_frame_index=i, R_obj2cam=Rotation.from_euler('z', i * 10, degrees=True).as_matrix().tolist(),
                   t_obj2cam_m=[float(i), 0., 1.]) for i in range(20)]
    flags = np.zeros(20, bool); flags[3:6] = True
    corrupted = json.loads(json.dumps(frames))
    for i in [3, 4, 5]: corrupted[i]['t_obj2cam_m'] = [100., 100., 100.]
    repaired, ids = interpolate(corrupted, flags)
    assert ids == [3, 4, 5]
    for i in range(20):
        assert np.allclose(repaired[i]['t_obj2cam_m'], frames[i]['t_obj2cam_m'])
        assert np.allclose(repaired[i]['R_obj2cam'], frames[i]['R_obj2cam'])
    flags[:] = False; flags[:2] = True; flags[3:20] = True
    assert interpolate(corrupted, flags)[1] == []


def audit_masks(inputs, output):
    data = json.loads(inputs.read_text()); records = []; sources = {str(inputs): sha(inputs)}
    for item in data['items']:
        if item['dataset'] != 'BEHAVE': continue
        manifest = Path(item['masks']) / 'MASKS.json'; rows = json.loads(manifest.read_text())['rows']
        sources[str(manifest)] = sha(manifest); features = []
        for row in rows:
            p = Path(row['mask']); sources[str(p)] = sha(p)
            y, x = np.nonzero(np.asarray(Image.open(p)) > 127)
            features.append([float(x.mean()) if len(x) else np.nan,
                             float(y.mean()) if len(y) else np.nan, len(x)])
        features = np.array(features); n = len(features)
        med = np.array([np.nanmedian(features[max(0, i-15):min(n, i+16)], axis=0) for i in range(n)])
        dev = np.linalg.norm(features[:, :2] - med[:, :2], axis=1) / np.sqrt(np.maximum(med[:, 2], 1))
        ratio = features[:, 2] / np.maximum(med[:, 2], 1)
        flags = (dev > PROTOCOL['normalized_centroid_threshold']) & ((ratio < PROTOCOL['area_ratio_low']) | (ratio > PROTOCOL['area_ratio_high']))
        records.append(dict(sequence=item['name'], source_frame_ids=[int(Path(p).stem) for p in item['source_paths']],
                            flags=flags.tolist(), rejected_local_indices=np.flatnonzero(flags).tolist(),
                            normalized_centroid_deviation=dev.tolist(), area_ratio=ratio.tolist(),
                            features_centroid_xy_area=features.tolist()))
    audit = dict(protocol=PROTOCOL, source_sha256=sources, records=records,
                 GT_used=False, full_clip_image_evidence=True,
                 limitation='A temporal association prior; fast true motion and occlusion can produce false positives.')
    output.write_text(json.dumps(audit, indent=2, allow_nan=False))
    return audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', type=Path)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, default=ROOT/'labeled_predictions_v2/RUN_INPUTS.json')
    args = ap.parse_args(); self_test()
    if args.audit.exists():
        audit = json.loads(args.audit.read_text()); assert audit['protocol'] == PROTOCOL
        for p, h in audit['source_sha256'].items(): assert sha(p) == h, p
    else:
        audit = audit_masks(args.inputs, args.audit)
    if args.source is None: return
    args.out.mkdir(exist_ok=False); lookup = {x['sequence']: x for x in audit['records']}; reports = []
    for original in sorted(args.source.glob('*_PREDICTIONS.json')):
        data = json.loads(original.read_text()); counts = {}
        for sequence in data['sequences']:
            row = lookup[sequence['sequence']]
            assert [x['video_frame_index'] for x in sequence['frames']] == row['source_frame_ids']
            result, repaired = interpolate(sequence['frames'], row['flags'])
            assert len(result) == len(sequence['frames'])
            sequence['frames'] = result; counts[sequence['sequence']] = repaired
            sequence['source']['mask_continuity'] = dict(audit_sha256=sha(args.audit),
                original_prediction_sha256=sha(original), repaired_local_indices=repaired,
                interpolation='linear camera translation and proper SO3 Slerp; bounded interior spans only')
        target = args.out/original.name; target.write_text(json.dumps(data, indent=2, allow_nan=False))
        reports.append(dict(prediction=str(target), prediction_sha256=sha(target),
                            source=str(original), source_sha256=sha(original), repaired=counts))
    assert len(reports) == 6
    (args.out/'RESULT.json').write_text(json.dumps(dict(protocol=PROTOCOL, audit_sha256=sha(args.audit),
        records=reports, GT_used=False, all_frames_retained=True), indent=2))


if __name__ == '__main__': main()
