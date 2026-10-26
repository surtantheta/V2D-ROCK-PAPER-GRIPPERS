#!/usr/bin/env python3
"""CPU-only independent frozen-request and exact FP mask/depth initialization gates."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime_path = ROOT/'labeled_intercap_v1/FP_RUNTIME.json'
    request_path = ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json'
    runtime = json.loads(runtime_path.read_text())
    requests = {r['take']: r for r in json.loads(request_path.read_text())['InterCap']}
    records = runtime['records']
    assert len(records) == len(requests) == 8 and runtime['complete_take_coverage']
    assert {r['take'] for r in records} == set(requests)
    checks = []
    for record in records:
        request = requests[record['take']]
        assert record['source_paths'] == request['rgb_paths']
        assert record['source_sha256'] == request['rgb_sha256']
        assert record['native_frame_numbers'] == request['native_frame_numbers']
        assert record['sequence'] == record['take'] and request['camera'] == 1
        key = json.loads((Path(record['directory'])/'keyframe.json').read_text())
        run = json.loads((Path(record['depth'])/'RUN.json').read_text())
        rows = json.loads((Path(record['masks'])/'MASKS.json').read_text())['rows']
        i = key['keyframe']; width, height = run['model_wh']
        assert len(rows) == len(request['rgb_paths'])
        assert key['video_frame_index'] == request['native_frame_numbers'][i]
        K = np.asarray(key['K_predicted'])
        assert np.array_equal(K, np.asarray(run['K_fixed_source']))
        mask = np.asarray(Image.open(rows[i]['mask']).resize((width, height), Image.Resampling.NEAREST)) > 127
        depth = np.load(Path(record['depth'])/f'{i:06d}.npz')['depth'].astype(np.float32)
        ok = rows[i]['selected'] and rows[i]['score'] >= .8 and mask.sum() >= 20
        fallback = key.get('mask_confidence_fallback') and mask.sum() >= 20 and rows[i]['score'] >= .3
        assert ok or fallback
        fraction = float(((depth > 0)&np.isfinite(depth)&mask).sum()/mask.sum())
        assert fraction >= .5
        checks.append(dict(take=record['take'], frames=len(rows), keyframe_local=i,
            keyframe_native=key['video_frame_index'], model_mask_pixels=int(mask.sum()),
            valid_depth_fraction_under_registration_mask=fraction,
            explicit_confidence_fallback=bool(fallback and not ok)))
    assert sum(r['frames'] for r in checks) == 246
    output = dict(records=checks, takes=8, source_frames=246,
        frozen_final_request_join_passed=True, exact_FP_initialization_conditions_passed=True,
        GT_read=False, GPU_used=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), runtime_path, request_path]})
    dest = ROOT.parents[1]/'validation/OPT_INTERCAP_FP_INITIALIZATION_20261020.json'
    dest.write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
