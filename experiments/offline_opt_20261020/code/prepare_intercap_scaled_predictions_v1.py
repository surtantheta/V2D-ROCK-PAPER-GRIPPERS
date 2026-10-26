#!/usr/bin/env python3
"""Apply one predicted-human alpha to the mesh and complete camera trajectory."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runtime = ROOT/'labeled_intercap_v1'; raw = runtime/'baseline/foundationpose_raw_PREDICTIONS.json'
    hp = runtime/'human_scale/RESULT.json'; manifest = runtime/'FP_RUNTIME.json'
    anchor = json.loads(hp.read_text()); mapping = {x['take']: x['name'] for x in json.loads(manifest.read_text())['records']}
    tracking = json.loads((runtime/'TRACKING_COMPLETE.json').read_text())
    assert tracking['prediction_sha256'] == sha(raw) and tracking['frames'] == 246 and tracking['takes'] == 8
    for p, expected in anchor['source_sha256'].items(): assert sha(p) == expected, p
    alphas = {x['sequence']: x for x in anchor['records']}; pred = json.loads(raw.read_text()); notes = []
    assert len(mapping) == len(alphas) == len(pred['sequences']) == 8
    assert not anchor['GT_inputs_provided'] and not anchor['native_object_geometry_provided']
    for seq in pred['sequences']:
        name = mapping[seq['sequence']]; assert seq['source']['runtime_name'] == name
        entry = alphas[name]; alpha = entry['alpha']; fallback = alpha is None
        if fallback: alpha = 1.
        assert np.isfinite(alpha) and alpha > 0
        seq['object_scale'] *= alpha
        for frame in seq['frames']: frame['t_obj2cam_m'] = (alpha*np.array(frame['t_obj2cam_m'])).tolist()
        seq['source']['human_anchor'] = dict(alpha=alpha, no_anchor_fallback_identity=fallback,
            anchor_result_sha256=sha(hp), original_prediction_sha256=sha(raw),
            same_scale_applied_to_geometry_and_translation=True, GT_used_to_estimate_alpha=False)
        notes.append(dict(sequence=seq['sequence'], name=name, alpha=alpha, fallback_identity=fallback))
    assert sum(len(s['frames']) for s in pred['sequences']) == 246
    output = runtime/'baseline/foundationpose_human_scale_PREDICTIONS.json'; assert not output.exists()
    output.write_text(json.dumps(pred, indent=2, allow_nan=False))
    (runtime/'SCALE_COMPLETE.json').write_text(json.dumps(dict(prediction=str(output), prediction_sha256=sha(output),
        notes=notes, full_frames=246, GT_inputs_provided=False,
        source_sha256={str(p): sha(p) for p in [Path(__file__), raw, hp, manifest]}), indent=2))


if __name__ == '__main__': main()
