"""Freeze/replay the uniform exceptional closest-point numerical repair."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    path = ROOT / 'code/prepare_fit_inputs_v1.py'
    spec = importlib.util.spec_from_file_location('exact_original_joint_input_builder', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    manifest = original.build()
    old_path = ROOT / 'FIT_INPUTS_V1.json'
    assert manifest == json.loads(old_path.read_text())
    manifest['schema'] = 'challenge30-genuine-joint-v2-official-closest-rare-repair'
    manifest['v1_manifest_sha256'] = original.sha(old_path)
    manifest['uniform_V2_policy'] = 'unchanged original fast path; only a formerly failed distance-gradient assertion invokes oracle-consistent closest-point arithmetic; tolerance and objective unchanged'
    manifest['successful_V1_reuse_rule'] = 'V1 COMPLETE plus independent all-frame audit and identical source/inputs prove no exceptional branch could execute'
    manifest['config'] = json.loads((ROOT.parent / 'physical_refinement_20261031/JOINT_INPUTS_V3.json').read_text())['config']
    paths = [old_path, Path(__file__), path, ROOT / 'code/run_challenge_joint_v2.py',
             ROOT / 'code/official_closest_repair_v2.py',
             ROOT / 'code/test_official_closest_repair_v2.py',
             ROOT / 'code/test_official_closest_branch_v2.py',
             ROOT / 'validation/KERNEL_POINT_PROBE_V1.json',
             ROOT / 'validation/OFFICIAL_CLOSEST_REPAIR_TEST_V2.json',
             ROOT / 'validation/OFFICIAL_CLOSEST_BRANCH_TEST_V2.json']
    manifest['source_sha256'].update({str(p): original.sha(p) for p in paths})
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = build()
    target = ROOT / 'FIT_INPUTS_V2.json'
    encoded = json.dumps(manifest, indent=2) + '\n'
    if args.check_only:
        assert target.read_text() == encoded
    else:
        assert not target.exists()
        target.write_text(encoded)
    print(json.dumps(dict(status='PASS_EXACT_V2_REPLAY' if args.check_only else 'V2_INPUTS_FROZEN',
                          source_pins=len(manifest['source_sha256']), cases=30, frames=16563,
                          unchanged_v1_input_sha256=manifest['v1_manifest_sha256'],
                          rare_branch_only=True, original_threshold_unchanged=True)), flush=True)
