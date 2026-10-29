"""Rebuild the image-only production manifest from the original MAIN artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

R = Path('/home/tanuj/V2D_challenge')
K = R/'experiments/pi3x_kaggle_20261029'
L = Path('/rhome/tanuj/v2d_pi3x_kaggle_20261029')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def prepare(check_only=False):
    model = json.loads((K/'MODEL_SPEC.json').read_text())
    choice = json.loads((R/'work/MASK_CHOICE.json').read_text())
    pins = {str(R/'work/MASK_CHOICE.json'): sha(R/'work/MASK_CHOICE.json')}
    cases, metadata = [], {}
    for ep in range(30):
        name = f'episode_{ep:06d}'
        human = R/f'work/submission_v2/{name}.npz'
        with np.load(human, allow_pickle=False) as data:
            n = len(data['pose'])
        track = R/f'work/s6/{name}/TRACK.json'
        staged = R/f'work/frames/{name}/FRAMES.json'
        assert json.loads(track.read_text())['frames'] == json.loads(staged.read_text())['frames'] == n
        masks = R/'work'/('masks_v2' if choice.get(str(ep)) == 'v2' else 'masks')/name/'MASKS.json'
        rows = {row['frame']: row for row in json.loads(masks.read_text())['rows']}
        frames = []
        for i in np.rint(np.linspace(0, n-1, 16)).astype(int).tolist():
            rgb = R/f'work/frames/{name}/{i:06d}.jpg'
            row = rows[f'{i:06d}.jpg']
            mask = Path(row['mask']) if row.get('mask') else None
            pins[str(rgb)] = sha(rgb)
            if mask:
                pins[str(mask)] = sha(mask)
            frames.append(dict(local_index=i, native_frame_id=i, time_seconds=i/30,
                rgb=str(rgb), rgb_wh=list(Image.open(rgb).size), rgb_sha256=pins[str(rgb)],
                mask=str(mask) if mask else None, mask_sha256=pins.get(str(mask)),
                mask_selected=bool(row.get('selected', False)), mask_score=float(row.get('score', 0))))
        assert len({tuple(f['rgb_wh']) for f in frames}) == 1
        for path in [human, masks, track, staged]:
            pins[str(path)] = sha(path)
        ids = list(range(n))
        times = [i/30 for i in ids]
        metadata[str(ep)] = dict(frames=n, frame_ids=ids, times=times, source_wh=frames[0]['rgb_wh'],
            crop_xyxy=None, dataset='challenge', sequence=name)
        cases.append(dict(episode=ep, dataset='challenge', sequence=name, total_frames=n,
            sampling='16 roundedlinspace wholeclip fixed beforeinference', native_wh=frames[0]['rgb_wh'],
            native_to_staged_crop_xyxy=None, fixed_MAIN_human_npz=str(human), human_conditioning_used=False,
            mask_manifest=str(masks), frames=frames, full_native_frame_ids=ids, full_frame_times_seconds=times))
    manifest = dict(schema='pi3x-challenge30-geometry-v1', cases=cases, models=[model['model']],
        input_sha256=pins, large_output_root=str(L/'geometry'), hf_cache=model['hf_cache'],
        model_conditioning='RGBonly; noGT/camera/bodyinputs',
        controls='No repeatedcontrols; identicalofficialnormalinferencepath alreadyvalidatedOct28',
        no_ground_truth_or_native_calibration=True)
    for name, data in [('PILOT_INPUTS.json', manifest), ('CHALLENGE_EPISODES.json', metadata)]:
        path = K/name
        if check_only:
            assert json.loads(path.read_text()) == data, ('Production input replay differs', name)
        else:
            assert not path.exists(), 'Preserve frozen inputs; use a new output version'
            path.write_text(json.dumps(data, indent=2)+'\n')
    if not check_only:
        sources = dict(model['provider_source_sha256'])
        for path in [K/'code/run_geometry_model.py', K/'code/geometry_contract.py',
                     K/'PILOT_INPUTS.json', K/'CHALLENGE_EPISODES.json']:
            sources[str(path)] = sha(path)
        path = K/'SOURCE_FREEZE.json'
        assert not path.exists()
        path.write_text(json.dumps(dict(sha256=sources, checkpoint=model['model']), indent=2)+'\n')
    print('PRODUCTION_MANIFEST_REPLAY_PASS' if check_only else 'PRODUCTION_INPUTS_FROZEN', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args().check_only)
