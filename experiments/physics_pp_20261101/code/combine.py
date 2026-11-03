#!/usr/bin/env python3
"""Combine per-episode outputs: MHR pose from one directory, object translation/scale from another."""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', required=True); ap.add_argument('--object', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for e in range(30):
        h = np.load(Path(a.human) / f'episode_{e:06d}.npz'); o = np.load(Path(a.object) / f'episode_{e:06d}.npz')
        assert np.array_equal(h['frames'], o['frames']), e
        np.savez(out / f'episode_{e:06d}.npz', frames=h['frames'], object_translation=o['object_translation'],
                 mhr_pose=h['mhr_pose'], object_scale=o['object_scale'])
        rh = json.loads((Path(a.human) / f'episode_{e:06d}.json').read_text())
        ro = json.loads((Path(a.object) / f'episode_{e:06d}.json').read_text())
        rep = {'episode': e, 'human_from': a.human, 'object_from': a.object, 'pen_before': ro['pen_before'],
               'pen_after': ro['pen_after'], 'own_acc_h_before': rh['own_acc_h_before'], 'own_acc_h_after': rh['own_acc_h_after'],
               'own_acc_o_before': ro['own_acc_o_before'], 'own_acc_o_after': ro['own_acc_o_after'],
               'human_first_frame_exact': rh.get('human_first_frame_exact', True)}
        (out / f'episode_{e:06d}.json').write_text(json.dumps(rep, indent=1))
    print('combined 30 episodes into', out)


if __name__ == '__main__':
    main()
