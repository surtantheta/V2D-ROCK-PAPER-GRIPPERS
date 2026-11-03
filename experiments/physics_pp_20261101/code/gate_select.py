#!/usr/bin/env python3
"""Per-episode choice between two episode directories by a rule on the candidate's own report (no GT, no scores).

Used for the gated contact snap: take the snapped episode only where the median closest-hand gap before snapping
exceeds --min-gap-cm (scored frames are inside the contact span, so a large gap means a depth/scale error), and
keep the base episode elsewhere, where the snap objective is flat and the move is noise.
"""
import argparse, json, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--cand', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--min-gap-cm', type=float, default=5.0); ap.add_argument('--episodes', type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out) / 'episodes'; out.mkdir(parents=True, exist_ok=True)
    picks = {}
    for e in range(a.episodes):
        name = f'episode_{e:06d}'
        cand = Path(a.cand) / 'episodes' / f'{name}.json'
        rep = json.loads(cand.read_text()) if cand.exists() else {'closest_hand_cm_before_p10_p50_p90': [None, -1.0, None]}
        gap = rep['closest_hand_cm_before_p10_p50_p90'][1]               # missing candidate: keep the base
        src = Path(a.cand if gap > a.min_gap_cm else a.base) / 'episodes'
        for ext in ('npz', 'json'):
            shutil.copyfile(src / f'{name}.{ext}', out / f'{name}.{ext}')
        picks[e] = {'source': str(src), 'gap50_cm_before': gap, 'k': rep.get('k')}
    (Path(a.out) / 'GATE.json').write_text(json.dumps({'args': vars(a), 'picks': picks}, indent=1))
    print(json.dumps({e: (p['source'].split('/')[-2], round(p['gap50_cm_before'], 1)) for e, p in picks.items()}))


if __name__ == '__main__':
    main()
