#!/usr/bin/env python3
"""Bounded 4-BEHAVE-clip Hunyuan shape proposal trial, predicted masks only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from hunyuan21_shape_adapter import ROOT, sha


def pick_candidate(rows):
    for threshold in [.8, .3]:
        eligible = [(r['area'], i) for i, r in enumerate(rows) if r['selected'] and r['score'] >= threshold]
        if eligible:
            _, index = max(eligible)  # same first area-ranked candidate as SAM3D
            return index, threshold
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--host-peak-limit-gib', type=float, default=56.)
    ap.add_argument('--out', type=Path, default=ROOT/'labeled_predictions_v1/hunyuan21_shapes_v1')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('output directory must be new')
    request = ROOT/'labeled_predictions_v1/RUN_INPUTS.json'
    manifest = json.loads(request.read_text())
    if manifest['GT_fields_included'] is not False:
        raise ValueError('runtime manifest must exclude GT')
    adapter = ROOT/'code/hunyuan21_shape_adapter.py'
    sources = {str(p):sha(p) for p in [Path(__file__), adapter, request]}
    items = [i for i in manifest['items'] if i['dataset']=='BEHAVE']
    if len(items) != 4:
        raise ValueError('bounded initial cohort must be four frozen BEHAVE clips')
    if any(not (Path(i['masks'])/'MASKS.json').is_file() for i in items):
        raise ValueError('all four mask manifests required before creating output')
    args.out.mkdir(parents=True)
    plan = []
    for item in items:
        masks = Path(item['masks'])/'MASKS.json'
        sources[str(masks)] = sha(masks)
        rows = json.loads(masks.read_text())['rows']
        if len(rows)!=len(item['source_paths']):
            raise ValueError('mask/RGB frame count mismatch')
        index, threshold = pick_candidate(rows)
        if index is None:
            plan.append({'sequence':item['sequence'],'status':'no_admissible_predicted_mask'});continue
        row=rows[index];rgb=Path(item['source_paths'][index]);mask=Path(row['mask'])
        if sha(rgb)!=item['source_sha256'][index] or row['source_rgb_sha256']!=item['source_sha256'][index] or sha(mask)!=row['mask_sha256'] or str(rgb)!=row['source_rgb']:
            raise ValueError('selected RGB/mask provenance mismatch')
        image=np.asarray(Image.open(rgb).convert('RGB'));binary=np.asarray(Image.open(mask))>127
        if binary.shape!=image.shape[:2] or not binary.any() or binary.all():
            raise ValueError('predicted mask is empty, full, or wrong resolution')
        # Byte-equivalent SAM3D Inference.merge_mask_to_rgba construction.
        rgba=np.concatenate([image,(binary.astype(np.uint8)*255)[...,None]],axis=-1)
        out=args.out/item['name'];out.mkdir()
        rgba_path=out/'input_predicted_rgba.png';Image.fromarray(rgba,'RGBA').save(rgba_path)
        plan.append({'sequence':item['sequence'],'status':'planned','local_frame_index':index,
                     'video_frame_index':int(rgb.stem),'mask_threshold':threshold,
                     'fallback_outside_SAM3D_highconfidence_pair':threshold<.8,
                     'mask_confidence':row['score'],'mask_area':row['area'],
                     'source_rgb':str(rgb),'source_rgb_sha256':sha(rgb),'source_mask':str(mask),'source_mask_sha256':sha(mask),
                     'rgba':str(rgba_path),'rgba_sha256':sha(rgba_path),'output':str(out/'proposal')})
    run={'schema':'hunyuan21-labeled-open-world-plan-v1','GT_inputs_provided':False,'native_CAD_input':False,
         'selection':'first largest-area predicted-mask candidate at >=0.8; explicit >=0.3 fallback if none; one per clip',
         'source_sha256':sources,'steps':args.steps,'resolution':args.resolution,'seed':42,'records':plan}
    (args.out/'PLAN.json').write_text(json.dumps(run,indent=2))
    completed=0
    for rec in plan:
        if rec['status']!='planned':continue
        cmd=[sys.executable,str(adapter),'--rgba',rec['rgba'],'--rgba-sha256',rec['rgba_sha256'],'--out',rec['output'],
             '--seed','42','--steps',str(args.steps),'--resolution',str(args.resolution)]
        with (Path(rec['output']).parent/'generation.log').open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        rec['exit_code']=result.returncode
        if result.returncode:
            rec['status']='generation_failed';run['stop_reason']='first generation error; bounded run stops without later clips'
            (args.out/'RESULT.json').write_text(json.dumps(run,indent=2));raise SystemExit(result.returncode)
        proposal=json.loads((Path(rec['output'])/'PROPOSAL.json').read_text())
        peak=proposal['cuda_peak_reserved_bytes'];limit=proposal['cuda_total_memory_bytes']*.90
        rec.update(status='generated',proposal=str(Path(rec['output'])/'PROPOSAL.json'),proposal_sha256=sha(Path(rec['output'])/'PROPOSAL.json'))
        completed+=1
        run['completed']=completed
        (args.out/'RESULT.json').write_text(json.dumps(run,indent=2))
        print(rec['sequence'],'generated', 'GPUpeak',peak,'hostKiB',proposal['host_peak_rss_KiB'],flush=True)
        if peak>limit or proposal['host_peak_rss_KiB']>args.host_peak_limit_gib*1024*1024:
            run['stop_reason']=f'valid mesh but peak memory exceeds continuation gate (90% GPU or {args.host_peak_limit_gib}GiB host)'
            (args.out/'RESULT.json').write_text(json.dumps(run,indent=2));raise SystemExit(2)
    if not completed:raise RuntimeError('no admissible predicted mask; no geometry generated')
    run['completed_all_admissible']=True
    (args.out/'RESULT.json').write_text(json.dumps(run,indent=2))


if __name__=='__main__':main()
