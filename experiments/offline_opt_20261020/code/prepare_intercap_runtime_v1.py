#!/usr/bin/env python3
"""Freeze only eight development RGB takes and existing predicted object masks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 old_path=ROOT/'labeled_predictions_v1/RUN_INPUTS.json';req_path=ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json';old=json.loads(old_path.read_text());requests=json.loads(req_path.read_text())['InterCap'];allowed={x['take']:x for x in requests};items=[];sources={str(p):sha(p) for p in [Path(__file__),old_path,req_path]}
 for original in old['items']:
  if original['dataset']!='InterCap':continue
  item=dict(original);take=item['take'];expected=allowed[take]
  assert expected['camera']==1 and expected['GT_fields_provided'] is False
  assert item['source_paths']==expected['rgb_paths'] and item['source_sha256']==expected['rgb_sha256']
  object_manifest=Path(item['masks'])/'MASKS.json';obj=json.loads(object_manifest.read_text());assert len(obj['rows'])==len(item['source_paths']);sources[str(object_manifest)]=sha(object_manifest)
  for i,row in enumerate(obj['rows']):
   p=Path(item['source_paths'][i]);m=Path(row['mask']);assert sha(p)==item['source_sha256'][i]==row['source_rgb_sha256'] and str(p)==row['source_rgb'];assert sha(m)==row['mask_sha256'];sources[str(p)]=sha(p);sources[str(m)]=sha(m)
   staged=Path(item['frames'])/f'{i:06d}.jpg';assert sha(staged)==sha(p);sources[str(staged)]=sha(staged)
  item.update(sequence=take,object_mask_manifest=str(object_manifest),object_mask_manifest_sha256=sha(object_manifest),masks=str(ROOT/'labeled_intercap_v1/masks'/item['name']),depth=str(ROOT/'labeled_intercap_v1/depth'/item['name']),native_frame_numbers=expected['native_frame_numbers'],source_frame_cadence=5,nominal_source_fps=30.,timestamps_seconds=[x/30. for x in expected['native_frame_numbers']],timestamp_basis='native frame numbers / declared30fps; nominal not measured hardware time')
  items.append(item)
 assert len(items)==8 and {x['take'] for x in items}==set(allowed) and sum(len(x['source_paths']) for x in items)==246
 out=ROOT/'labeled_intercap_v1/INPUTS.json';assert not out.exists()
 out.write_text(json.dumps({'schema':'intercap-dev-predicted-only-runtime-v1','GT_fields_included':False,'native_CAD_input':False,'reserve_takes_included':False,'items':items,'source_sha256':sources,'person_mask_policy':'fresh SAM3 person union, original RGB resolution; existing object masks reused, duplicateperson gate before selection'},indent=2))
 print(json.dumps({'input':str(out),'sha256':sha(out),'takes':8,'frames':246,'source_files':len(sources)}))
if __name__=='__main__':main()
