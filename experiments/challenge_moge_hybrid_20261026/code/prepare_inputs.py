#!/usr/bin/env python3
"""Stage challenge inputs with symlinks; source/main/old experiments read-only."""
import hashlib,json
from pathlib import Path
from PIL import Image
P=Path('/home/tanuj/V2D_challenge');R=Path(__file__).resolve().parents[1];W=R/'work'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 W.mkdir(exist_ok=True);sources={};cases=[];choice=json.loads((P/'work/MASK_CHOICE.json').read_text());sources[str(P/'work/MASK_CHOICE.json')]=sha(P/'work/MASK_CHOICE.json')
 for ep in range(30):
  e=f'episode_{ep:06d}';frames=P/'work/frames'/e;maskdir=P/'work'/('masks_v2' if choice.get(str(ep),'v1')=='v2' else 'masks')/e;files=sorted(frames.glob('*.jpg'));fr=json.loads((frames/'FRAMES.json').read_text());assert len(files)==fr['frames'] and [p.stem for p in files]==[f'{i:06d}' for i in range(len(files))]
  rows=json.loads((maskdir/'MASKS.json').read_text())['rows'];assert len(rows)==len(files);wh=list(Image.open(files[0]).size);assert abs(wh[0]/wh[1]-4/3)<1e-8
  for kind,src in [('frames',frames)]:
   dest=W/kind/e;dest.parent.mkdir(exist_ok=True);dest.symlink_to(src,target_is_directory=True)
  md=W/'masks'/e;md.mkdir(parents=True);(md/'MASKS.json').write_text(json.dumps(dict(rows=rows,source_manifest=str(maskdir/'MASKS.json'),source_manifest_sha256=sha(maskdir/'MASKS.json')),indent=1)+'\n')
  for p in [frames/'FRAMES.json',maskdir/'MASKS.json',*files,*[Path(x['mask']) for x in rows]]:sources[str(p)]=sha(p)
  cases.append(dict(episode=ep,frames=len(files),source_wh=wh,source_frames=str(frames),source_masks=str(maskdir),mask_manifest=str(md/'MASKS.json'),stage_shard='A0' if ep%2==0 else 'A1',frame_rgb_sha256=[sources[str(x)] for x in files]))
 assert len(cases)==30 and sum(x['frames'] for x in cases)==16563
 spec=dict(schema='challenge-full-MoGe-hybrid-input-v1',cases=cases,frames=16563,scored_frames=9877,GT_inputs=False,source_sha256=sources,source_policy='samecurrentMAINRGBandMASK_CHOICE; noGT/providerselector; all30',S5_treatment='layoutFalse asfullICrecipe; known supplied-pointmap normalizedK override newlyenabled, explicitdifference fromoldfullMoGedev')
 (R/'INPUTS.json').write_text(json.dumps(spec,indent=2)+'\n');print('INPUTS_READY',len(cases),spec['frames'],len(sources),flush=True)
if __name__=='__main__':main()
