#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 hf=Path('/home/tanuj/.cache/huggingface/hub');sam3=hf/'models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7';root=Path('/rhome/tanuj/projects/sam-3d-objects/checkpoints/hf');pipeline=root/'pipeline.yaml'
 objects=[pipeline]+[root/x for x in re.findall(r'^\w+_path: ([^\s]+\.(?:ckpt|yaml))$',pipeline.read_text(),re.M)]
 moge=hf/'models--Ruicheng--moge-vitl';ref=moge/'refs/main';revision=ref.read_text().strip();objects.extend([ref,moge/'snapshots'/revision/'model.pt'])
 groups={'SAM3':sorted(x for x in sam3.iterdir() if x.is_file()),'SAM3D_Objects_and_MoGe':objects}
 out={}
 for group,paths in groups.items():
  out[group]={}
  for p in paths:
   out[group][str(p)]={'sha256':digest(p),'bytes':p.stat().st_size,'resolved_path':str(p.resolve())};print(group,p.name,p.stat().st_size,flush=True)
 (ROOT/'labeled_predictions_v1/MODEL_SOURCE_RECEIPT.json').write_text(json.dumps({'schema':'labeled-model-source-receipt-v1','groups':out,'MoGe_HF_main_revision_at_freeze':revision},indent=2))
if __name__=='__main__':main()
