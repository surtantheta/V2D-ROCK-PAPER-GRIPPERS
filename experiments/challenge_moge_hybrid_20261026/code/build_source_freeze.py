#!/usr/bin/env python3
"""Rebuild guards from own source, supplied challenge RGB/masks and local model metadata.
No dataset contents or model weights are copied. Run before launch; never mutate a live freeze.
"""
import argparse,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
 sources={str(p):sha(p) for directory in ['code','frozen_main_code'] for p in sorted((R/directory).iterdir()) if p.is_file() and p.suffix in ['.py','.sh']}
 for p in [R/'INPUTS.json',R/'MODEL_EXPECTED.json',R/'MODEL_SOURCE_RECEIPT.json']:sources[str(p)]=sha(p)
 for p in sorted((R/'work/masks').glob('episode_*/MASKS.json')):sources[str(p)]=sha(p)
 spec=json.loads((R/'INPUTS.json').read_text());sources.update(spec['source_sha256'])
 for p in ['/home/tanuj/.conda/envs/sam3d-objects/lib/python3.11/site-packages/moge/model/v1.py','/rhome/tanuj/projects/sam-3d-objects/sam3d_objects/pipeline/depth_models/moge.py','/rhome/tanuj/projects/sam-3d-objects/sam3d_objects/pipeline/depth_models/base.py','/rhome/tanuj/projects/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py']:sources[p]=sha(p)
 data=dict(schema='challenge-MoGe-full-inference-source-v1',source_sha256=sources,stage_shards={'A0':list(range(0,30,2)),'A1':list(range(1,30,2))},GT_inputs=False,S5_knownK_new_treatment=True,S7_MAIN_repair_code_unchanged=True,hash_policy='full source/input sweep worker start+final; caseRGB+mask before/after use; staticcode/manifests postcase; modelweights once atworkerstart')
 a.out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');print('FREEZE',len(sources),sha(a.out),flush=True)
if __name__=='__main__':main()
