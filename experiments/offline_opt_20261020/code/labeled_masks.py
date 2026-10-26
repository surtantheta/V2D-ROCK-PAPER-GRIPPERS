#!/usr/bin/env python3
"""Predicted object/person masks for frozen RGB-only engineering requests."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import Sam3Model,Sam3Processor

ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
HELPER=Path('/home/tanuj/UniCon3Rpp-push-b3c6624/tools/wild_video_masks.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=ROOT/'labeled_predictions_v1/RUN_INPUTS.json';a=json.loads(manifest.read_text());spec=importlib.util.spec_from_file_location('mask_helper',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert not a['GT_fields_included'];torch.set_num_threads(4)
    model=Sam3Model.from_pretrained('facebook/sam3',revision=helper.REV,local_files_only=True).to('cuda').eval()
    proc=Sam3Processor.from_pretrained('facebook/sam3',revision=helper.REV,local_files_only=True)
    source={'inputs_sha256':sha(manifest),'code_sha256':sha(__file__),'selector_sha256':sha(HELPER),'model':'facebook/sam3','revision':helper.REV,'dtype':'float32','external_GT_inputs':False}
    for item in a['items']:
        dest=Path(item['masks']);dest.mkdir(parents=True,exist_ok=False);previous=None;last_hit=-10**9;rows=[];t0=time.time()
        for i,path in enumerate(item['source_paths']):
            assert sha(path)==item['source_sha256'][i],path
            im=Image.open(path).convert('RGB');chosen=None;used=None;person_path=None
            prompts=[item['prompt'],'the object the person is holding or carrying']
            for prompt in prompts:
                inp=proc(images=im,text=prompt,return_tensors='pt').to('cuda')
                with torch.inference_mode():out=model(**inp)
                res=proc.post_process_instance_segmentation(out,threshold=.3,mask_threshold=.5,target_sizes=[(im.height,im.width)])[0]
                mm,ss=res.get('masks'),res.get('scores')
                if mm is not None and len(mm):
                    chosen=helper.pick(helper.candidates([x.cpu() for x in mm],[float(x) for x in ss],(im.height,im.width)),previous,i-last_hit)
                if chosen is not None:used=prompt;break
            if chosen is None:mask=np.zeros((im.height,im.width),bool);score=area=0.;centroid=None
            else:mask=chosen['mask'];score=chosen['score'];area=chosen['area'];centroid=chosen['centroid'];previous=centroid;last_hit=i
            mp=dest/f'{i:06d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(mp)
            if item['dataset']=='BEHAVE':
                inp=proc(images=im,text='person',return_tensors='pt').to('cuda')
                with torch.inference_mode():out=model(**inp)
                res=proc.post_process_instance_segmentation(out,threshold=.3,mask_threshold=.5,target_sizes=[(im.height,im.width)])[0]
                pp=res.get('masks');person=np.zeros((im.height,im.width),bool) if pp is None or not len(pp) else pp.cpu().numpy().astype(bool).any(0)
                person_path=dest/f'{i:06d}_person.png';Image.fromarray(person.astype(np.uint8)*255).save(person_path)
            rows.append({'frame':f'{i:06d}.jpg','source_rgb':path,'source_rgb_sha256':item['source_sha256'][i],'mask':str(mp),'mask_sha256':sha(mp),'selected':bool(mask.any()),'score':float(score),'area':float(area),'centroid':centroid,'prompt':used,'person_mask':str(person_path) if person_path else None,'person_mask_sha256':sha(person_path) if person_path else None,'hw':[im.height,im.width]})
            if i%16==0:print(item['name'],i,'/',len(item['source_paths']),'object area',area,'score',score,flush=True)
        report={'schema':'labeled-predicted-masks-v1','dataset':item['dataset'],'name':item['name'],'frames':len(rows),'empty':sum(not x['selected'] for x in rows),'rows':rows,'source':source,'seconds':time.time()-t0,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        (dest/'MASKS.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('complete',item['name'],'empty',report['empty'],'seconds',report['seconds'],flush=True)
    (ROOT/'labeled_predictions_v1/MASKS_COMPLETE.json').write_text(json.dumps({'source':source,'sequences':len(a['items']),'frames':sum(len(x['source_paths']) for x in a['items'])},indent=2))
if __name__=='__main__':main()
