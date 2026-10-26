#!/usr/bin/env python3
"""Fresh person masks only for eight frozen InterCap development takes; no GT."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import Sam3Model,Sam3Processor
from model_source_guard import verify_model_group
ROOT=Path(__file__).resolve().parents[1]
REV='3c879f39826c281e95690f02c7821c4de09afae7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 manifest=ROOT/'labeled_intercap_v1/INPUTS.json';data=json.loads(manifest.read_text());assert data['GT_fields_included'] is False and data['reserve_takes_included'] is False
 torch.set_num_threads(4);provenance=verify_model_group('SAM3')
 model=Sam3Model.from_pretrained('facebook/sam3',revision=REV,local_files_only=True).to('cuda').eval();processor=Sam3Processor.from_pretrained('facebook/sam3',revision=REV,local_files_only=True)
 source={str(p):sha(p) for p in [Path(__file__),manifest,ROOT/'code/model_source_guard.py',ROOT/'labeled_predictions_v1/MODEL_SOURCE_RECEIPT.json']};records=[]
 for item in data['items']:
  assert item['dataset']=='InterCap';old_path=Path(item['object_mask_manifest']);assert sha(old_path)==item['object_mask_manifest_sha256'];old=json.loads(old_path.read_text());dest=Path(item['masks']);dest.mkdir(parents=True,exist_ok=False);rows=[];rejected=[];start=time.monotonic()
  for i,(path,row0) in enumerate(zip(item['source_paths'],old['rows'])):
   assert sha(path)==item['source_sha256'][i]==row0['source_rgb_sha256'];assert sha(row0['mask'])==row0['mask_sha256'];im=Image.open(path).convert('RGB')
   inp=processor(images=im,text='person',return_tensors='pt').to('cuda')
   with torch.inference_mode():prediction=model(**inp)
   result=processor.post_process_instance_segmentation(prediction,threshold=.3,mask_threshold=.5,target_sizes=[(im.height,im.width)])[0];people=result.get('masks');person=np.zeros((im.height,im.width),bool) if people is None or not len(people) else people.cpu().numpy().astype(bool).any(0)
   pp=dest/f'{i:06d}_person.png';Image.fromarray(person.astype(np.uint8)*255).save(pp);obj=np.asarray(Image.open(row0['mask']))>127;assert obj.shape==person.shape
   inter=int((obj&person).sum());overlap=inter/max(int((obj|person).sum()),1);inside=inter/max(int(obj.sum()),1);bad=bool(row0['selected'] and overlap>=.8 and inside>=.95)
   row=dict(row0);row.update(person_mask=str(pp),person_mask_sha256=sha(pp),selected=bool(row0['selected'] and not bad),person_duplicate_gate={'object_person_iou':overlap,'object_pixels_inside_person_fraction':inside,'rejected':bad,'threshold_iou':.8,'threshold_object_inside_person':.95})
   rows.append(row)
   if bad:rejected.append(i)
   if i%8==0:print(item['name'],i,'/',len(item['source_paths']),'person_pixels',int(person.sum()),'duplicate',bad,flush=True)
  report={'schema':'intercap-dev-object-reuse-fresh-person-masks-v1','dataset':'InterCap','name':item['name'],'sequence':item['take'],'frames':len(rows),'rows':rows,'rejected_person_duplicates':rejected,'seconds':time.monotonic()-start,'object_parent_manifest':str(old_path),'object_parent_manifest_sha256':sha(old_path),'source_sha256':source,'model':provenance,'GT_inputs_provided':False,'native_CAD_input':False,'original_RGB_resolution':True}
  mp=dest/'MASKS.json';mp.write_text(json.dumps(report,indent=2,allow_nan=False));records.append({'sequence':item['take'],'name':item['name'],'mask_manifest':str(mp),'mask_manifest_sha256':sha(mp),'frames':len(rows),'rejected':rejected});print('complete',item['name'],'seconds',report['seconds'],flush=True)
 receipt={'schema':'intercap-dev-person-masks-complete-v1','records':records,'source_sha256':source,'model':provenance,'takes':len(records),'frames':sum(x['frames'] for x in records),'GT_inputs_provided':False,'reserve_takes_included':False,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()};assert receipt['takes']==8 and receipt['frames']==246
 complete=ROOT/'labeled_intercap_v1/PERSON_MASKS_COMPLETE.json';complete.write_text(json.dumps(receipt,indent=2))
 data['person_masks_complete_receipt']=str(complete);data['person_masks_complete_receipt_sha256']=sha(complete);data['person_masks_are_fresh']=True
 (ROOT/'labeled_intercap_v1/RUN_INPUTS.json').write_text(json.dumps(data,indent=2));print('all8takes246personmasks complete',flush=True)
if __name__=='__main__':main()
