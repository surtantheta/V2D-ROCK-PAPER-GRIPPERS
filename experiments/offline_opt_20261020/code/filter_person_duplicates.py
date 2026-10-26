#!/usr/bin/env python3
"""Image-only rejection of object masks duplicating the predicted person."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidates(rows):
    for threshold in [.8,.3]:
        eligible=sorted([(r['area'],i) for i,r in enumerate(rows) if r['selected'] and r['score']>=threshold],reverse=True)
        if eligible:break
    result=[]
    for _,i in eligible:
        if all(abs(i-j)>=30 for j in result):result.append(i)
        if len(result)==3:break
    return result

def main():
    src=ROOT/'labeled_predictions_v1/RUN_INPUTS.json';data=json.loads(src.read_text());out=ROOT/'labeled_predictions_v2';out.mkdir(exist_ok=False);audit=[]
    for item in data['items']:
        if item['dataset']!='BEHAVE':continue
        old=Path(item['masks'])/'MASKS.json';a=json.loads(old.read_text());old_indices=candidates(a['rows']);rejected=[]
        dest=out/'masks'/item['name'];dest.mkdir(parents=True)
        for i,row in enumerate(a['rows']):
            m=np.asarray(Image.open(row['mask']))>127;person=np.asarray(Image.open(row['person_mask']))>127
            inter=int((m&person).sum());union=int((m|person).sum());iou=inter/max(union,1);inside=inter/max(int(m.sum()),1)
            bad=row['selected'] and iou>=.8 and inside>=.95
            row['person_duplicate_gate']={'object_person_iou':iou,'object_pixels_inside_person_fraction':inside,'rejected':bool(bad),'threshold_iou':.8,'threshold_object_inside_person':.95}
            if bad:row['selected']=False;rejected.append(i)
        a['parent_manifest_sha256']=sha(old);a['semantic_filter_code_sha256']=sha(__file__);a['image_only_filter']='reject object/person mask IoU>=.8 AND >=.95 object pixels inside person';a['rejected_person_duplicates']=rejected;a['empty_or_rejected']=sum(not x['selected'] for x in a['rows'])
        (dest/'MASKS.json').write_text(json.dumps(a,indent=2));item['masks']=str(dest);new_indices=candidates(a['rows']);audit.append({'sequence':item['name'],'rejected_frames':rejected,'rejected_count':len(rejected),'old_candidates':old_indices,'new_candidates':new_indices,'generation_candidates_changed':old_indices!=new_indices,'source_manifest':str(old),'source_manifest_sha256':sha(old)})
        print(item['name'],'rejected',len(rejected),'candidates',old_indices,'->',new_indices,flush=True)
    data['parent_RUN_INPUTS_sha256']=sha(src);data['semantic_filter_code_sha256']=sha(__file__);data['cohort_frames_changed']=False
    (out/'RUN_INPUTS.json').write_text(json.dumps(data,indent=2));(out/'SEMANTIC_MASK_AUDIT.json').write_text(json.dumps({'records':audit,'source_sha256':{str(src):sha(src),str(Path(__file__)):sha(__file__)},'GT_used':False,'original_results_preserved':True,'thresholds_chosen_from_near_identical_person_failure_without_GT':True},indent=2))
if __name__=='__main__':main()
