#!/usr/bin/env python3
"""Development-only leave-one-object-out global camera-scale diagnostic."""
import importlib.util,json
from pathlib import Path
import numpy as np
from cached_stage1_coordinate_audit_v2 import ROOT,CACHE,SEQUENCES,sha,cd

def main():
    output=ROOT/'runs/cached_scale_LOO_v1';output.mkdir()
    parent=ROOT/'runs/cached_stage1_coordinate_audit_v2/RESULT.json';base=json.loads(parent.read_text());manifest=CACHE/'gate3_manifest.json';meta={x['sample_id']:x for x in json.loads(manifest.read_text())['samples']}
    helper=Path('/home/tanuj/projects/UniCon3Rpp_M5_src/tools/sam3d_behave_gate3.py');spec=importlib.util.spec_from_file_location('coarse_helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    records=[]
    for held in SEQUENCES:
        training=[x for x in base['records'] if x['sequence']!=held]
        # Uses labeled OTHER objects only; the test object's GT is scorer-only.
        alpha=float(np.median([1/x['predicted_to_GT_median_depth_ratio'] for x in training]))
        for row in base['records']:
            if row['sequence']!=held:continue
            s=meta[row['sample_id']];p=CACHE/'predictions'/f'{row["sample_id"]}__seed{row["seed"]}.npz';pred=np.load(p)
            cloud=pred['voxel'].astype(float);rng=np.random.default_rng(20261020)
            if len(cloud)>4000:cloud=cloud[rng.choice(len(cloud),4000,replace=False)]
            flip=np.diag([-1.,-1.,1.]);R=flip@h.rotation_from_quaternion_wxyz(pred['rotation']).T;t=flip@pred['translation'];points=(cloud*pred['scale'])@R.T+t
            template=h.load_template(s,count=4000);gt=template@np.array(s['gt_rotation_cam']).T+np.array(s['gt_translation_cam'])
            records.append({'held_out_object_sequence':held,'training_objects':[x for x in SEQUENCES if x!=held],'sample_id':s['sample_id'],'seed':row['seed'],'alpha_from_other_objects':alpha,'raw_coarse_point_CD_sum_cm':cd(points,gt),'LOO_scaled_coarse_point_CD_sum_cm':cd(alpha*points,gt),'primary_inference_changed':False})
    summary={s:{'alpha_from_other_objects':next(x['alpha_from_other_objects'] for x in records if x['held_out_object_sequence']==s),'raw_median_CD_sum_cm':float(np.median([x['raw_coarse_point_CD_sum_cm'] for x in records if x['held_out_object_sequence']==s])),'LOO_scaled_median_CD_sum_cm':float(np.median([x['LOO_scaled_coarse_point_CD_sum_cm'] for x in records if x['held_out_object_sequence']==s]))} for s in SEQUENCES}
    report={'schema':'coarse-voxel-oracle-mask-scale-LOO-diagnostic-v1','summary':summary,'records':records,'source_sha256':{str(x):sha(x) for x in [Path(__file__),parent,helper,manifest]},'scope':'48cachedstage1occupancypoints withoracle masks; scalertrainedonGTdepthofother3objects; testGTusedonlybyscorer. No rotation/translation/shapeGTregistration. Uniform camera-origin scale is learneddevelopmentcalibration, not an RGB-onlyproduction estimate. This is NOT decodedmeshsurfaceaccuracy or ancestry-cleangeneralization. Primaryfreshinference remains unchanged.'}
    (output/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
