#!/usr/bin/env python3
"""Validated downstream chain: budget, fixed-mesh tracking, paired fits and CPU scorer."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
CARI=Path('/home/tanuj/projects/CARI4D')
PY='/home/tanuj/.conda/envs/cari4d/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(argv,cwd=ROOT,env=None):
    print('launch',argv,flush=True);subprocess.run(argv,cwd=str(cwd),env=env,check=True)
def main():
    from model_source_guard import verify_model_group
    model_provenance=verify_model_group('FoundationPose')
    runtime=ROOT/'labeled_predictions_v1';request=ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json';refs=ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json'
    codes=[Path(__file__),ROOT/'code/budget_labeled_proposals.py',ROOT/'code/labeled_track_fp_snapshot_v1.py',ROOT/'code/sam3d_mesh_contract_snapshot_v1.py',ROOT/'code/labeled_refine_compare.py',ROOT/'code/silhouette_pilot_v6.py',ROOT/'code/labeled_object_scorer.py',CARI/'estimater.py',request,refs]
    sources={str(p):sha(p) for p in codes};(runtime/'COMPARISON_SOURCE_RECEIPT.json').write_text(json.dumps(sources,indent=2))
    run(['/home/tanuj/.conda/envs/v2d/bin/python',str(ROOT/'code/budget_labeled_proposals.py')])
    inputs=json.loads((runtime/'RUN_INPUTS.json').read_text());assets={x['sequence']:x for x in json.loads((runtime/'BUDGETED.json').read_text())['records']}
    env=os.environ.copy()
    if torch.cuda.get_device_capability()==(7,5):env['PYTHONPATH']=str(ROOT/'deps/sm75')
    for item in inputs['items']:
        if item['dataset']!='BEHAVE' or item['name'] not in assets:continue
        run([PY,str(ROOT/'code/labeled_track_fp_snapshot_v1.py'),'--frames',item['frames'],'--masks',item['masks'],'--human',str(runtime/'depth'/item['name']),'--object',assets[item['name']]['directory'],'--out',str(runtime/'tracks'/item['name'])],cwd=CARI,env=env)
    run([PY,str(ROOT/'code/labeled_refine_compare.py')],env=env)
    comparison=runtime/'comparison'
    arm_names=['foundationpose','plain','person_band','person_band_smooth1','person_band_high_confidence','person_band_high_confidence_smooth1']
    coverage={name:{seq['sequence']:sorted(x['video_frame_index'] for x in seq['frames']) for seq in json.loads((comparison/f'{name}_PREDICTIONS.json').read_text())['sequences']} for name in arm_names}
    assert all(coverage[name]==coverage[arm_names[0]] for name in arm_names),'paired arms have different coverage'
    expected={x['name']:len(x['source_paths']) for x in inputs['items'] if x['dataset']=='BEHAVE'}
    coverage_report={'expected_frames':sum(expected.values()),'actual_frames':sum(len(x) for x in coverage[arm_names[0]].values()),'failed_sequences':sorted(set(expected)-set(coverage[arm_names[0]])),'identical_coverage_across_arms':True}
    assert coverage_report['actual_frames']>0,'no predictions; refuse completion'
    (comparison/'COVERAGE.json').write_text(json.dumps(coverage_report,indent=2))
    result=[]
    for arm in ['foundationpose','plain','person_band','person_band_smooth1','person_band_high_confidence','person_band_high_confidence_smooth1']:
        pred=runtime/'comparison'/f'{arm}_PREDICTIONS.json';out=runtime/'comparison'/f'{arm}_SCORES.json'
        # The scorer discloses any missing sequence; all generated trajectories retain all96frames.
        run([PY,str(ROOT/'code/labeled_object_scorer.py'),'--prediction',str(pred),'--requests',str(request),'--references',str(refs),'--out',str(out),'--allow-partial'])
        result.append({'arm':arm,'prediction':str(pred),'prediction_sha256':sha(pred),'scores':str(out),'scores_sha256':sha(out)})
    (runtime/'LABELED_COMPARISON_COMPLETE.json').write_text(json.dumps({'arms':result,'source_sha256':sources,'GT_used_by_inference':False,'model_provenance':model_provenance,'scoring':'raw camera-frame metric proxy; no human-derived Sim3; generated/native origins not certified corresponding','all_arms_retained':True,'coverage':coverage_report},indent=2));print('completed labeled paired comparison',flush=True)
if __name__=='__main__':main()
