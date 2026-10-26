#!/usr/bin/env python3
"""CPU-only paired evaluation of predicted-human scale, with explicit no-anchor fallback."""
import copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/home/tanuj/V2D_challenge/experiments/offline_opt_20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    runtime=ROOT/'labeled_predictions_v1';anchor=runtime/'human_scale/RESULT.json';scale=json.loads(anchor.read_text());lookup={x['sequence']:x for x in scale['records']};out=runtime/'human_anchored_comparison';out.mkdir(exist_ok=False)
    request=ROOT/'benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json';refs=ROOT/'benchmarks/labeled_dev_v2/scorer/BEHAVE_LABEL_REFERENCES_PINNED.json';scorer=ROOT/'code/labeled_object_scorer.py';report=[]
    for original in sorted((runtime/'comparison').glob('*_PREDICTIONS.json')):
        pred=json.loads(original.read_text());notes=[]
        for seq in pred['sequences']:
            entry=lookup.get(seq['sequence'],{});alpha=entry.get('alpha');fallback=alpha is None
            if fallback:alpha=1.
            assert alpha>0
            seq['object_scale']*=alpha
            for frame in seq['frames']:frame['t_obj2cam_m']=[alpha*x for x in frame['t_obj2cam_m']]
            seq['source']['human_anchor']={'alpha':alpha,'no_anchor_fallback_identity':fallback,'anchor_result_sha256':sha(anchor),'original_prediction_sha256':sha(original),'same_scale_applied_to_geometry_and_translation':True,'GT_used_to_estimate_alpha':False}
            notes.append({'sequence':seq['sequence'],'alpha':alpha,'no_anchor_fallback_identity':fallback})
        dest=out/original.name;dest.write_text(json.dumps(pred,indent=2,allow_nan=False));scores=out/original.name.replace('_PREDICTIONS.json','_SCORES.json')
        subprocess.run(['/home/tanuj/.conda/envs/cari4d/bin/python',str(scorer),'--prediction',str(dest),'--requests',str(request),'--references',str(refs),'--out',str(scores),'--allow-partial'],check=True)
        report.append({'arm':original.stem.replace('_PREDICTIONS',''),'prediction':str(dest),'scores':str(scores),'scale_notes':notes,'prediction_sha256':sha(dest),'scores_sha256':sha(scores),'raw_scores':str(original.with_name(original.name.replace('_PREDICTIONS.json','_SCORES.json')))})
    assert report
    (out/'RESULT.json').write_text(json.dumps({'schema':'predicted-human-scale-paired-comparison-v1','records':report,'source_sha256':{str(x):sha(x) for x in [Path(__file__),anchor,scorer,request,refs]},'GT_used_by_scale_estimator':False,'same_coverage_as_unanchored':True,'human_alignment':'noGTderivedSim3; RGB-predictedhumanprovidesconstantmetricprior','all_arms_retained':True,'image_projection_invariant':True,'acceleration_magnitude_not_invariant_under_scale':True},indent=2));print('complete predicted-human-scale paired scoring',flush=True)
if __name__=='__main__':main()
