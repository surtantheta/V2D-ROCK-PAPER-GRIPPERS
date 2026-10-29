"""AST allowlist for cohort transport plus existing 12-case coordinate joins."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/home/tanuj/V2D_challenge/experiments/pi3x_kaggle_20261029')
spec=importlib.util.spec_from_file_location('cpu_transport_under_review',ROOT/'code/run_cpu_adapter_v2.py')
adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)


def tree(s):return ast.parse(s).body[0]
def canon(n):return ast.dump(n,include_attributes=False)


class Allowlist(ast.NodeTransformer):
    def __init__(self,kind):self.kind=kind;self.changes=[]
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='metadata':
                self.changes.append('metadata_path');node.value=ast.parse('json.loads(CHALLENGE_METADATA.read_text())',mode='eval').body
            elif name=='controls' and self.kind=='calibration':
                assert isinstance(node.value,ast.Call) and node.value.func.id=='control_diagnostics'
                self.changes.append('omit_diagnostic_controls');node.value=ast.List(elts=[],ctx=ast.Load())
        return node
    def visit_If(self,node):
        if self.kind=='calibration' and canon(node.test)==canon(ast.parse("case['dataset']=='challenge'",mode='eval').body):
            assert len(node.body)==1 and isinstance(node.body[0],ast.Continue)
            self.changes.append('accept_challenge');return None
        if self.kind=='optimizer' and canon(node.test)==canon(ast.parse('len(records)==12',mode='eval').body):
            assert len(node.body)==1 and isinstance(node.body[0],ast.Assert)
            assert canon(node.body[0].test)==canon(ast.parse('total==630',mode='eval').body)
            self.changes.append('cohort_completeness')
            return ast.parse("assert len(records)==30 and total==sum(c['total_frames'] for c in j['cases'])").body[0]
        return self.generic_visit(node)
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        if self.kind=='calibration' and canon(node)==canon(ast.parse('len(records)==36',mode='eval').body):
            self.changes.append('cohort_completeness');return ast.parse('len(records)==30',mode='eval').body
        return node
    def visit_Subscript(self,node):
        if self.kind=='optimizer' and canon(node)==canon(ast.parse("case['calibration']['frames'][p]['calibrated_body_residual_cm']",mode='eval').body):
            self.changes.append('optional_calibration_residual')
            return ast.parse("case['calibration']['frames'][p].get('calibrated_body_residual_cm')",mode='eval').body
        return self.generic_visit(node)


def main():
    diag,base,surface=adapter.modules()
    checks=[]
    for kind,module,name,replacements in [('calibration',diag,'predicted_stage',adapter.CALIBRATION_TRANSPORT),('optimizer',base,'run',adapter.OPTIMIZER_TRANSPORT)]:
        original=inspect.getsource(getattr(module,name));expected=Allowlist(kind)
        allowed=expected.visit(tree(original));actual=original
        for before,after in replacements:
            assert actual.count(before)==1;actual=actual.replace(before,after)
        assert canon(allowed)==canon(tree(actual)),('Unapproved AST change',kind)
        assert len(expected.changes)==(4 if kind=='calibration' else 3)
        checks.append(dict(function=name,approved_changes=expected.changes,
            all_other_AST_exact=True,original_source_sha256=hashlib.sha256(original.encode()).hexdigest()))
    prev=adapter.PREV;manifest=json.loads((prev/'PILOT_INPUTS.json').read_text())
    metadata=json.loads((adapter.R/'work_labeled/LABELED_EPISODES.json').read_text())
    cal=json.loads(Path('/rhome/tanuj/v2d_fusion_20261028/diagnostics/body_surface_calibration_v3/CALIBRATION_AND_CLOUDS_FROZEN.json').read_text())
    count=0
    for case in manifest['cases']:
        if case['dataset']=='challenge':continue
        ep=case['episode'];record=next(r for r in cal['records'] if r['episode']==ep and r['model']=='pi3x')
        g=np.load(f'/rhome/tanuj/v2d_fusion_20261028/runs/pilot_v1/pi3x/episode_{ep:06d}_normal/GEOMETRY.npz')
        assert len(record['frames'])==16 and record['calibration'] is not None
        for pos,fr in enumerate(case['frames']):
            idx=fr['local_index'];m=metadata[str(ep)]
            assert g['frame_local_indices'][pos]==idx and g['native_frame_ids'][pos]==m['frame_ids'][idx]==fr['native_frame_id']
            assert abs(g['time_seconds'][pos]-m['times'][idx])<1e-8
        count+=1
    assert count==12
    import torch
    assert not torch.cuda.is_initialized()
    receipt=dict(status='AST_ALLOWLIST_AND_EXISTING12_INPUT_CONTRACT_PASS',checks=checks,
        existing_cases=12,existing_native_time_joins=192,CUDA_initialized=False,
        new_model_inference=False,new_calibration_fit=False,new_GT_values_read=False,
        guard_change='Optional residual lookup and absent-calibration empty-cloud behavior explicitly declared; not claimed unchanged',
        original_control_source_guards=adapter.EXPECTED_CONTROL)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
