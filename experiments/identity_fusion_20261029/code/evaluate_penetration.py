"""Paired full630 hand penetration proxy using the fixed established hand IDs.

The official winding/distance kernel is reused, but the 5840 points differ from
the hosted 512 roles. Fixed MAIN human alignment contributes scale once only.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np

P = Path('/home/tanuj/V2D_challenge')
H = P/'experiments/identity_fusion_20261029'
OLD = P/'experiments/post_feedback_20261026/human_penetration_audit'
BASE = OLD/'code/audit_human_penetration.py'
HAND = OLD/'run_v2_cari4d/HAND_VERTEX_IDS.npy'
PRIOR = HAND.parent/'COMPLETE.json'
KIT = P/'data/assets/submission_kit/v2d_submission_kit'
PEN = KIT/'metric_code/track_1/PEN.py'
PROTOCOL = H/'PEN_PROTOCOL.json'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, j):
    Path(p).write_text(json.dumps(j, indent=2, allow_nan=False)+'\n')


def verify(pins):
    for p, s in pins.items():
        assert sha(p) == s, ('input changed', p)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def self_test(pen):
    import trimesh
    from scipy.spatial.transform import Rotation
    mesh = trimesh.creation.box(extents=[2.,2.,2.])
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
    points = np.array([[0.,0.,0.],[.5,.1,.2],[2.,0.,0.],[0.,0.,1.1]])
    depth = pen._penetration_depth(points,v,f)
    assert np.allclose(depth,[1.,.5,0.,0.],atol=1e-10,rtol=0)
    assert np.allclose(pen._penetration_depth(points,v,f[:,::-1].copy()),depth,atol=1e-10,rtol=0)
    rot = Rotation.from_rotvec([.3,-.2,.1]).as_matrix(); shift=np.array([.7,.8,2.4]);scale=2.3
    physical = scale*(points@rot.T)+shift
    recovered = (physical-shift)@rot/scale
    again = pen._penetration_depth(recovered,v,f)
    assert np.allclose(again,depth,atol=1e-10,rtol=0)
    assert np.allclose(again*scale*.8*100,depth*184,atol=1e-8,rtol=0)
    return dict(known_box_inside_outside=True,winding_orientation_invariant=True,
                inverse_pose_and_scale_once=True,native_GT_geometry=False)


def freeze():
    assert not PROTOCOL.exists()
    prior = json.loads(PRIOR.read_text())
    assert prior['output_sha256'][str(HAND)] == sha(HAND)
    assert prior['source_sha256'][str(P/'models/mhr/mhr_model.pt')] == sha(P/'models/mhr/mhr_model.pt')
    paths = [Path(__file__), BASE, HAND, PRIOR, PEN, KIT/'v2dlb/mesh_budget.py',
        H/'EVAL_PROTOCOL.json',
        P/'models/mhr/mhr_model.pt', P/'work_labeled/LABELED_EPISODES.json',
        OLD/'depth_only_pair_v1/COMPLETE.json', OLD/'depth_only_pair_v1/SUMMARY.json']
    for ds in ['BEHAVE', 'InterCap']:
        paths += [P/f'work_labeled/align_{ds}/TRANSFORMS.json']
    evaluation = json.loads((H/'EVAL_PROTOCOL.json').read_text())
    for ep in range(900, 912):
        for suffix in ['.npz', '_object.ply']:
            p = P/f'work_labeled/submission/episode_{ep:06d}{suffix}'
            assert sha(p) == evaluation['sha256'][str(p)]
            paths.append(p)
    pins = {str(p): sha(p) for p in paths}
    dump(PROTOCOL, dict(schema='paired-candidate-hand-PEN-proxy-v1', cases=12,
        frames_per_arm=630, arms=['MAIN', 'candidate'], hand_points=5840,
        hand_ID_source=str(HAND), mesh_budget_vertices=4096, mesh_budget_faces=4096,
        inside='absolute winding>0.5; official point-triangle distance kernel',
        metric='mean over all fixed hand IDs and frames per sequence, then mean sequences; cm',
        units='local depth times object scale times fixed MAIN GT-human alignment scale times100 once',
        hosted512_roles=False, exact_hosted_metrics=False, native_object_GT_read=False,
        reporting_reserve_used=False, sha256=pins))
    print('PEN_POLICY_FROZEN', len(pins), sha(PROTOCOL), flush=True)


def run(args):
    assert os.environ.get('PBS_JOBID')
    started = time.monotonic(); policy = json.loads(PROTOCOL.read_text()); verify(policy['sha256'])
    root = args.candidate.resolve(); completion = args.completion.resolve()
    receipt = json.loads(completion.read_text())
    assert receipt['GT_used_for_inference'] is False
    verify(receipt['output_sha256'])
    out = args.out.resolve(); assert out.is_relative_to(H) and not out.exists(); out.mkdir(parents=True)
    metadata = json.loads((P/'work_labeled/LABELED_EPISODES.json').read_text())
    assert set(metadata) == {str(e) for e in range(900, 912)}
    candidate_pins = {str(completion): sha(completion)}
    for ep in range(900, 912):
        for suffix in ['.npz', '_object.ply']:
            p = root/f'episode_{ep:06d}{suffix}'; candidate_pins[str(p)] = sha(p)
            assert receipt['output_sha256'][str(p)] == candidate_pins[str(p)]
    verify(candidate_pins)
    dump(out/'CANDIDATE_FROZEN_BEFORE_GT_SCALE.json', dict(sha256=candidate_pins,
        GT_used_for_inference=False, cases=12, frames=630))
    import torch, numba
    torch.set_num_threads(2); torch.set_num_interop_threads(1); numba.set_num_threads(2)
    base = module(BASE, 'identity_pen_decode')
    pen = module(PEN, 'identity_official_pen')
    assert pen._PENETRATION_KERNELS is not None
    tests = self_test(pen); dump(out/'SYNTHETICS.json',tests)
    sys.path.insert(0, str(KIT)); from v2dlb.mesh_budget import budget_mesh
    hands = np.load(HAND); assert len(hands) == len(np.unique(hands)) == 5840
    model = torch.jit.load(str(P/'models/mhr/mhr_model.pt'), map_location='cpu').eval()
    maps = {}
    for ds in ['BEHAVE', 'InterCap']:
        side = json.loads((P/f'work_labeled/align_{ds}/TRANSFORMS.json').read_text())
        assert side['GT_human_only_for_scoring'] and side['single_transform_per_clip']
        assert not side['native_object_GT_fields_accessed']
        maps[ds] = {r['sequence']: float(r['primary_transform']['scale']) for r in side['records']}
    records = []
    for ep in range(900, 912):
        name = f'episode_{ep:06d}'; info = metadata[str(ep)]; n = len(info['frame_ids'])
        baseline = dict(np.load(P/f'work_labeled/submission/{name}.npz', allow_pickle=False))
        candidate = dict(np.load(root/f'{name}.npz', allow_pickle=False))
        for k in ['pose', 'scales', 'shape']:
            assert baseline[k].shape == candidate[k].shape and baseline[k].dtype == candidate[k].dtype
            assert baseline[k].tobytes() == candidate[k].tobytes()
        human, _ = base.decode(model, baseline, torch)
        assert human.shape == (n, 18439, 3)
        alignment_scale = maps[info['dataset']][info['sequence']]; assert alignment_scale > 0
        row = dict(episode=ep, dataset=info['dataset'], sequence=info['sequence'], frames=n, arms={})
        for arm, data, folder in [('MAIN', baseline, P/'work_labeled/submission'), ('candidate', candidate, root)]:
            v, f = budget_mesh(folder/f'{name}_object.ply', 4096, 4096)
            rot = data['object_rotation']; t = data['object_translation']; scale = float(data['object_scale'])
            assert rot.shape == (n,3,3) and t.shape == (n,3) and scale > 0
            assert np.isfinite(rot).all() and np.isfinite(t).all() and np.isfinite(scale)
            assert np.allclose(rot@rot.transpose(0,2,1), np.eye(3), atol=1e-4, rtol=0)
            assert np.allclose(np.linalg.det(rot), 1, atol=1e-4, rtol=0)
            local = np.einsum('tpi,tij->tpj', human[:,hands]-t[:,None], rot)/scale
            depth = pen._penetration_depth(local.reshape(-1,3), v, f).reshape(n,len(hands))
            assert np.isfinite(depth).all() and (depth >= 0).all()
            cm = depth*scale*alignment_scale*100
            all_front = all(bool((scale*(v@r.T)[:,2]+tr[2] > 0).all()) for r,tr in zip(rot,t))
            row['arms'][arm] = dict(aligned_mean_cm=float(cm.mean()),
                max_point_cm=float(cm.max()), frame_means_cm=cm.mean(axis=1).tolist(),
                positive_point_count=int((depth>0).sum()), all_vertices_front=all_front,
                object_scale=scale, MAIN_human_alignment_scale=alignment_scale)
            np.savez_compressed(out/f'{name}_{arm}_PEN.npz', mesh_local_depth_m=depth,
                object_scale=scale, human_alignment_scale=alignment_scale)
            print('PEN_CASE', ep, arm, row['arms'][arm]['aligned_mean_cm'], flush=True)
        records.append(row); dump(out/f'CASE_{ep}.json', row)
    assert sum(r['frames'] for r in records) == 630
    summaries = []
    previous = json.loads((OLD/'depth_only_pair_v1/SUMMARY.json').read_text())
    for ds in ['BEHAVE', 'InterCap']:
        selected = [r for r in records if r['dataset'] == ds]
        old_main = next(r for r in previous['summary'] if r['dataset'] == ds and r['arm'] == 'main')
        for arm in ['MAIN', 'candidate']:
            values = [r['arms'][arm] for r in selected]; mean = float(np.mean([v['aligned_mean_cm'] for v in values]))
            if arm == 'MAIN':
                assert abs(mean-old_main['mean_sequence_penetration_proxy_cm']) < 1e-10
            summaries.append(dict(dataset=ds, arm=arm, cases=len(selected),
                frames=sum(r['frames'] for r in selected), mean_sequence_PEN_proxy_cm=mean,
                max_point_cm=max(v['max_point_cm'] for v in values),
                all_meshes_front=all(v['all_vertices_front'] for v in values)))
    verify(policy['sha256']); verify(candidate_pins)
    output_pins = {str(p): sha(p) for p in out.iterdir() if p.is_file()}
    dump(out/'COMPLETE.json', dict(status='PAIRED_FULL630_PEN_PROXY_COMPLETE',
        policy=policy, records=records, summaries=summaries, cases=12, frames_per_arm=630,
        source_sha256=dict(policy['sha256'], **candidate_pins), output_sha256=output_pins,
        MAIN_human_bitexact=True, native_object_GT_read=False, reporting_reserve_used=False,
        exact_hosted_metrics=False, synthetics=tests,
        CPU_job=os.environ['PBS_JOBID'], elapsed_seconds=time.monotonic()-started))
    print('PEN_COMPLETE', out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--candidate', type=Path); ap.add_argument('--completion', type=Path); ap.add_argument('--out', type=Path)
    args = ap.parse_args(); assert args.freeze != args.run
    if args.freeze: freeze()
    else:
        assert all(x is not None for x in [args.candidate, args.completion, args.out]); run(args)


if __name__ == '__main__':
    main()
