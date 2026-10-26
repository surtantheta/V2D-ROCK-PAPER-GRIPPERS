#!/usr/bin/env python3
"""CPU source/coordinate validation of predicted-only placement assets; no labels."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from prepare_hunyuan_predicted_placement import ROOT,sha,valid_mesh,rotation

root=ROOT/'labeled_hunyuan_placement_v1'
runtime=root/'RUNTIME.json';run=json.loads(runtime.read_text());budget=json.loads((root/'BUDGETED.json').read_text());baselines=json.loads((root/'SAM3D_MATCHED_BUDGETED.json').read_text());sam_by_seq={r['sequence']:r for r in baselines['records']}
allowed=[ROOT/'code',ROOT/'labeled_predictions_v1',ROOT/'labeled_predictions_v2',ROOT/'benchmarks/labeled_dev_v2/rgb',ROOT.parents[1]/'data/assets/submission_kit/v2d_submission_kit/v2dlb']
if run['GT_inputs_provided'] is not False or run['native_CAD_input'] is not False or run['scorer_transform_used'] is not False:raise ValueError('predicted-only flags required')
for path,h in run['source_sha256'].items():
 p=Path(path).resolve()
 if not any(p.is_relative_to(a.resolve()) for a in allowed):raise ValueError('source outside permitted predicted runtime/code roots')
 if sha(p)!=h:raise ValueError('input/code source hash changed')
records=[]
for rec in run['records']:
 seq=rec['sequence'];d=Path(rec['directory']);key=json.loads((d/'keyframe.json').read_text());sam=json.loads(Path(rec['source_SAM_proposal']).read_text());hy=json.loads(Path(rec['source_H_proposal']).read_text());hy_npz=Path(rec['source_H_proposal']).parent/'shape_normalized_4096.npz'
 src=np.load(hy_npz);h=np.asarray(src['vertices'],np.float64);faces=np.asarray(src['faces'],np.int64);mesh=trimesh.load(d/'object_mesh.ply',force='mesh',process=False);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
 count=valid_mesh(v,f)
 fit=rec['generated_to_generated_fit'];r=rotation(fit['R']);t=np.asarray(fit['t']);scale=fit['scale'];expected=scale*h@r.T+t
 error=float(abs(v-expected).max())
 if error>1e-6 or not np.array_equal(f,faces):raise ValueError('baked mesh file differs from intended Sim3')
 for k in ['R_obj2cam','t_obj2cam_m','K_predicted']:
  if not np.array_equal(np.array(key[k]),np.array(sam[k])):raise ValueError('inherited camera pose/K changed')
 if key['keyframe']!=sam['local_frame_index'] or rec['keyframe']!=key['keyframe']:raise ValueError('registration keyframe changed')
 source_rgb=np.asarray(Image.open(rec['source_RGB']).convert('RGB'));source_mask=np.asarray(Image.open(rec['source_mask']))>127;rgba=np.concatenate([source_rgb,(source_mask.astype(np.uint8)*255)[...,None]],axis=-1)
 if not np.array_equal(np.asarray(Image.open(hy['input_rgba'])),rgba):raise ValueError('H conditioning pixels differ from target SAM RGB/mask')
 baseline=Path(sam_by_seq[seq]['directory']);baseline_key=json.loads((baseline/'keyframe.json').read_text())
 if baseline_key['keyframe']!=key['keyframe'] or baseline_key['mesh_sha256']!=sha(baseline/'object_mesh.ply'):raise ValueError('matched baseline frame/mesh mismatch')
 if key['mesh_sha256']!=sha(d/'object_mesh.ply') or sha(hy_npz)!=hy['output_sha256'][str(hy_npz)]:raise ValueError('output/input geometry hash mismatch')
 records.append({'sequence':seq,'keyframe':key['keyframe'],'baked_mesh_max_roundtrip_error_nominal_m':error,'proper_alignment_determinant':float(np.linalg.det(r)),'positive_scale':scale,'positive_faces':count,'same_camera_pose_and_K':True,'same_RGBA_conditioning_pixels':True,'matched_SAM_baseline_same_frame':True})
if len(records)!=4 or len({r['sequence'] for r in records})!=4:raise ValueError('four unique cases required')
report={'schema':'predicted-only-placement-preflight-v1','GT_inputs_read':False,'source_hashes_checked':len(run['source_sha256']),'source_paths_within_permitted_runtime_roots':True,'records':records,'validator_sha256':sha(__file__)}
receipt=root/'PREFLIGHT.json';receipt.write_text(json.dumps(report,indent=2))
files=[runtime,root/'BUDGETED.json',root/'SAM3D_MATCHED_BUDGETED.json',root/'KEYFRAME_PREDICTIONS.json',receipt,Path(__file__)]
files += [p for label in ['objects','sam3d_matched'] for p in (root/label).rglob('*') if p.is_file()]
guard=dict(run['source_sha256']);guard.update({str(p.resolve()):sha(p) for p in files})
(root/'PREPARED_ASSET_GUARD.json').write_text(json.dumps({'schema':'predicted-only-placement-launch-source-output-guard-v1','source_sha256':guard,'GT_or_scorer_sources_present':False},indent=2))
print(json.dumps(report,indent=2))
