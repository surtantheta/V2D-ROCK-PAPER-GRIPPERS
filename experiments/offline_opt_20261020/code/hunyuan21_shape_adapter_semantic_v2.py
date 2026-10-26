#!/usr/bin/env python3
"""Shape-only open-world proposal from a hash-pinned predicted RGBA image.

No metric size, pose, depth or native CAD is inferred by this adapter. Output is
in the model's arbitrary normalized object coordinates. Run only inside an
agreed scheduler GPU allocation; --verify-setup is CPU-only and loads no weights.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/tanuj/projects/GRAIL/imports/Hunyuan3D-2.1')
SOURCE_COMMIT = '82920d643c0dc2f7bfd7255f45f62d386edfe60c'
SETUP = ROOT / 'setup/hunyuan21'
DEPS = ROOT / 'deps/hunyuan21'
CHECKPOINT_SHA = '6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def setup_check():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError('local Hunyuan source commit changed')
    # Fail on modified tracked source, not merely a matching git HEAD.
    changed = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no', '--', 'hy3dshape'], text=True)
    if changed.strip():
        raise ValueError('tracked Hunyuan shape source differs from frozen commit')
    os.environ['USE_SAGEATTN'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path[:0] = [str(DEPS), str(SOURCE / 'hy3dshape'), str(ROOT.parents[1] / 'data/assets/submission_kit/v2d_submission_kit')]
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from hy3dshape.models.conditioner import SingleImageEncoder
    import fast_simplification
    from v2dlb.mesh_budget import budget_mesh
    import torch
    sources = {str(p): sha(p) for p in (SOURCE / 'hy3dshape/hy3dshape').rglob('*.py')}
    sources[str(Path(__file__))] = sha(__file__)
    return {'source_commit': commit, 'source_sha256': sources,
            'dependencies': {n: importlib.metadata.version(n) for n in ['torch', 'diffusers', 'transformers', 'huggingface_hub', 'pymeshlab', 'fast-simplification']},
            'torch_cuda': torch.version.cuda, 'compiled_arches': torch.cuda.get_arch_list(),
            'checkpoint_ready': (SETUP / 'DOWNLOAD_RECEIPT.json').exists(),
            'imports': 'pipeline, actual configured denoiser, conditioner, mesh budget PASS',
            'gpu_smoke_completed': False, 'dtype': 'float16; no bfloat16 or SageAttention'}, Hunyuan3DDiTFlowMatchingPipeline, budget_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-setup', action='store_true')
    parser.add_argument('--rgba', type=Path)
    parser.add_argument('--rgba-sha256')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--num-chunks', type=int, default=8000)
    args = parser.parse_args()
    checked, Pipeline, budget_mesh = setup_check()
    if args.verify_setup:
        print(json.dumps(checked, indent=2)); return
    if args.rgba is None or args.rgba_sha256 is None or args.out is None:
        parser.error('--rgba, --rgba-sha256 and --out required for generation')
    if args.out.exists() or not 1 <= args.steps <= 100 or not 32 <= args.resolution <= 512 or args.num_chunks < 1:
        parser.error('output must be new; steps[1,100], resolution[32,512], positive chunks')
    if sha(args.rgba) != args.rgba_sha256:
        raise ValueError('input RGBA SHA mismatch')
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    image = Image.open(args.rgba)
    if image.mode != 'RGBA':
        raise ValueError('require the same predicted RGBA artifact used in paired generator arm')
    alpha = np.asarray(image)[..., 3]
    if not (alpha == 0).any() or not (alpha > 0).any():
        raise ValueError('RGBA must contain both foreground and transparent background')
    receipt = json.loads((SETUP / 'DOWNLOAD_RECEIPT.json').read_text())
    for record in receipt['records']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('checkpoint/config content differs from pinned download receipt')
    checkpoint = SETUP / 'checkpoint/hunyuan3d-dit-v2-1/model.fp16.ckpt'
    config = checkpoint.parent / 'config.yaml'
    if receipt['records'][-1]['sha256'] != CHECKPOINT_SHA:
        raise ValueError('checkpoint does not match official pinned LFS SHA')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; require coordinated scheduler allocation')
    args.out.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    # Exercise the configured denoiser's FP16 non-math SDPA backend before loading
    # multi-GB weights. This is a real model-readiness gate, not an occupancy load.
    q = torch.zeros((1, 16, 64, 128), device='cuda', dtype=torch.float16)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        attention_gate = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    if not torch.isfinite(attention_gate).all():
        raise RuntimeError('configured FP16 SDPA returned nonfinite values')
    del q, attention_gate
    begin = time.monotonic()
    pipeline = Pipeline.from_single_file(str(checkpoint), str(config), device='cuda', dtype=torch.float16, use_safetensors=False)
    loaded = time.monotonic()
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    with torch.inference_mode():
        mesh = pipeline(image=image, generator=generator, num_inference_steps=args.steps,
                        octree_resolution=args.resolution, num_chunks=args.num_chunks,
                        guidance_scale=5.0, mc_algo='mc', output_type='trimesh')[0]
    torch.cuda.synchronize()
    generated = time.monotonic()
    if mesh is None or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise RuntimeError('generator returned no valid finite surface')
    raw = args.out / 'shape_normalized_raw.ply'
    mesh.export(raw)
    vertices, faces = budget_mesh(raw, faces=4096, vertices=4096)
    budgeted = args.out / 'shape_normalized_4096.ply'
    trimesh.Trimesh(vertices, faces, process=False).export(budgeted)
    artifact = args.out / 'shape_normalized_4096.npz'
    np.savez_compressed(artifact, vertices=vertices.astype(np.float32), faces=faces.astype(np.int32))
    if len(vertices) != 4096 or len(faces) != 4096 or not np.isfinite(vertices).all():
        raise RuntimeError('submission mesh budget gate failed')
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if not (area > 1e-14).any():
        raise RuntimeError('budgeted mesh has no nondegenerate surface')
    report = {'schema': 'hunyuan21-open-world-shape-proposal-v1', 'setup': checked,
              'input_rgba': str(args.rgba.resolve()), 'input_rgba_sha256': args.rgba_sha256,
              'checkpoint_receipt_sha256': sha(SETUP / 'DOWNLOAD_RECEIPT.json'),
              'checkpoint_revision': receipt['revision'], 'checkpoint_sha256': CHECKPOINT_SHA,
              'output_sha256': {str(p): sha(p) for p in [raw, budgeted, artifact]},
              'coordinate_units': 'model-normalized arbitrary units; NOT metres',
              'metric_scale_recovered': False, 'camera_pose_recovered': False,
              'GT_inputs_provided': False, 'native_CAD_input': False,
              'origin_changed_by_adapter': False, 'seed': args.seed,
              'inference': {'steps': args.steps, 'resolution': args.resolution, 'num_chunks': args.num_chunks, 'guidance_scale': 5.0, 'dtype': 'float16', 'batch_size': 1, 'surface_extractor': 'mc'},
              'preprocessing': 'same pinned RGBA input; native ImageProcessorV2 recenter border0.15 and resize512; DINO resize518',
              'raw_vertices': len(mesh.vertices), 'raw_faces': len(mesh.faces),
              'budget_vertices': len(vertices), 'budget_faces': len(faces), 'nondegenerate_faces': int((area>1e-14).sum()),
              'raw_extent_arbitrary_units': np.ptp(mesh.vertices, axis=0).tolist(),
              'seconds_model_loading': loaded-begin, 'seconds_generation': generated-loaded,
              'host_peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'configured_fp16_nonmath_SDPA_gate': True,
              'cuda_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'cuda_device': torch.cuda.get_device_name(), 'cuda_capability': torch.cuda.get_device_capability(),
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'mesh_budget_source_sha256': sha(ROOT.parents[1] / 'data/assets/submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py')}
    report['setup']['gpu_smoke_completed'] = True
    (args.out / 'PROPOSAL.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
