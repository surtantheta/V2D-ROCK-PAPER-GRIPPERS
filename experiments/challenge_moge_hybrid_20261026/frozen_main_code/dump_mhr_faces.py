#!/usr/bin/env python3
"""Save the MHR lod1 triangle list (18,439-vertex topology) to models/mhr/mhr_faces_lod1.npy.
Tries the kit TorchScript model's buffers/attributes first; falls back to the SAM 3D Body head (head_pose.faces)."""
import sys
import numpy as np
import torch

OUT = "/home/tanuj/V2D_challenge/models/mhr/mhr_faces_lod1.npy"
m = torch.jit.load("/home/tanuj/V2D_challenge/models/mhr/mhr_model.pt", map_location="cpu")
cands = []
for name, t in list(m.named_buffers()) + list(m.named_parameters()):
    if t.dim() == 2 and t.shape[1] == 3 and not t.is_floating_point():
        cands.append((name, t))
for name in dir(m):
    try:
        t = getattr(m, name)
    except Exception:
        continue
    if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == 3 and not t.is_floating_point():
        cands.append((name, t))
cands = [(n, t) for n, t in cands if int(t.max()) == 18438 and int(t.min()) == 0]
print("torchscript candidates:", [(n, tuple(t.shape)) for n, t in cands])
if cands:
    faces = cands[0][1].numpy().astype(np.int64)
else:
    sys.path.insert(0, "/home/tanuj/V2D_challenge/video_to_data/reconstruction/modules/v2d_sam3d_body/lib")
    import os
    os.environ.setdefault("TORCH_HOME", "/home/tanuj/V2D_challenge/models/sam3d_body/torch_home")
    from sam_3d_body import load_sam_3d_body
    W = "/home/tanuj/V2D_challenge/models/sam3d_body/sam-3d-body-dinov3"
    model, _ = load_sam_3d_body(checkpoint_path=f"{W}/model.ckpt", device="cpu", mhr_path=f"{W}/assets/mhr_model.pt")
    faces = model.head_pose.faces.cpu().numpy().astype(np.int64)
    print("from SAM 3D Body head_pose.faces")
assert faces.ndim == 2 and faces.shape[1] == 3 and faces.min() == 0 and faces.max() == 18438, faces.shape
np.save(OUT, faces)
print("saved", OUT, faces.shape)
