#!/usr/bin/env python3
"""S4: MetaV3RSE SMPL-X camera-frame meshes -> MHR parameters for one episode (single identity).

Steps
  1. Load per-frame npz from metav3rse_human_infer.py (--mode full). Track ONE person: frame-0 slot with the
     tallest projected body (the subject is closest/largest in FORM-HOI captures), then per frame the slot whose
     pelvis is nearest the previous pelvis. Frames without a person are filled by linear interpolation of the
     vertex arrays between the nearest valid frames (edges: hold). Every fill is recorded.
  2. Apply the declared axis transform --flip (diagonal, default '1,-1,-1') to the camera-frame vertices so that
     the HOST's MHR decode (diag(1,-1,-1) @ MHR/100, kit MESH_TO_MHR_PARAMS.md) lands back in the camera frame.
     The validator confirms/overrides this recipe; the value used is written to MHR.json.
  3. MHR tools/mhr_smpl_conversion Conversion(method='pytorch'), single_identity=True, is_tracking=True,
     exclude_expression=True. Inputs are metres (the tool converts to MHR centimetres internally).
  4. Re-decode the fitted parameters with the kit's TorchScript lod1 model (the host's model) and report the
     per-frame vertex error against the (flipped) SMPL-X->MHR barycentric target as a sanity number.
Output: <out>/mhr_params.npz with pose[T,136], scales[68], shape[45], plus MHR.json diagnostics.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/tanuj/V2D_challenge")
MHR_REPO = ROOT / "third_party/MHR"
SMPLX_DIR = Path("/home/tanuj/UniCon3Rpp-push-b3c6624/src/models")


def track_person(rows, K, gate_m=0.5):
    T = len(rows)
    verts = [None] * T
    chosen, prev = [], None
    for i, r in enumerate(rows):
        if r is None or int(r["n_humans"]) == 0 or "verts" not in r:
            chosen.append(-1)
            continue
        V, P = r["verts"], r["pelvis"]
        if prev is None:
            # tallest projected body
            uv = (V[..., :2] / np.clip(V[..., 2:3], 1e-3, None)) * K[0, 0]
            height = uv[..., 1].max(1) - uv[..., 1].min(1)
            k = int(np.argmax(height))
        else:
            dist = np.linalg.norm(P - prev, axis=1)
            k = int(np.argmin(dist))
            if dist[k] > gate_m:          # no slot continues the subject: treat as missing
                chosen.append(-2)
                continue
        verts[i], prev = V[k], P[k]
        chosen.append(k)
    valid = np.array([v is not None for v in verts])
    if not valid.any():
        raise SystemExit("no person detected in any frame")
    idx = np.nonzero(valid)[0]
    filled = []
    out = np.zeros((T, *verts[idx[0]].shape), np.float32)
    for i in range(T):
        if valid[i]:
            out[i] = verts[i]
            continue
        lo, hi = idx[idx < i], idx[idx > i]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            w = (i - a) / (b - a)
            out[i] = (1 - w) * verts[a] + w * verts[b]
        else:
            out[i] = verts[lo[-1] if len(lo) else hi[0]]
        filled.append(i)
    return out, chosen, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True, help="work/s3_human/episode_XXXXXX")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flip", default="1,-1,-1")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    run = json.loads((args.human / "RUN.json").read_text())
    T = run["frames"]
    W, H = run["source_wh"]
    K = np.array([[run["focal_source_px"], 0, W / 2], [0, run["focal_source_px"], H / 2], [0, 0, 1]], np.float32)
    rows = []
    for i in range(T):
        f = args.human / f"{i:06d}.npz"
        rows.append(dict(np.load(f)) if f.exists() else None)
    verts_cam, chosen, filled = track_person(rows, K)
    flip = np.array([float(x) for x in args.flip.split(",")], np.float32)
    target = verts_cam * flip

    sys.path.insert(0, str(MHR_REPO / "tools/mhr_smpl_conversion"))
    import smplx
    from mhr.mhr import MHR
    from conversion import Conversion
    dev = "cuda"
    mhr_model = MHR.from_files(folder=ROOT / "models/mhr/full/assets", device=dev, lod=1)
    smplx_model = smplx.SMPLX(model_path=str(SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL.npz"), gender="neutral",
                              use_pca=False, flat_hand_mean=True, num_betas=10)
    conv = Conversion(mhr_model, smplx_model, method="pytorch", batch_size=args.batch)
    res = conv.convert_smpl2mhr(smpl_vertices=torch.from_numpy(target), single_identity=True, is_tracking=True,
                                exclude_expression=True, return_mhr_vertices=True, return_fitting_errors=True)
    prm = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in res.result_parameters.items()}
    lbs, ident = prm["lbs_model_params"], prm["identity_coeffs"]
    if lbs.shape != (T, 204) or ident.shape[-1] != 45:
        raise SystemExit(f"unexpected MHR parameter shapes {lbs.shape} {ident.shape}")
    pose = lbs[:, :136].astype(np.float32)
    scales = lbs[:, 136:204]
    spread_s = float(np.abs(scales - scales[0]).max())
    ident2 = ident.reshape(-1, 45)
    spread_i = float(np.abs(ident2 - ident2[0]).max())
    scales, shape = scales[0].astype(np.float32), ident2[0].astype(np.float32)

    # host-model re-decode (kit TorchScript lod1), host frame = diag(1,-1,-1) @ MHR / 100
    ts = torch.jit.load(str(ROOT / "models/mhr/mhr_model.pt"), map_location=dev).eval()
    with torch.no_grad():
        p = torch.from_numpy(np.concatenate([pose, np.repeat(scales[None], T, 0)], 1)).to(dev)
        z = torch.from_numpy(shape[None]).to(dev).expand(T, -1)
        vs = []
        for a in range(0, T, 128):
            v, _ = ts(z[a:a + 128].contiguous(), p[a:a + 128], torch.zeros(len(p[a:a + 128]), 72, device=dev), True)
            vs.append(v.float().cpu())
        host_v = torch.cat(vs).numpy() * np.array([1, -1, -1], np.float32) / 100.0
    conv_v = np.asarray(res.result_vertices, np.float32) / (100.0 if np.abs(res.result_vertices).max() > 20 else 1.0)
    # host vertices back in the camera frame iff flip matches the host convention
    host_in_cam = host_v
    centroid_gap = np.linalg.norm(host_in_cam.mean(1) - verts_cam.mean(1), axis=1)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "mhr_params.npz", pose=pose, scales=scales, shape=shape)
    np.save(args.out / "host_vertices_cam.npy", host_in_cam.astype(np.float16))
    errs = np.asarray(res.result_errors, np.float64).reshape(-1) if res.result_errors is not None else np.array([np.nan])
    rep = dict(frames=T, flip=flip.tolist(), filled_frames=filled, n_filled=len(filled),
               chosen_slot_hist={str(k): chosen.count(k) for k in sorted(set(chosen))},
               conversion_error=dict(mean=float(np.nanmean(errs)), p90=float(np.nanpercentile(errs, 90)),
                                     max=float(np.nanmax(errs))),
               identity_spread=dict(scales=spread_s, shape=spread_i),
               host_vs_smplx_centroid_gap_m=dict(mean=float(centroid_gap.mean()), max=float(centroid_gap.max())),
               conv_vertex_abs_max=float(np.abs(conv_v).max()))
    (args.out / "MHR.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "filled_frames"}), flush=True)


if __name__ == "__main__":
    main()
