#!/usr/bin/env python3
"""Run the parallel optimization branch's frozen trajectory fitter on one V2D Track 1 test episode.

The fitter, arm definitions and export validation are imported unchanged from
experiments/offline_opt_20261020/code (run_identity_scale_v2.fit_arm / export_validated,
centroid_scale_trajectory_v2.solve, constrained_trajectory_v1.solve). Only the observation builder is
re-implemented here, mirroring run_identity_scale_v2.observe() line for line, on OUR artifacts:

  baseline translation  raw S6 FoundationPose track (work/s6, the branch's "legacy FP"), invalid frames interpolated
  rotation (fixed)      v2.1 object_rotation (S7-smoothed), so the comparison with v2.1 differs only in translation/scale
  depth                 MetaV3RSE depth * depth_scale (the metric frame S6 tracked in), 512x384 model resolution
  object mask           our chosen SAM3 masks (MASK_CHOICE.json); enabled = selected and score >= 0.8
  person mask           SAM 3D Body silhouette (same splat as cari4dc_prepare.py) at model resolution
  fingertips            SAM 3D Body MHR vertices -> SMPL-H fingertip vertices (branch mapping), every 8th frame
Writes <out>/<arm>/episode_XXXXXX.npz (+ mesh) in submission format and <out>/FIT_XXXXXX.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

BR = Path("/home/tanuj/V2D_challenge/experiments/offline_opt_20261020/code")
sys.path.insert(0, str(BR))
import run_identity_scale_v2 as frozen                      # noqa: E402  (fit_arm, export_validated, sha)
from run_constrained_intercap_v1 import sample_surface      # noqa: E402

W = Path("/home/tanuj/V2D_challenge/work")
PROJECT = Path("/home/tanuj/V2D_challenge")
FACES = PROJECT / "models/mhr/mhr_faces_lod1.npy"
MAPPING = PROJECT / "third_party/MHR/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz"
TIP_NAMES = [s + f for s in ["l", "r"] for f in ["thumb", "index", "middle", "ring", "pinky"]]
TIP_IDS = [2746, 2319, 2445, 2556, 2673, 6191, 5782, 5905, 6016, 6133]   # branch EXPECTED_IDS (SMPL-H)
ARMS = ["identity_temporal", "fixedscale_geometry", "shared_scale", "shared_scale_contact"]
_B = np.array([(a, b, 1 - a - b) for a in np.linspace(0, 1, 4) for b in np.linspace(0, 1, 4) if a + b <= 1 + 1e-9])


def silhouette(V, faces, k, w, h):
    P = np.einsum("sk,fkd->fsd", _B, V[faces]).reshape(-1, 3)
    z = P[:, 2]; ok = z > 1e-3
    u = np.round(P[ok, 0] / z[ok] * k[0, 0] + k[0, 2]).astype(int)
    v = np.round(P[ok, 1] / z[ok] * k[1, 1] + k[1, 2]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), bool); m[v[inb], u[inb]] = True
    return ndimage.binary_erosion(ndimage.binary_dilation(m, iterations=3), iterations=2)


def observe(ep, v21, mesh, baseline="s6"):
    e = f"episode_{ep:06d}"
    tr = np.load(W / f"s6/{e}/poses.npz")
    P = tr["T_obj2cam"].astype(np.float64); T = len(P)
    k = tr["K_model"].astype(np.float64); w, h = 512, 384
    assert np.isclose(k[0, 2], w / 2) and np.isclose(k[1, 2], h / 2)
    ok = np.isfinite(P).all(axis=(1, 2)) & (P[:, 2, 3] > 0.05)
    idx = np.flatnonzero(ok)
    translation = P[:, :3, 3].copy()
    for c in range(3):
        translation[~ok, c] = np.interp(np.flatnonzero(~ok), idx, translation[ok, c])
    if baseline == "v21":                                         # hybrid: start from v2.1's repaired, smoothed translation
        translation = v21["object_translation"].astype(np.float64).copy()
    rotations = v21["object_rotation"].astype(np.float64)
    assert len(rotations) == T and np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    alpha = 1.0                                                   # S5 mesh is already metric (object_scale 1)
    surface = sample_surface(mesh, 4096, 271828) * alpha
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    mu = np.average(tri.mean(1), axis=0, weights=area) * alpha
    quality = np.nan_to_num(tr["iou"].astype(np.float64), nan=.1).clip(.1, 1.)
    mc = json.loads((W / "MASK_CHOICE.json").read_text()) if (W / "MASK_CHOICE.json").exists() else {}
    md = W / ("masks_v2" if mc.get(str(ep), "v1") == "v2" else "masks") / e
    rows = json.loads((md / "MASKS.json").read_text())["rows"]
    assert len(rows) == T
    ds = np.load(W / f"depth_scale/{e}.npz"); sc, of = ds["scale"], ds["offset"]
    sb = np.load(W / f"s3db/{e}/sam3d_body.npz")
    Vb, Kb = sb["vertices_cam"].astype(np.float64), sb["K"].astype(np.float64)
    faces = np.load(FACES)
    mp = np.load(MAPPING); tid, bary = mp["triangle_ids"][TIP_IDS], mp["baryc_coords"][TIP_IDS]
    tip_tri = faces[tid]
    sw = Image.open(W / f"frames/{e}/000000.jpg").size[0]
    k_sil = np.diag([w / sw, w / sw, 1.]) @ Kb                   # SAM 3D Body K at source res -> model res
    clouds, images, confidence, contacts, audit = [], [], [], {}, []
    for i, row in enumerate(rows):
        obj = np.asarray(Image.open(row["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
        person = silhouette(Vb[i], faces, k_sil, w, h) & ~obj
        depth = np.load(W / f"s3_human/{e}/{i:06d}.npz")["depth"].astype(np.float64) * float(sc[i]) + float(of[i])
        depth *= alpha
        assert depth.shape == (h, w)
        enabled = bool(row.get("selected") and row.get("score", 0) >= .8)
        if not enabled:
            quality[i] = .1
        positive = obj
        observed = ndimage.binary_erosion(obj, iterations=1) & ~ndimage.binary_dilation(person, iterations=1)
        observed &= np.isfinite(depth) & (depth > 0)
        if enabled and observed.any():
            z = depth[observed]; med = np.median(z); mad = 1.4826 * np.median(abs(z - med))
            observed &= abs(depth - med) <= max(.05, 3 * mad)
        else:
            observed[:] = False
        y, x = np.nonzero(observed); eligible = len(x)
        rng = np.random.default_rng(1000 + i)
        if len(x) >= 32:
            pick = rng.choice(len(x), min(len(x), 256), replace=False)
            x, y = x[pick], y[pick]; z = depth[y, x]
            cloud = np.c_[((x + .5 - k[0, 2]) / k[0, 0]) * z, ((y + .5 - k[1, 2]) / k[1, 1]) * z, z]
        else:
            cloud = np.empty((0, 3))
        clouds.append(cloud); confidence.append(float(row["score"]) if enabled else 0.)
        y, x = np.nonzero(positive if enabled else np.zeros_like(obj))
        if len(x):
            pick = rng.choice(len(x), min(len(x), 128), replace=False)
            maskuv = np.c_[x[pick] + .5, y[pick] + .5]
        else:
            maskuv = np.empty((0, 2))
        images.append(dict(Kmodel=k.copy(), maskuv=maskuv, maskdistancefield=ndimage.distance_transform_edt(~(obj | person)),
                           confidence=float(row["score"]) if enabled else 0., enabled=enabled))
        hands = []
        if enabled and i % 8 == 0 and obj.any():                 # branch contact inputs: every 8th frame
            positions = np.sum(Vb[i][tip_tri] * bary[:, :, None], axis=1)
            if np.isfinite(positions).all() and (positions[:, 2] > .001).all():
                distance = ndimage.distance_transform_edt(~obj)
                uv = positions[:, :2] / positions[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]
                for side in ["l", "r"]:
                    keep = []
                    for j, name in enumerate(TIP_NAMES):
                        if not name.startswith(side) or positions[j, 2] <= 0:
                            continue
                        u, v = np.floor(uv[j]).astype(int)
                        if 0 <= u < w and 0 <= v < h and distance[v, u] <= 4.:
                            keep.append(j)
                    if keep:
                        contacts.setdefault(i, []).append(positions[keep].copy())
                        hands.append(dict(side=side, tips=[TIP_NAMES[j] for j in keep]))
        audit.append(dict(local_frame_index=i, selected=enabled, eligible_depth_pixels=eligible,
                          sampled_depth_points=len(cloud), positive_mask_points=len(maskuv), active_hands=hands))
    times, ids, step = np.arange(T) / 30., list(range(T)), 1
    lab = W / "LABELED_EPISODES.json"                            # labeled clips: declared source frame ids / times / cadence
    if lab.exists() and str(ep) in (info := json.loads(lab.read_text())):
        ids = [int(x) for x in info[str(ep)]["frame_ids"]]; times = np.asarray(info[str(ep)]["times"], float)
        step = 5 if info[str(ep)]["dataset"] == "InterCap" else 1
    obs = dict(surface_metric=surface, mu_metric=mu, rotations=rotations, baseline_translation=translation,
               clouds=clouds, quality=quality, depth_confidence=np.asarray(confidence), times=times,
               image_observations=images, contacts=contacts, frame_ids=ids, frame_step=step)
    return obs, audit, int((~ok).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--work", type=Path, default=W, help="work root (work_labeled for the labeled-proxy clips)")
    ap.add_argument("--v21", type=Path, default=None, help="default <work>/submission_v2")
    ap.add_argument("--out", type=Path, default=None, help="default <work>/branchfit")
    ap.add_argument("--baseline", choices=["s6", "v21"], default="s6",
                    help="initial translation: raw S6 track (branch workflow) or v2.1 S7 output (hybrid)")
    args = ap.parse_args()
    globals()["W"] = args.work
    args.v21 = args.v21 or args.work / "submission_v2"
    args.out = args.out or args.work / "branchfit"
    e = f"episode_{args.episode:06d}"
    v21 = dict(np.load(args.v21 / f"{e}.npz"))
    mesh_path = args.v21 / f"{e}_object.ply"
    mesh = trimesh.load(mesh_path, process=False)
    obs, audit, filled = observe(args.episode, v21, mesh, args.baseline)
    T = len(obs["rotations"]); ids = obs["frame_ids"]
    rep = dict(episode=args.episode, frames=T, baseline=args.baseline, s6_invalid_frames_interpolated=filled,
               enabled_frames=int(sum(a["selected"] for a in audit)),
               depth_frames=int(sum(a["sampled_depth_points"] > 0 for a in audit)),
               contact_frames=len(obs["contacts"]), arms={})
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    for arm in ARMS:
        t, ratio, diag = frozen.fit_arm(arm, obs, obs["times"], ids, obs["frame_step"])
        t, ratio, diag = frozen.export_validated(t, ratio, diag, obs, vertices)
        out = args.out / arm; out.mkdir(parents=True, exist_ok=True)
        sub = dict(v21)
        sub["object_translation"] = np.asarray(t, np.float64)
        sub["object_scale"] = np.float64(ratio)
        np.savez_compressed(out / f"{e}.npz", **sub)
        shutil.copyfile(mesh_path, out / f"{e}_object.ply")
        d = np.linalg.norm(t - obs["baseline_translation"], axis=1) * 100
        dv = np.linalg.norm(t - v21["object_translation"], axis=1) * 100
        ev = diag["export_validation"]
        rep["arms"][arm] = dict(relative_scale=float(ratio), fallback=ev["whole_case_identity_fallback"],
                                fallback_reason=ev["fallback_reason"], min_mesh_z_m=ev["candidate_all_mesh_vertices_min_z_m"],
                                shift_vs_s6_cm=dict(mean=float(d.mean()), max=float(d.max())),
                                shift_vs_v21_cm=dict(mean=float(dv.mean()), max=float(dv.max())),
                                numerical_failure=diag.get("numerical_failure"))
        print(e, arm, json.dumps(rep["arms"][arm]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"FIT_{args.episode:06d}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
