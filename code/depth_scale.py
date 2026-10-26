#!/usr/bin/env python3
"""Depth anchoring v3 (validator 03 must-fix 3): static-background window alignment + ONE episode-level scale.

History: v1 used the MetaV3RSE body (wrong premise, validator 02 D1); v2 anchored each 16-frame window to the SAM 3D
Body surface, which passed SAM 3D Body's per-frame depth jitter into the object (validator 03: hoop-region depth still
stepped 5.3-5.9 cm at window boundaries).

v3:
  1. Reference depth map R = per-pixel median of the MetaV3RSE depth over a frame subsample (static camera, so the
     background is the same scene in every frame).
  2. Background pixels per frame = not human (SAM 3D Body surface z-buffer, dilated 4 px), not object (SAM3 mask,
     dilated 4 px), finite depth. Per-frame ratio r_t = median(R / D_t) over them; per-window scale a_w = median r_t
     inside the window (MetaV3RSE re-initialises per forward window). Aligned depth A_t = a_w(t) * D_t.
  3. Episode scale c = median over frames of median(z_body / A_t) on the eroded human pixels (one number per
     episode, so SAM 3D Body depth jitter is averaged out instead of injected).
  Final per-frame multiplier scale_t = c * a_w(t). Diagnostics: background step at window boundaries before/after.
Output: <out>.npz (scale[T], window_align[T], c) + <out>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

FACES = "/home/tanuj/V2D_challenge/models/mhr/mhr_faces_lod1.npy"
_B = np.array([(a, b, 1 - a - b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5) if a + b <= 1 + 1e-9])


def surface_points(V, faces):
    return np.einsum("sk,fkd->fsd", _B, V[faces]).reshape(-1, 3)


def zbuffer(P, K, w, h):
    z = P[:, 2]
    ok = z > 1e-3
    u = np.floor(P[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(int)
    v = np.floor(P[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(int)
    zz = z[ok]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    buf = np.full((h, w), np.inf)
    np.minimum.at(buf, (v[inb], u[inb]), zz[inb])
    return buf


def morph(m, r, grow):
    out = m.copy()
    for _ in range(r):
        n = out.copy()
        if grow:
            n[1:] |= out[:-1]; n[:-1] |= out[1:]; n[:, 1:] |= out[:, :-1]; n[:, :-1] |= out[:, 1:]
        else:
            n[1:] &= out[:-1]; n[:-1] &= out[1:]; n[:, 1:] &= out[:, :-1]; n[:, :-1] &= out[:, 1:]
        out = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--s3db", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref-stride", type=int, default=4)
    ap.add_argument("--affine", action="store_true", help="per-window a*D+b fit on background instead of scale only")
    args = ap.parse_args()
    run = json.loads((args.human / "RUN.json").read_text())
    T, win = run["frames"], int(run.get("window", 16))
    W, H = run["source_wh"]
    w, h = run["model_wh"]
    f = run["focal_source_px"] * w / W
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    V = np.load(args.s3db / "sam3d_body.npz")["vertices_cam"]
    faces = np.load(FACES)
    mrows = json.loads((args.masks / "MASKS.json").read_text())["rows"]

    D = np.stack([np.load(args.human / f"{i:06d}.npz")["depth"].astype(np.float32) for i in range(T)])
    R = np.median(D[:: args.ref_stride], axis=0)
    hum, obj = [], []
    for i in range(T):
        buf = zbuffer(surface_points(V[i].astype(np.float64), faces), K, w, h)
        hum.append(np.isfinite(buf))
        m = np.asarray(Image.open(mrows[i]["mask"]).resize((w, h), Image.NEAREST)) > 127
        obj.append(m)
        if i == 0:
            zb = [buf]
        else:
            zb.append(buf)
    r = np.full(T, np.nan)
    bgs = []
    for i in range(T):
        bg = ~morph(hum[i], 4, True) & ~morph(obj[i], 4, True) & (D[i] > 1e-3) & (R > 1e-3)
        bgs.append(bg)
        if bg.sum() > 500:
            r[i] = np.median(R[bg] / D[i][bg])
    a = np.full(T, np.nan)
    b = np.zeros(T)
    for k in range(-(-T // win)):
        sl = slice(k * win, min((k + 1) * win, T))
        if not np.isfinite(r[sl]).any():
            continue
        if args.affine:
            xs = np.concatenate([D[i][bgs[i]] for i in range(sl.start, sl.stop) if bgs[i].sum() > 500])
            ys = np.concatenate([R[bgs[i]] for i in range(sl.start, sl.stop) if bgs[i].sum() > 500])
            A_ = np.stack([xs, np.ones_like(xs)], 1)
            ww = np.ones_like(xs)
            for _ in range(5):                         # IRLS (Huber-like) robust fit
                coef = np.linalg.lstsq(A_ * ww[:, None], ys * ww, rcond=None)[0]
                res = np.abs(A_ @ coef - ys)
                ww = 1.0 / np.maximum(res / max(np.median(res), 1e-4), 1.0)
            a[sl], b[sl] = coef
        else:
            a[sl] = np.nanmedian(r[sl])
    good = np.isfinite(a)
    if (~good).any():
        a[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), a[good])
    ratios = []
    for i in range(T):
        hm = morph(hum[i], 1, False) & (D[i] > 1e-3)
        if hm.sum() >= 5:
            ratios.append(np.median(zb[i][hm] / (a[i] * D[i][hm] + b[i])))
    c = float(np.median(ratios))
    s = c * a
    off = c * b       # aligned depth = scale_t * D_t + offset_t

    # diagnostics: median |background depth step| across window boundaries vs inside windows, before/after
    def steps(Dx):
        inside, across = [], []
        for i in range(1, T):
            bg = ~morph(hum[i], 4, True) & ~morph(obj[i], 4, True) & ~morph(hum[i - 1], 4, True)
            d = np.median(np.abs(Dx[i][bg] - Dx[i - 1][bg])) if bg.sum() > 500 else np.nan
            (across if i % win == 0 else inside).append(d)
        return float(np.nanmedian(inside)) * 100, float(np.nanmedian(across)) * 100
    A = D * a[:, None, None] + b[:, None, None]
    bi, ba = steps(D)
    ai, aa = steps(A)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, scale=s, offset=off, window_align=a, window_offset=b, c=c, frame_ratio=r)
    rep = dict(frames=T, window=win, affine=bool(args.affine), episode_scale_c=c, n_human_ratio_frames=len(ratios),
               human_ratio_p05_p95=[float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))],
               window_align_range=[float(a.min()), float(a.max())],
               bg_step_cm_inside_across_before=[bi, ba], bg_step_cm_inside_across_after=[ai, aa])
    args.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
