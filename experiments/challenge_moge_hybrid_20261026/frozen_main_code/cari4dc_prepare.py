#!/usr/bin/env python3
"""Prepare commercial-CARI4D (toolkit v2d_cari4d) inputs for one V2D Track 1 episode.

  <out>/<ep>.0.color.mp4         symlink to the episode video (CARI4D requires the .0.color.mp4 suffix)
  <out>/human/000000.png ...     SUBJECT silhouette from our SAM 3D Body track (surface samples splatted + closed):
                                 same tracked person we submit, so the bystander never enters CARI4D's prompt
  <out>/object/000000.png ...    our chosen SAM3 object masks (work/MASK_CHOICE.json)
  <out>/object_mesh.glb          our S5 SAM 3D Objects mesh (metric, anchored to the SAM 3D Body human)
Then packs <out>/<ep>_masks_k0.h5 with the toolkit's own pack_masks (run separately, see run script).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

W = Path("/home/tanuj/V2D_challenge/work")
FACES = "/home/tanuj/V2D_challenge/models/mhr/mhr_faces_lod1.npy"
_B = np.array([(a, b, 1 - a - b) for a in np.linspace(0, 1, 4) for b in np.linspace(0, 1, 4) if a + b <= 1 + 1e-9])


def dilate(m, r):
    for _ in range(r):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m


def erode(m, r):
    for _ in range(r):
        n = m.copy()
        n[1:] &= m[:-1]; n[:-1] &= m[1:]; n[:, 1:] &= m[:, :-1]; n[:, :-1] &= m[:, 1:]
        m = n
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    e = f"episode_{args.episode:06d}"
    out = args.out
    (out / "human").mkdir(parents=True, exist_ok=True)
    (out / "object").mkdir(parents=True, exist_ok=True)
    info = json.loads(Path("/home/tanuj/V2D_challenge/data/video_to_data_challenge/track_1/meta/info.json").read_text())
    vid = Path("/home/tanuj/V2D_challenge/data/video_to_data_challenge/track_1") / info["video_path"].format(
        episode_chunk=0, episode_index=args.episode, video_key="observation.images.exo_camera")
    link = out / f"{e}.0.color.mp4"
    if link.is_symlink():
        link.unlink()
    if not link.exists():                       # real copy: run_inference resolves symlinks and checks the suffix
        import shutil
        shutil.copyfile(vid, link)
    sb = np.load(W / f"s3db/{e}/sam3d_body.npz")
    V, K = sb["vertices_cam"], sb["K"]
    faces = np.load(FACES)
    T = len(V)
    Wd, Hd = Image.open(W / f"frames/{e}/000000.jpg").size
    mc = json.loads((W / "MASK_CHOICE.json").read_text())
    md = W / ("masks_v2" if mc.get(str(args.episode), "v1") == "v2" else "masks") / e
    for i in range(T):
        P = np.einsum("sk,fkd->fsd", _B, V[i][faces]).reshape(-1, 3)
        z = P[:, 2]
        ok = z > 1e-3
        u = np.round(P[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(P[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(int)
        inb = (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)
        m = np.zeros((Hd, Wd), bool)
        m[v[inb], u[inb]] = True
        m = erode(dilate(m, 3), 2)                     # close the splat holes
        om = np.asarray(Image.open(md / f"{i:06d}.png")) > 127
        m &= ~om                                       # instance masks must not overlap: the object occludes the body
        Image.fromarray(m.astype(np.uint8) * 255).save(out / "human" / f"{i:06d}.png")
    for i in range(T):
        dst = out / "object" / f"{i:06d}.png"
        if not dst.exists():
            os.symlink(md / f"{i:06d}.png", dst)
    import trimesh
    trimesh.load(W / f"s5/{e}/object_mesh.ply", process=False).export(out / "object_mesh.glb")
    (out / "PREP.json").write_text(json.dumps(dict(episode=args.episode, frames=T, video=str(vid),
                                                   masks=str(md), mesh=str(W / f"s5/{e}/object_mesh.ply")), indent=1))
    print("prepared", out, T, "frames", flush=True)


if __name__ == "__main__":
    main()
