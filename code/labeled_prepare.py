#!/usr/bin/env python3
"""Stage the parallel branch's 12 labeled development clips (4 BEHAVE + 8 InterCap) as pseudo-episodes 900-911 for
our S1-S7 pipeline, so both methods are scored on the same frames by the branch's labeled_object_scorer.

Inputs are exactly what the branch's inference was permitted: the RGB frames and the object text prompt of
benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json (no calibration, no released masks, no GT).
  <W>/frames/episode_000900/000000.jpg ...  same frames, renumbered 0..N-1
     InterCap 1920x1080 is centre-cropped to 1440x1080 (4:3, as our pipeline assumes); a centre crop keeps the
     camera frame and principal-point convention, so camera-space predictions are directly scorable
  <W>/LABELED_EPISODES.json                 episode -> dataset, sequence/take, frame ids, times, prompt, crop
"""
import json
from pathlib import Path

from PIL import Image

REQ = Path("/home/tanuj/V2D_challenge/experiments/offline_opt_20261020/benchmarks/labeled_dev_v2/INFERENCE_REQUESTS_FINAL.json")
W = Path("/home/tanuj/V2D_challenge/work_labeled")


def main():
    req = json.loads(REQ.read_text())
    episodes = {}
    ep = 900
    for dataset in ["BEHAVE", "InterCap"]:
        for r in req[dataset]:
            e = f"episode_{ep:06d}"
            out = W / "frames" / e; out.mkdir(parents=True, exist_ok=True)
            crop = None
            for i, p in enumerate(r["rgb_paths"]):
                im = Image.open(p).convert("RGB")
                w, h = im.size
                if abs(w / h - 4 / 3) > 1e-3:
                    cw = int(round(h * 4 / 3)); x0 = (w - cw) // 2
                    crop = [x0, 0, x0 + cw, h]
                    im = im.crop(crop)
                im.save(out / f"{i:06d}.jpg", quality=95)
            ids = r["native_frame_numbers"] if dataset == "InterCap" else r["video_frame_indices"]
            times = [x / 30. for x in ids] if dataset == "InterCap" else r["timestamps_seconds"]
            (out / "FRAMES.json").write_text(json.dumps(dict(episode=ep, frames=len(r["rgb_paths"]), source=str(REQ), quality=95)))
            episodes[str(ep)] = dict(dataset=dataset, sequence=r.get("take") or r["sequence"], frame_ids=ids, times=times,
                                     prompt=r["object_prompt"], source_wh=[w, h], crop_xyxy=crop, frames=len(ids))
            print(ep, dataset, episodes[str(ep)]["sequence"], len(ids), r["object_prompt"], crop, flush=True)
            ep += 1
    (W / "LABELED_EPISODES.json").write_text(json.dumps(episodes, indent=1))


if __name__ == "__main__":
    main()
