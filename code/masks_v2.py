#!/usr/bin/env python3
"""S2 v2: SAM 3 object masks with a multi-prompt, best-score selector (validator 03 must-fix 1).

The v1 run (UniCon3R++ tools/wild_video_masks.py) took the FIRST prompt that returned any admissible instance.
With the supplied object_prompt this found nothing for the iron (eps 6, 8) and low scores for several objects.
v2 evaluates every prompt in --prompt on each frame and keeps the highest-scoring admissible instance across prompts;
when a previous mask is recent, candidates within the top half by score are re-ranked by centroid distance (same
temporal rule as v1). Prompts are the dataset's object name / object_prompt plus hand-written synonyms of the object
name (e.g. "steam iron"); no GT or reference data is used.

--probe-every N: run only every N-th frame and write PROBE.json with per-prompt detection rate and scores
(used to choose prompts; no masks written).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

REV = "3c879f39826c281e95690f02c7821c4de09afae7"
MAX_GAP = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--min-area", type=float, default=0.0001)
    ap.add_argument("--max-area", type=float, default=0.60)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--probe-every", type=int, default=0)
    ap.add_argument("--person-boxes", type=Path, default=None,
                    help="sam3d_body.npz with per-frame subject boxes: run SAM 3 on a zoomed person-centred crop")
    ap.add_argument("--crop-scale", type=float, default=2.5)
    ap.add_argument("--crop-size", type=int, default=1008)
    ap.add_argument("--near-subject", type=Path, default=None,
                    help="sam3d_body.npz: rank candidates by score * proximity to the subject box (interaction prior)")
    args = ap.parse_args()
    import torch
    from transformers import Sam3Model, Sam3Processor
    dev = "cuda"
    model = Sam3Model.from_pretrained("facebook/sam3", revision=REV).to(dev).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3", revision=REV)
    files = sorted(args.frames.glob("*.jpg"))
    boxes = np.load(args.person_boxes)["boxes"] if args.person_boxes else None
    sboxes = np.load(args.near_subject)["boxes"] if args.near_subject else None
    fidx = list(range(len(files)))
    if args.probe_every:
        fidx = fidx[:: args.probe_every]
        files = files[:: args.probe_every]
    args.out.mkdir(parents=True, exist_ok=True)
    probe = {p: dict(hits=0, scores=[], near=0) for p in args.prompt}
    rows, empty, prev, last = [], 0, None, -10 ** 9
    for i, f in enumerate(files):
        full = Image.open(f).convert("RGB")
        img, crop = full, None
        if boxes is not None:
            x0, y0, x1, y1 = boxes[fidx[i]]
            cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * args.crop_scale
            side = min(side, full.width, full.height)
            cx0 = int(round(min(max(cx - side / 2, 0), full.width - side)))
            cy0 = int(round(min(max(cy - side / 2, 0), full.height - side)))
            crop = (cx0, cy0, cx0 + int(side), cy0 + int(side))
            img = full.crop(crop).resize((args.crop_size, args.crop_size), Image.BILINEAR)
        HW = full.height * full.width
        cands = []
        for p in args.prompt:
            inp = proc(images=img, text=p, return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = model(**inp)
            res = proc.post_process_instance_segmentation(out, threshold=args.threshold, mask_threshold=0.5,
                                                          target_sizes=[(img.height, img.width)])[0]
            best_p, best_c = 0.0, None
            for m, s in zip(res.get("masks", []), res.get("scores", [])):
                mm = m.cpu().numpy().astype(bool)
                if crop is not None:          # paste the crop mask back into the full frame
                    cm = Image.fromarray(mm.astype(np.uint8) * 255).resize((crop[2] - crop[0], crop[3] - crop[1]), Image.NEAREST)
                    big = np.zeros((full.height, full.width), bool)
                    big[crop[1]:crop[3], crop[0]:crop[2]] = np.asarray(cm) > 127
                    mm = big
                af = mm.sum() / HW
                if args.min_area <= af <= args.max_area:
                    ys, xs = np.nonzero(mm)
                    cands.append(dict(mask=mm, score=float(s), area=float(af), prompt=p,
                                      centroid=(float(xs.mean()), float(ys.mean()))))
                    if float(s) > best_p:
                        best_p, best_c = float(s), cands[-1]["centroid"]
            if best_p > 0:
                probe[p]["hits"] += 1
                probe[p]["scores"].append(best_p)
                if sboxes is not None:
                    x0, y0, x1, y1 = sboxes[fidx[i]]
                    probe[p]["near"] += (abs(best_c[0] - (x0 + x1) / 2) <= (x1 - x0)) and (abs(best_c[1] - (y0 + y1) / 2) <= (y1 - y0))
        if args.probe_every:
            continue
        chosen = None
        if cands and sboxes is not None:
            x0, y0, x1, y1 = sboxes[fidx[i]]
            hb = max(y1 - y0, 1.0)
            for c in cands:     # 0 distance inside the subject box enlarged x1.5, else exp decay in box heights
                ex, ey = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
                dx = max(x0 - ex - c["centroid"][0], 0, c["centroid"][0] - x1 - ex)
                dy = max(y0 - ey - c["centroid"][1], 0, c["centroid"][1] - y1 - ey)
                c["near_w"] = float(np.exp(-np.hypot(dx, dy) / hb))
                c["rank"] = c["score"] * c["near_w"]
            cands = [c for c in cands if c["near_w"] >= 0.3]      # far from the subject: leave the frame empty
        for c in cands:
            c.setdefault("rank", c["score"])
        if cands:
            cands.sort(key=lambda c: -c["rank"])
            if prev is not None and i - last <= MAX_GAP and sboxes is None:   # near-subject prior replaces temporal rule
                top = cands[: max(1, len(cands) // 2)]
                chosen = min(top, key=lambda c: (c["centroid"][0] - prev[0]) ** 2 + (c["centroid"][1] - prev[1]) ** 2)
            else:
                chosen = cands[0]
        if chosen is None:
            mask = np.zeros((full.height, full.width), bool)
            empty += 1
            row = dict(score=0.0, prompt=None, area=0.0, centroid=None)
        else:
            mask, prev, last = chosen["mask"], chosen["centroid"], i
            row = {k: chosen[k] for k in ("score", "prompt", "area", "centroid")}
        dst = args.out / (f.stem + ".png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(dst)
        rows.append(dict(frame=f.name, mask=str(dst), selected=bool(mask.any()),
                         mask_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), hw=[full.height, full.width],
                         crop=list(crop) if crop else None, **row))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} empty={empty}", flush=True)
    if args.probe_every:
        rep = {p: dict(frames=len(files), hit_rate=v["hits"] / len(files), near_rate=v["near"] / max(v["hits"], 1),
                       median_score=float(np.median(v["scores"])) if v["scores"] else 0.0) for p, v in probe.items()}
        (args.out / "PROBE.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep), flush=True)
        return
    (args.out / "MASKS.json").write_text(json.dumps(dict(
        frames=len(files), empty=empty, prompts=args.prompt, revision=REV, min_area=args.min_area,
        max_area=args.max_area, max_gap=MAX_GAP, selector="best score across prompts", rows=rows), indent=1))
    print(f"DONE {args.frames.name}: {len(files)} frames, {empty} missed", flush=True)


if __name__ == "__main__":
    main()
