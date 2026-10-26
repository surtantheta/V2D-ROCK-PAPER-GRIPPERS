#!/usr/bin/env python3
"""Contact sheet: frames with the SAM3 object mask overlaid (red) for visual QA."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=Path, required=True)
ap.add_argument("--masks", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
ms = sorted(a.masks.glob("*.png"))
pick = [ms[int(i)] for i in np.linspace(0, len(ms) - 1, a.n)]
tiles = []
for m in pick:
    img = np.asarray(Image.open(a.frames / (m.stem + ".jpg")).convert("RGB")).astype(np.float32)
    mk = np.asarray(Image.open(m)) > 127
    img[mk] = 0.5 * img[mk] + 0.5 * np.array([255, 0, 0])
    t = Image.fromarray(img.astype(np.uint8)).resize((384, 288))
    tiles.append(t)
cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (384 * cols, 288 * rows))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 384, (i // cols) * 288))
a.out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(a.out)
print("wrote", a.out, [p.stem for p in pick])
