#!/usr/bin/env python3
"""S1: decode every Track 1 episode video to full-resolution JPEGs (frame t -> 00000t.jpg).

Frame count is checked against meta/episodes.jsonl; a mismatch fails the episode.
"""
import argparse
import json
from pathlib import Path

import av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    info = json.loads((args.track1 / "meta/info.json").read_text())
    lengths = {e["episode_index"]: e["length"] for e in map(json.loads, (args.track1 / "meta/episodes.jsonl").read_text().splitlines())}
    meta = {e["episode_index"]: e for e in map(json.loads, (args.track1 / "meta/episodes_metadata.jsonl").read_text().splitlines())}
    eps = args.episodes if args.episodes else sorted(lengths)
    for ep in eps:
        dst = args.out / f"episode_{ep:06d}"
        done = dst / "FRAMES.json"
        if done.exists():
            print(f"skip {ep}"); continue
        dst.mkdir(parents=True, exist_ok=True)
        vid = args.track1 / info["video_path"].format(episode_chunk=0, episode_index=ep, video_key=meta[ep]["video_key"])
        n = 0
        with av.open(str(vid)) as c:
            for frame in c.decode(c.streams.video[0]):
                frame.to_image().save(dst / f"{n:06d}.jpg", quality=args.quality)
                n += 1
        if n != lengths[ep]:
            raise SystemExit(f"episode {ep}: decoded {n} != meta {lengths[ep]}")
        done.write_text(json.dumps(dict(episode=ep, frames=n, source=str(vid), quality=args.quality)))
        print(f"episode {ep}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
