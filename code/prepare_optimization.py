#!/usr/bin/env python3
"""Prepare observation manifests for offline HOI optimization; no GPU or writes to v0."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read_record(path, sources):
    raw = path.read_bytes()
    data = json.loads(raw)
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return data


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def spaced_candidates(rows, episode_dir, minimum_score=0.8, separation=30, limit=6):
    candidates = []
    for row in rows:
        if not row.get("selected") or not finite_positive(row.get("area")):
            continue
        score = row.get("score")
        if not finite_positive(score) or score < minimum_score:
            continue
        frame = int(Path(row["frame"]).stem)
        mask_path = Path(row["mask"])
        if not mask_path.is_absolute():
            mask_path = episode_dir / mask_path
        if not mask_path.is_file():
            continue
        candidates.append((float(row["area"]) * float(score), frame, row, mask_path))
    picked = []
    for _, frame, row, mask_path in sorted(candidates, key=lambda x: (-x[0], x[1])):
        if all(abs(frame - other["frame"]) >= separation for other in picked):
            picked.append({"frame": frame, "mask": str(mask_path), "area": row["area"], "score": row["score"]})
        if len(picked) >= limit:
            break
    return picked


def prepare(root):
    sources = {}
    verified = read_record(root / "logs/track1_verify_report.json", sources)
    episodes = []
    focal_groups = {}
    by_object = {}
    pilot_ids = {0, 3, 6, 15, 24, 27}
    for meta in verified["episodes"]:
        ep = meta["episode"]
        name = f"episode_{ep:06d}"
        frame_count = meta["length_meta"]
        scored = verified["sample_submission"]["per_episode"][str(ep)]
        blockers = []
        frames_marker = root / "work/frames" / name / "FRAMES.json"
        if not frames_marker.is_file():
            blockers.append("frames_incomplete")
        else:
            frames = read_record(frames_marker, sources)
            if frames.get("frames") != frame_count:
                blockers.append("frame_count_mismatch")
        mask_dir = root / "work/masks" / name
        mask_marker = mask_dir / "MASKS.json"
        mask_stats = {"state": "pending", "frames": None, "empty": None, "empty_fraction": None}
        keyframes = []
        if mask_marker.is_file():
            mask = read_record(mask_marker, sources)
            rows = mask.get("rows", [])
            frame_ids = [int(Path(row["frame"]).stem) for row in rows]
            actual_empty = sum(not bool(row.get("selected")) for row in rows)
            mask_stats = {"state": "complete", "frames": mask.get("frames"), "empty": actual_empty,
                          "empty_fraction": actual_empty / len(rows) if rows else None,
                          "prompts": mask.get("prompts"), "min_area": mask.get("min_area"),
                          "declared_empty": mask.get("empty")}
            if mask.get("frames") != frame_count or frame_ids != list(range(frame_count)):
                blockers.append("mask_frame_coverage_mismatch")
            if mask.get("empty") != actual_empty:
                blockers.append("mask_summary_mismatch")
            if not rows or actual_empty == len(rows):
                blockers.append("all_object_masks_missing")
            keyframes = spaced_candidates(rows, mask_dir)
            if not keyframes:
                blockers.append("no_confident_keyframe")
            mask_stats["first_scored_selected"] = bool(rows[scored["first"]].get("selected")) if len(rows) > scored["first"] else False
        else:
            blockers.append("masks_incomplete")
        focal_path = root / "work/s3_focal" / name / "FOCAL.json"
        focal = None
        if focal_path.is_file():
            focal_record = read_record(focal_path, sources)
            focal = focal_record.get("focal_source_px")
            if not finite_positive(focal):
                blockers.append("invalid_focal")
                focal = None
            elif focal_record.get("source_wh") != [meta["width"], meta["height"]]:
                blockers.append("focal_resolution_mismatch")
                focal = None
            else:
                group = (meta["camera"], meta["width"], meta["height"])
                focal_groups.setdefault(group, []).append({"episode": ep, "focal_source_px": focal})
        else:
            blockers.append("focal_pending")
        human_marker = root / "work/s3_human" / name / "RUN.json"
        human_complete = False
        if human_marker.is_file():
            human = read_record(human_marker, sources)
            human_complete = human.get("frames") == frame_count
        if not human_complete:
            blockers.append("scene_inference_pending")
        observed_frames = {k["frame"] for k in keyframes}
        validation_frames = sorted({0, frame_count - 1, scored["first"], scored["last"],
                                    *[round(i * (frame_count - 1) / 12) for i in range(13)]} - observed_frames)
        entry = {
            "episode": ep, "object": meta["object"], "camera": meta["camera"],
            "sequence_id": meta["sequence_id"], "frames": frame_count,
            "first_scored_frame": scored["first"], "last_scored_frame": scored["last"],
            "scored_frame_count": scored["n_scored_frames"], "masks": mask_stats,
            "focal_source_px": focal, "scene_run_marker_complete": human_complete,
            "proposal_keyframes": keyframes, "validation_frame_candidates": validation_frames,
            "blockers": blockers, "pilot": ep in pilot_ids,
            "ready_for_optimizer": False,
            "remaining_gates": ["manual_instance_and_visibility_QA", "validated_MHR_identity_and_first_scored_frame",
                                "accepted_mesh_proposal", "depth_and_pose_observability", "optimizer_GPU_smoke"],
        }
        episodes.append(entry)
        by_object.setdefault(meta["object"], []).append(ep)
    pooled = []
    for (camera, width, height), values in sorted(focal_groups.items()):
        fs = [v["focal_source_px"] for v in values]
        median = statistics.median(fs)
        spread = (max(fs) - min(fs)) / median
        pooled.append({"camera": camera, "source_wh": [width, height], "episodes": values,
                       "candidate_focal_source_px": median,
                       "mad_px": statistics.median(abs(f - median) for f in fs),
                       "range_fraction": spread, "review_required": len(fs) < 3 or spread > 0.05,
                       "status": "diagnostic_candidate_not_calibrated_intrinsics"})
    return {
        "schema": "v2d-optimization-preparation-v1",
        "created_sgt": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
        "root": str(root), "source_sha256": sources,
        "scope": "Cached metadata audit; does not inspect mask pixels, validate depth, or certify optimizer readiness.",
        "summary": {"episodes": len(episodes),
                    "completed_mask_manifests": sum(e["masks"]["state"] == "complete" for e in episodes),
                    "all_masks_missing_episodes": [e["episode"] for e in episodes if "all_object_masks_missing" in e["blockers"]],
                    "focal_estimates": sum(e["focal_source_px"] is not None for e in episodes),
                    "scene_run_markers": sum(e["scene_run_marker_complete"] for e in episodes),
                    "pilot_episodes": sorted(pilot_ids)},
        "shared_shape_candidates": [{"object": obj, "episodes": ids, "enabled": False,
                                     "gate": "Verify physical shape identity and rigidity from RGB before sharing a mesh."}
                                    for obj, ids in sorted(by_object.items())],
        "camera_focal_candidates": pooled, "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths are refused.")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    report = prepare(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "PREPARATION.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.out / "EPISODES.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "object", "camera", "first_scored", "mask_empty_fraction", "focal_source_px", "proposal_frames", "blockers"])
        for ep in report["episodes"]:
            writer.writerow([ep["episode"], ep["object"], ep["camera"], ep["first_scored_frame"],
                             ep["masks"]["empty_fraction"], ep["focal_source_px"],
                             " ".join(str(k["frame"]) for k in ep["proposal_keyframes"]), " ".join(ep["blockers"])])
    print(json.dumps(report["summary"], indent=2))
    print(f"Prepared {args.out}")


if __name__ == "__main__":
    main()
