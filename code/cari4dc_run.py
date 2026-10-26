#!/usr/bin/env python3
"""Run the toolkit's commercial CARI4D inference natively (no Docker): load v2d/cari4d/lib/run_inference.py with its two
hard-coded Docker paths (/workspace/v2d_sam3d_body/lib, /workspace/v2d_foundation_pose/lib/FoundationPose) replaced
by the local clone, then call its main() unchanged (same CLI as `python -m v2d.cari4d.docker.run_inference`)."""
import sys
import types
from pathlib import Path

M = Path("/home/tanuj/V2D_challenge/video_to_data/reconstruction/modules")
src = (M / "v2d_cari4d/lib/run_inference.py").read_text()
src = src.replace('"/workspace/v2d_sam3d_body/lib"', f'"{M}/v2d_sam3d_body/lib"')
src = src.replace('"/workspace/v2d_foundation_pose/lib/FoundationPose"', f'"{M}/v2d_foundation_pose/lib/FoundationPose"')
assert "/workspace" not in src, "unpatched /workspace path remains"
mod = types.ModuleType("v2d.cari4d.lib.run_inference_native")
mod.__file__ = str(M / "v2d_cari4d/lib/run_inference.py")
exec(compile(src, mod.__file__, "exec"), mod.__dict__)
sys.exit(mod.main())
