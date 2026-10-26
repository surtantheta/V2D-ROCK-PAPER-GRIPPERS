# V2D Challenge Track 1 — v0 reconstruction pipeline

Monocular third-person video -> MHR human + object mesh + per-frame object pose, packed with the official
submission kit. Stages (per episode), driver `run_stage.sh`:

| stage | script | what |
|---|---|---|
| S1 | extract_frames.py | decode episode video to JPEG frames |
| S2 | (UniCon3R++ tools/wild_video_masks.py) | SAM 3 object masks from the episode's object prompt |
| S3 | metav3rse_human_infer.py | MetaV3RSE human/scene: focal estimate, subject track, camera-frame depth |
| S3b | sam3d_body_infer.py | SAM 3D Body per-frame MHR (subject bbox from S3 track) |
| dscale | depth_scale.py | per-window anchoring of S3 depth to the SAM 3D Body surface |
| S5 | object_mesh_sam3d.py | SAM 3D Objects mesh + keyframe pose on the anchored pointmap |
| S6 | object_track_fp.py | FoundationPose register/track/re-acquire |
| S4b | mhr_single_identity.py | kit mesh_to_mhr_params single-identity refit |
| S7 | assemble_episode.py | packer input npz + object mesh |
| QA | qa_overlay.py | project submission arrays (host decode) onto frames |

Frame convention: OpenCV camera frame, metres, for human (host decode diag(1,-1,-1) @ MHR / 100) and object.

## Reproducing the first submission (v1)

`jobs/` holds the PBS wrappers and drivers used for the run (`submit_object_chain.sh`, `stage_*.pbs`,
`masks_v2_run.sh`, `pack_full.sh`), the per-episode mask choice (`MASK_CHOICE.json`) and the per-camera focal
estimates (`FOCALS.json`). External models: MetaV3RSE (UniCon3R++) checkpoint, SAM 3, SAM 3D Body (dinov3),
SAM 3D Objects, FoundationPose (CARI4D-vendored), MHR lod1 TorchScript.
