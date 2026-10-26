# Independent upload preparation audit — 2026-10-26

Status: both prepared Parquet payloads PASS independent validation; publication and a real commit URL remain pending. No publishing, submission, inference, fitting, or score-based arm selection was performed.

Package `work/branchfit/shared_scale_contact` as the existing optimization candidate alongside the main v2.1 pack. This preserves the documented candidate; it does not establish that the branch wins on these unlabeled test episodes. These 30 branch outputs use the main pipeline's MetaV3RSE depth, raw S6 translations and v2.1 rotations. They are not the independently generated MoGe-based development pipeline that obtained the earlier InterCap result. The canonical record documents known sparse-mask/contact regressions, particularly episodes 8 and 18.

## Contract from the installed submission kit

- One Parquet, exact columns `row_id,x,y,z,code_commit_url`, exact 740,780 sample row IDs including order, no duplicates, finite values.
- All 30 episodes and 9,877 scored frames; human pose 136 parameters, scales 68 and shape 45; rotations 3×3 per frame, translations in metres, one positive object scale per episode.
- Object coordinates obey `X_scene = object_scale * (V_local @ R.T) + t`. The kit budgets meshes to 4,096 vertex rows and 4,096 face rows, preserving their local coordinate origin. Faces must be integral and in range; padding can be degenerate.
- One identical GitHub `/commit/<sha>` URL on every row. The commit must be pushed and inspectable; private repositories must grant `Nvidia-V2D-Challenge` access. The placeholder URL passes the local regex but is not valid provenance.
- The same Parquet is used for all five Track 1 competitions. This audit does not authorize or perform an upload.

Sources: `data/assets/submission_kit/v2d_submission_kit/tracks/track_1/README.md`, `tools/pack_reconstruction.py`, `v2dlb/mesh_budget.py`, `v2dlb/report_schema.py`. MCP graph lookup found no relevant packaging code in its indexed experiment project; targeted source reads were used.

## Independent saved-output checks

All 30 branch NPZ/mesh pairs pass over all 16,563 source frames. Human pose, human shape/scales, object rotations and mesh bytes equal `work/submission_v2` exactly. Every main scale equals 1.0; branch scales equal the corresponding `FIT_XXXXXX.json` shared-scale/contact ratio exactly. There are no whole-case fallbacks and no interpolated invalid S6 frames recorded in these fits.

All arrays have the expected episode lengths/shapes and finite values. Maximum rotation orthogonality error is 6.5197985e-6, minimum determinant 0.999995745, branch scale range 0.853376085–1.160521515, and minimum camera Z across every full-resolution mesh vertex and every frame is 2.009653209 m. Recomputed minima match the FIT records within 1e-9 m. Source meshes have about 20,000 faces and will be budgeted by the kit.

Do not reset branch `object_scale` to 1, multiply the fitted translation by scale, recenter/renormalize the mesh, or interpret the stored translation as an area centroid. The frozen fitter already compensates the original mesh origin when exporting translation. Keep the scalar, translation and original mesh together; the official packer handles the geometry budget. Do not introduce the development scorer's human-GT Sim3 into either submission.

The existing main `work/pack_v21_test/track1.parquet` independently has 740,780 rows, exact sample ordering and finite XYZ. SHA256: `1f8423517e77121ae4396bbcc9a7fe1dd697fc3805ce0e39b05c3280e77ce0e9`. Its sole commit value remains `https://github.com/placeholder/placeholder/commit/0000000000000000000000000000000000000000`.

## Required changes before upload

1. Publish the prepared reproducible source snapshot. The original `code/` repository has no remote and its old commit omits the untracked branch adapter and external solver dependencies. The new isolated `source_repository/` includes those exact sources; use its final commit rather than the old main-only commit. Private repositories must grant organizer access.
2. Replace the placeholder commit column in both prepared Parquets with the corresponding real pushed commit URL; one shared commit is possible if it contains both pipelines. Repack or rewrite only that column and preserve a numeric content digest.
3. Run the final local structural/input-equivalence check after that replacement. Pushed-commit accessibility is a separate check from regex validity.

These are provenance/packaging requirements. The saved branch NPZ arrays and mesh scale/origin pairing do not need modification.

## Episode source checks

| Episode | Frames | Scale | Enabled masks | Contact frames | Minimum full-mesh Z (m) |
|---:|---:|---:|---:|---:|---:|
| 0 | 790 | 1.022815987 | 729 | 40 | 3.113110 |
| 1 | 668 | 1.052399899 | 662 | 58 | 2.389578 |
| 2 | 866 | 0.853376085 | 847 | 64 | 2.498276 |
| 3 | 592 | 1.039215501 | 524 | 49 | 2.496701 |
| 4 | 747 | 0.931707296 | 721 | 72 | 2.641350 |
| 5 | 668 | 1.160521515 | 663 | 58 | 2.710581 |
| 6 | 816 | 1.040628614 | 124 | 11 | 3.098637 |
| 7 | 777 | 0.993975605 | 410 | 14 | 2.723424 |
| 8 | 634 | 1.011412419 | 27 | 4 | 3.205470 |
| 9 | 415 | 1.000000000 | 0 | 0 | 2.767552 |
| 10 | 577 | 0.975122894 | 577 | 31 | 2.266508 |
| 11 | 877 | 0.975838722 | 763 | 52 | 2.009653 |
| 12 | 405 | 0.973688573 | 132 | 7 | 2.855308 |
| 13 | 425 | 0.998491504 | 18 | 1 | 2.962598 |
| 14 | 442 | 0.989153796 | 169 | 7 | 2.919069 |
| 15 | 501 | 1.014801948 | 190 | 16 | 3.007659 |
| 16 | 360 | 0.994017777 | 178 | 17 | 3.030194 |
| 17 | 419 | 0.993580871 | 187 | 8 | 2.963993 |
| 18 | 535 | 1.009715385 | 13 | 1 | 3.280492 |
| 19 | 443 | 1.003650460 | 9 | 0 | 2.840386 |
| 20 | 549 | 1.017983770 | 63 | 2 | 2.524504 |
| 21 | 563 | 0.962255650 | 440 | 19 | 2.372474 |
| 22 | 533 | 0.992495951 | 532 | 61 | 2.681036 |
| 23 | 552 | 0.989706375 | 326 | 20 | 2.949158 |
| 24 | 420 | 0.917136552 | 420 | 8 | 2.608663 |
| 25 | 365 | 1.014848458 | 352 | 26 | 2.093979 |
| 26 | 399 | 0.910244748 | 388 | 1 | 2.847773 |
| 27 | 440 | 0.976917582 | 108 | 13 | 2.506057 |
| 28 | 366 | 1.026097502 | 189 | 11 | 3.237305 |
| 29 | 419 | 0.971346397 | 406 | 26 | 2.894269 |

## Final prepared Parquets

Both prepared files PASS independent offline validation. Exact 740,780 sample row IDs/order, 9,877 scored frames, finite values, all 30 NPZ human/rotation/translation/scale values, exact official kit mesh-budget outputs, zero parameter padding, valid proper rotations, integral face indices, and 120 source hashes across the pair are verified. The official uploader's local validator passes without importing Kaggle or authenticating. Human parameters, mesh vertices/faces and rotations agree exactly between the candidates; only object translation and scale differ.

- `/home/tanuj/V2D_challenge/submissions/upload_prep_20261026/pending_commit/track1_main_v21.parquet`
  - File SHA256: `1f8423517e77121ae4396bbcc9a7fe1dd697fc3805ce0e39b05c3280e77ce0e9`.
  - Content SHA256 excluding commit: `36feaed7204e75c3cb53d94345e261087e62b2568b2dd347142081c87cdbf596`.
  - Minimum camera Z over packed meshes at scored frames: 1.909045765 m.
- `/home/tanuj/V2D_challenge/submissions/upload_prep_20261026/pending_commit/track1_optimization_shared_scale_contact.parquet`
  - File SHA256: `ecc5be08b9d4cc99dad7e93a9524633f9a117ad594d17560c5b909d4c868bec1`.
  - Content SHA256 excluding commit: `078ebe045ce820a7fad86a934a363ef720a5c863243c8b6e2a3021dcf921a5a9`.
  - Minimum camera Z over packed meshes at scored frames: 2.061715293 m.

Both still contain the explicit placeholder commit URL and remain **PAYLOAD VALID / NOT UPLOAD READY**. No push, authentication, dry-run against Kaggle, or submission occurred.

The isolated `submissions/upload_prep_20261026/source_repository/` contains the current main code, untracked branch adapter, all 92 Python files from the frozen branch code directory, configurations, and per-episode assembly/fit provenance. All 243 archived payload hashes match its source manifest; every live source other than this report (which changed during validation) also matches. Critical dependency coverage includes `run_identity_scale_v2.py`, `centroid_scale_trajectory_v2.py`, `constrained_trajectory_v1.py`, and `run_constrained_intercap_v1.py`. The packager will refresh this report's archived copy/manifest entry after this report freeze and create the final local commit. No original main repository files were modified by this audit.

The offline finalizer rejects placeholder/all-zero URLs and URLs naming a different SHA from the prepared source repository HEAD. It changes only `code_commit_url`, verifies unchanged numeric content and input hashes, and validates both resulting files. It deliberately does not verify publication or organizer access: the supplied URL must name the actually pushed commit. No code or model rerun is needed for this metadata change. The source snapshot documents historical absolute paths and external licensed checkpoints; it is not a standalone model distribution.

Independent validator report freeze: 2026-10-26. No further report changes are planned before the final source snapshot is committed.

