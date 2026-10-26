# Two Track 1 candidates prepared October 26, 2026

The payloads are complete. **Files under `pending_commit/` are not for upload:**
their `code_commit_url` still names the clearly marked placeholder repository.
Use the finalization command below after publishing the prepared source commit.
No Kaggle submission or GitHub publication has been performed by this preparation.

| Candidate | Pending file | Meaning |
|---|---|---|
| Main v2.1 | `pending_commit/track1_main_v21.parquet` | Main S1–S7 reconstruction with validated v2.1 depth repair |
| Optimization alternative | `pending_commit/track1_optimization_shared_scale_contact.parquet` | Existing shared-scale/contact fit on main test artifacts, starting from raw S6 translation and keeping v2.1 rotation |

Each file has the public sample's exact 740,780 rows across all 30 episodes.
Both carry the same human, object mesh and object rotations; object translations
and clip-level object scales differ. The branch file is **not** the original
MoGe-based development pipeline, and its development scores do not describe
this parquet. It regresses on some test-data geometry checks. Uploading both
provides a controlled hidden-GT comparison, not a demonstrated improvement.

`PREPARATION_MANIFEST.json` records payload hashes, content hashes excluding the
commit column, source hashes, coverage, scale, camera validity and exact roundtrip
checks. `pack_*.log` records the official packer. The offline check calls the
official uploader's `validate_submission`, the scorer's `_to_arrays`, `_unpack`,
`_object_scale` and `_rotations`, and compares every scored source array and
budgeted mesh. It does not compute hidden-GT scores.

## What you need to change

1. Publish the prepared **`source_repository`** Git repository to your GitHub
   repository. It includes the branch adapter AND its frozen solver dependencies;
   the old main-pipeline commit alone is insufficient for the branch candidate.
   See `SOURCE_COMMIT.txt` for the exact commit. For a private repository, give
   `Nvidia-V2D-Challenge` access to that commit.
2. Set `V2D_COMMIT_URL` to the resulting real GitHub commit URL and run:

```bash
V2D_COMMIT_URL='https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY/commit/FULL_SHA_FROM_SOURCE_COMMIT_TXT'
/home/tanuj/.conda/envs/v2d/bin/python -I \
  /home/tanuj/V2D_challenge/submissions/upload_prep_20261026/prepare_uploads.py \
  --commit-url "$V2D_COMMIT_URL"
```

This changes only `code_commit_url`, verifies unchanged numeric content, and
validates both new files in **`ready/`**. Never edit the `x/y/z` or row IDs to
replace the URL. No models need rerunning. The finalizer requires the SHA of
the prepared source snapshot, rejects placeholder/all-zero commit values,
and leaves the existing experimental inputs and pending files unchanged.
It validates locally; you must ensure the remote commit exists and organizers
can actually inspect it.

Example publication commands (you supply a repository you control):

```bash
git -C /home/tanuj/V2D_challenge/submissions/upload_prep_20261026/source_repository \
  remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git -C /home/tanuj/V2D_challenge/submissions/upload_prep_20261026/source_repository \
  push -u origin main
```

## Before uploading

Authenticate your Kaggle account and accept the rules for all five Track 1
competitions listed in the kit's `LEADERBOARDS.md`. From the kit root, run the
official uploader with `--dry-run` for each finalized file. This checks account
entry and quota without spending a submission:

No standard Kaggle API credential file or credential environment variables were
present during preparation. Configure credentials locally if using the CLI;
do not put them into the source repository or send them in chat. Browser upload
is also possible after accepting rules: upload the same finalized candidate
parquet to each of the five Track 1 metric competitions. An API token is only
needed for the CLI route.

```bash
cd /home/tanuj/V2D_challenge/data/assets/submission_kit/v2d_submission_kit
/home/tanuj/.conda/envs/v2d/bin/python submit/v2d_submit.py \
  /home/tanuj/V2D_challenge/submissions/upload_prep_20261026/ready/track1_main_v21.parquet \
  --track track_1 -m 'Main v2.1' --dry-run
/home/tanuj/.conda/envs/v2d/bin/python submit/v2d_submit.py \
  /home/tanuj/V2D_challenge/submissions/upload_prep_20261026/ready/track1_optimization_shared_scale_contact.parquet \
  --track track_1 -m 'Shared-scale contact alternative' --dry-run
```

To actually upload, replace `--dry-run` with `--wait`. Each command sends the
same candidate file to **all five** metric competitions: CD-H, CD-O, ACC-H,
ACC-O and PEN. The two candidates are separate submissions. Preserve the
receipts printed by the uploader. Consider the team's remaining weekly quota
before uploading both.

Official instructions:
https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#submission

Local kit instructions:
`data/assets/submission_kit/v2d_submission_kit/tracks/track_1/README.md`.
