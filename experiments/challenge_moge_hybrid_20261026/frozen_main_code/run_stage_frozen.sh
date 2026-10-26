#!/usr/bin/env bash
# usage: run_stage.sh <stage> <episode>    stages: s3db s4b s4mv s5 s6 s7
set -euo pipefail
STAGE=$1; EP=$2
ROOT=/home/tanuj/V2D_challenge
EP=$((10#$EP)); E=$(printf "%06d" $EP)
W=${W:-$ROOT/work}                      # override for the labeled-proxy run (work_labeled)
export PYTHONNOUSERSITE=1
focal() { python3 -c "
import json;print(json.load(open('$W/s3_human/episode_$E/RUN.json'))['focal_source_px'])"; }
DS=$W/depth_scale/episode_$E.npz
# per-episode mask version from work/MASK_CHOICE.json (validator 04 #4: v2 is not uniformly better); default v1
MV=$(python3 -c "import json,os;p='$W/MASK_CHOICE.json';print(json.load(open(p)).get('$EP','v1') if os.path.exists(p) else 'v1')")
if [ "$MV" = v2 ]; then MK=$W/masks_v2/episode_$E; else MK=$W/masks/episode_$E; fi
# validator 04 #3: never let a failed stage leave stale outputs for downstream stages
case $STAGE in
  dscale) rm -f $DS ${DS%.npz}.json ;;
  s5) rm -rf $W/s5/episode_$E ;;
  s6) rm -rf $W/s6/episode_$E ;;
  s4b) rm -rf $W/s4b/episode_$E ;;
  s7) rm -f $W/${SUBDIR:-submission_v1}/episode_$E.npz $W/${SUBDIR:-submission_v1}/episode_${E}_object.ply $W/${SUBDIR:-submission_v1}/ASSEMBLE_$E.json ;;
esac
[ -f $MK/MASKS.json ] || { echo "masks $MV missing for $EP"; [ "$STAGE" = s3full ] || [ "$STAGE" = s3db ] || [ "$STAGE" = s4b ] || exit 5; }
echo "[run_stage] $STAGE ep$EP masks=$MV"

case $STAGE in
  s3full) F=$(python3 -c "import json;print(json.load(open('$W/FOCALS.json'))['episode_focal_px']['$EP'])")
         [ -f $W/frames/episode_$E/FRAMES.json ] || { echo "frames missing"; exit 3; }
         cd /home/tanuj/UniCon3Rpp-push-b3c6624
         PYTHONPATH=/home/tanuj/UniCon3Rpp-push-b3c6624/src:/home/tanuj/UniCon3Rpp-push-b3c6624/src/croco \
         /home/tanuj/.conda/envs/human3r_new/bin/python $RUN_CODE/metav3rse_human_infer.py --frames $W/frames/episode_$E \
           --ckpt /rhome/tanuj/unicon3rpp_runs/U3PP-Gates/M10MeshQueryD1/spatial-gtcrop25/checkpoint-25.pth \
           --out $W/s3_human/episode_$E --mode full --focal $F --window ${WINDOW:-16} ;;
  s3db)  F=$(focal); /home/tanuj/envs/sam3dbody/bin/python $RUN_CODE/sam3d_body_infer.py --frames $W/frames/episode_$E \
           --human $W/s3_human/episode_$E --out $W/s3db/episode_$E --focal $F ;;
  s4b)   /home/tanuj/.conda/envs/v2d/bin/python $RUN_CODE/mhr_single_identity.py --sam3d-body $W/s3db/episode_$E/sam3d_body.npz \
           --out $W/s4b/episode_$E --smooth-sigma-xy ${SMOOTH_XY:-2.0} --smooth-sigma-z ${SMOOTH_Z:-5.0} ;;
  s4mv)  /home/tanuj/.conda/envs/v2d/bin/python $RUN_CODE/smplx_to_mhr.py --human $W/s3_human/episode_$E --out $W/s4mv/episode_$E ;;
  dscale) /home/tanuj/.conda/envs/v2d/bin/python $RUN_CODE/depth_scale.py --human $W/s3_human/episode_$E \
           --s3db $W/s3db/episode_$E --masks $MK --out $DS ;;
  s5)    F=$(focal); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/home/tanuj/.conda/envs/sam3d-objects/bin:$PATH CONDA_PREFIX=/home/tanuj/.conda/envs/sam3d-objects CUDA_HOME=/home/tanuj/.conda/envs/sam3d-objects
         cd /rhome/tanuj/projects/sam-3d-objects
         /home/tanuj/.conda/envs/sam3d-objects/bin/python $RUN_CODE/object_mesh_sam3d.py --frames $W/frames/episode_$E \
           --masks $MK --human $W/s3_human/episode_$E --focal $F --out $W/s5/episode_$E --depth-scale $DS ;;
  s6)    cd /home/tanuj/projects/CARI4D
         /home/tanuj/.conda/envs/cari4d/bin/python $RUN_CODE/object_track_fp.py --frames $W/frames/episode_$E \
           --masks $MK --human $W/s3_human/episode_$E --object $W/s5/episode_$E --out $W/s6/episode_$E --depth-scale $DS ;;
  s7)    /home/tanuj/.conda/envs/v2d/bin/python $RUN_CODE/assemble_episode.py --episode $EP --human $W/s4b/episode_$E \
           --object $W/s5/episode_$E --track $W/s6/episode_$E --masks $MK --s3db $W/s3db/episode_$E --s3-human $W/s3_human/episode_$E --depth-scale $DS --out $W/${SUBDIR:-submission_v1} ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
