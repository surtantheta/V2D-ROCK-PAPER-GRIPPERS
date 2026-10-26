#!/usr/bin/env bash
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
PY=/home/tanuj/envs/sam3/bin/python; C=/home/tanuj/V2D_challenge/code/masks_v2.py; W=/home/tanuj/V2D_challenge/work
for ep in 6 7 8; do E=$(printf "%06d" $ep)
  $PY $C --frames $W/frames/episode_$E --out $W/masks_probe_crop/episode_$E --probe-every 25 --person-boxes $W/s3db/episode_$E/sam3d_body.npz \
     --prompt "steam iron" --prompt "iron" --prompt "clothes iron" --prompt "electric iron" --prompt "An metallic iron with blue accents"
done
