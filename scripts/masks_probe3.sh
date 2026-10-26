#!/usr/bin/env bash
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
PY=/home/tanuj/envs/sam3/bin/python; C=/home/tanuj/V2D_challenge/code/masks_v2.py; W=/home/tanuj/V2D_challenge/work
for ep in 9 11; do E=$(printf "%06d" $ep)
  $PY $C --frames $W/frames/episode_$E --out $W/masks_probe_desk/episode_$E --probe-every 20 --near-subject $W/s3db/episode_$E/sam3d_body.npz \
     --prompt "table" --prompt "white table" --prompt "folding table" --prompt "overturned table" --prompt "desk" --prompt "white desk" --prompt "table legs"
done
