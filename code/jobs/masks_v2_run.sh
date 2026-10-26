#!/usr/bin/env bash
# usage: EPS="12 13" masks_v2_run.sh   (prompts chosen per object from work/masks_probe*/PROBE.json; see v2d_track1.md)
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
PY=/home/tanuj/envs/sam3/bin/python; C=/home/tanuj/V2D_challenge/code/masks_v2.py; W=/home/tanuj/V2D_challenge/work
for ep in ${EPS//_/ }; do E=$(printf "%06d" $ep); O=$W/masks_v2/episode_$E; [ -f $O/MASKS.json ] && continue
  CROPI="${CROP:-}"; NEAR=""
  case $ep in
    6|7|8) P=(--prompt "steam iron" --prompt "electric iron" --prompt "clothes iron"); CROPI=1 ;;
    9) P=(--prompt "overturned table"); NEAR=1 ;;
    10|11) P=(--prompt "overturned table" --prompt "white table" --prompt "table legs" --prompt "white desk"); NEAR=1 ;;
    12|13|14) P=(--prompt "black pan" --prompt "a black frying pan" --prompt "frying pan") ;;
    15|16|17) P=(--prompt "block" --prompt "foam block") ;;
    18|19|20) P=(--prompt "paint roller" --prompt "roller") ;;
    27|28|29) P=(--prompt "rolling white laptop desk with cup holder and basket" --prompt "white cart") ;;
    *) echo "no prompt set for $ep"; continue ;;
  esac
  X=(); [ -n "$CROPI" ] && X=(--person-boxes $W/s3db/episode_$E/sam3d_body.npz)
  [ -n "$NEAR" ] && X+=(--near-subject $W/s3db/episode_$E/sam3d_body.npz)
  echo "== episode $ep $(date -Is)"; $PY $C --frames $W/frames/episode_$E --out $O "${P[@]}" "${X[@]}"
done
