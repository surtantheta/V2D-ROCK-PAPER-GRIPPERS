#!/usr/bin/env bash
# probe candidate prompts on every 25th frame of the weak-mask episodes
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
PY=/home/tanuj/envs/sam3/bin/python; C=/home/tanuj/V2D_challenge/code/masks_v2.py; W=/home/tanuj/V2D_challenge/work
run() { ep=$1; shift; E=$(printf "%06d" $ep); $PY $C --frames $W/frames/episode_$E --out $W/masks_probe/episode_$E --probe-every 25 "$@"; }
run 6  --prompt "clothes iron" --prompt "iron" --prompt "steam iron" --prompt "electric iron" --prompt "An metallic iron with blue accents"
run 8  --prompt "clothes iron" --prompt "iron" --prompt "steam iron" --prompt "electric iron"
run 18 --prompt "paint roller" --prompt "roller" --prompt "paint roller with a blue handle" --prompt "white with yellow stipped paint roller with a blue handle"
run 13 --prompt "frying pan" --prompt "pan" --prompt "black pan" --prompt "a black frying pan"
run 15 --prompt "foam block" --prompt "green foam block" --prompt "cube" --prompt "block"
run 27 --prompt "laptop cart" --prompt "rolling cart" --prompt "white cart" --prompt "rolling white laptop desk with cup holder and basket"
