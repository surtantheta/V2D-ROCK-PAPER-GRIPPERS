#!/usr/bin/env bash
# usage: submit_object_chain.sh [-h] "<eps>"   submits dscale -> s5 (cvml04 RTX8000) -> s6 (A5000 hosts) ; s4b ; s7
# -h submits everything on user hold. Per-episode failures are isolated (stage jobs exit 0, work/STATUS markers).
set -euo pipefail
HOLD=""; [ "${1:-}" = "-h" ] && { HOLD="-h"; shift; }
EPS=($1); S=/home/tanuj/V2D_challenge/scripts; cd $S
A5000=(cvml03 cvml10 cvml12 cvml01)   # cvml05 GPU type unconfirmed (validator 04 #7)
D=$(qsub $HOLD -l select=1:ncpus=8:mem=48gb -v STAGE=dscale,EPS="${EPS[*]}" stage_cpu.pbs)
half=$(( (${#EPS[@]} + 1) / 2 )); P1="${EPS[*]:0:$half}"; P2="${EPS[*]:$half}"
S5=""; for P in "$P1" "$P2"; do [ -n "$P" ] || continue
  J=$(qsub $HOLD -W depend=afterany:$D -l select=1:ncpus=4:mem=18gb:ngpus=1:host=cvml04 -v STAGE=s5,EPS="$P" stage_gpu.pbs); S5="$S5:$J"; done
S6=""; S4=""; n=${#A5000[@]}
for k in $(seq 0 $((n-1))); do G=""; for idx in "${!EPS[@]}"; do [ $((idx % n)) -eq $k ] && G="$G ${EPS[$idx]}"; done
  [ -n "$G" ] || continue
  J=$(qsub $HOLD -W depend=afterany$S5 -l select=1:ncpus=4:mem=16gb:ngpus=1:host=${A5000[$k]} -v STAGE=s6,EPS="$G" stage_gpu.pbs); S6="$S6:$J"
  J=$(qsub $HOLD -l select=1:ncpus=4:mem=16gb:ngpus=1 -v STAGE=s4b,EPS="$G" stage_gpu.pbs); S4="$S4:$J"; done
S7=$(qsub $HOLD -W depend=afterany$S6$S4 -l select=1:ncpus=4:mem=16gb -v STAGE=s7,EPS="${EPS[*]}" stage_cpu.pbs)
echo "eps=[${EPS[*]}] dscale=$D s5=$S5 s6=$S6 s4b=$S4 s7=$S7"
