#!/usr/bin/env bash
echo "== HF download"; HF_HUB_OFFLINE=0 /home/tanuj/envs/sam3/bin/python /home/tanuj/V2D_challenge/code/fetch_cari4d_commercial.py 2>&1 | tail -3
echo "== docker on $(hostname)"; which docker; docker info 2>&1 | grep -E "Server Version|Runtimes|Default Runtime|Rootless|ERROR|permission" | head
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -3
