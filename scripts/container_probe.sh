#!/usr/bin/env bash
echo "host=$(hostname) groups=$(id -Gn)"
echo "== docker"; docker info 2>&1 | grep -E "Server Version|Runtimes|permission" | head -3
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -2
echo "== podman"; which podman && podman --version && podman info 2>&1 | grep -E "rootless|graphRoot|runRoot" | head -4
podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -3
echo "== apptainer/singularity"; which apptainer singularity 2>&1 | head -2
env | grep -iE "^PBS_CONTAINER|DOCKER" | sed 's/=.*/=<set>/'
