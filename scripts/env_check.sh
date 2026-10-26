#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
echo "== v2d"; /home/tanuj/.conda/envs/v2d/bin/python -c "import scipy,trimesh,fast_simplification;print('ok scipy',scipy.__version__)" 2>&1 | tail -1
echo "== sam3dbody"; cd /home/tanuj/V2D_challenge/video_to_data/reconstruction/modules/v2d_sam3d_body/lib && TORCH_HOME=/home/tanuj/V2D_challenge/models/sam3d_body/torch_home /home/tanuj/envs/sam3dbody/bin/python -c "
import sys; sys.path.insert(0,'.')
from sam_3d_body import load_sam_3d_body
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
print('ok sam3dbody imports')" 2>&1 | tail -3
echo "== cari4d"; cd /home/tanuj/projects/CARI4D && /home/tanuj/.conda/envs/cari4d/bin/python -c "
import sys; sys.path.insert(0,'.'); import cv2, trimesh, nvdiffrast.torch
from estimater import FoundationPose
print('ok cari4d FP import')" 2>&1 | tail -2
echo "== sam3d-objects"; /home/tanuj/.conda/envs/sam3d-objects/bin/python -c "import torch, trimesh, cv2; print('ok', torch.__version__)" 2>&1 | tail -1
