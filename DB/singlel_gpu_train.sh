#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--config_file "config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml"