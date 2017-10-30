#!/bin/bash
export PYTHONPATH="..:./../slim"
export TF_CPP_MIN_LOG_LEVEL=2.
python train.py \
    --logtostderr \
    --pipeline_config_path=samples/configs/ssd_mobilenet_v1_kitti.config \
    --train_dir kitti_mobilenet/
