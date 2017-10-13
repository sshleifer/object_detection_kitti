#!/bin/bash
export PYTHONPATH="..:./../slim"
export TF_CPP_MIN_LOG_LEVEL=2.
python train.py \
    --logtostderr \
    --pipeline_config_path=samples/configs/faster_rcnn_inception_resnet_v2_atrous_kitti.config \
    --train_dir atrous_train_check/
