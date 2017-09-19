#!/bin/bash
source tensorflow_env.sh
export PYTHONPATH="..:./../slim"
export TF_CPP_MIN_LOG_LEVEL=2.
python eval.py \
    --logtostderr \
    --checkpoint_dir=kitti_mobilenet/ \
    --eval_dir=kitti_eval \
    --pipeline_config_path=samples/configs/ssd_mobilenet_v1_kitti.config
