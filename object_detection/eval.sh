#!/bin/bash
source tensorflow_env.sh
export PYTHONPATH="..:./../slim"
export TF_CPP_MIN_LOG_LEVEL=2.
echo $1
echo $2
python eval.py \
    --logtostderr \
    --checkpoint_dir=$1/ \
    --eval_dir=kitti_eval/$1 \
    --pipeline_config_path=$2
