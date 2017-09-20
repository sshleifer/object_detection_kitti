#!/usr/bin/env bash
export PYTHONPATH="..:./../slim"
export TF_CPP_MIN_LOG_LEVEL=2.
rm -rf kitti_data/valid
python create_dataset.py && ./train.sh