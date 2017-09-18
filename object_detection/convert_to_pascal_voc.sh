#!/usr/bin/env bash
source tensorflow_env.sh
python vod-converter/vod_converter/main.py \
    --from kitti --from-path kitti_data  \
    --to voc --to-path voc_kitti


# Cloning instructions in README
# python 2.7 git clone git@github.com:nghiattran/vod-converter.git
# python 3.6+