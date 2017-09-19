#!/usr/bin/env bash
source tensorflow_env.sh
python vod_converter/main.py \
    --from kitti --from-path kitti_data  \
    --to voc --to-path voc_kitti
python vod_converter/main.py \
    --from kitti --from-path kitti_data  --train_ids kitti_data/valid.txt \
    --to voc --to-path voc_kitti_valid
    #--to tensorflow --to-path data


# Cloning instructions in README
# python 2.7 git clone git@github.com:nghiattran/vod-converter.git
# python 3.6+