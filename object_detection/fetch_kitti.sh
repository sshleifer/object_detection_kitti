#!/usr/bin/env bash

wget http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip
wget http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
unzip data_object_image_2.zip
unzip data_object_label_2.zip
#mkdir kitti_data/training
#mkdir kitti_data/testing
mv data_object_image_2 kitti_data
mv data_object_image_2/training/label_2  kitti_data/training/label_2
# make train.txt

## desired tree
# kitti_data
#  -- training
#       -- img_2
#        -- label_2
# train.txt


# python 2.7 git clone git@github.com:nghiattran/vod-converter.git
# python 3.6+
python vod-converter/vod_converter/main.py --from kitti --from-path kitti_trunc --to voc --to-path voc_kitti