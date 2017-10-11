#!/usr/bin/env bash
#source tensorflow_env.sh
wget http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip
wget http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
unzip -q data_object_image_2.zip
unzip -q data_object_label_2.zip
# this didnt work on ubuntu, instead did jar xvf 
rm -rf kitti_data
mkdir kitti_data
mv training kitti_data
mv testing kitti_data
#python create_dataset.py
#./train.sh
#mkdir kitti_data/training
#mkdir kitti_data/testing
#mv data_object_image_2 kitti_data
#mv data_object_image_2/training/label_2  kitti_data/training/label_2
# make train.txt

## desired tree
# kitti_data
#  -- training
#       -- img_2
#        -- label_2
# train.txt


