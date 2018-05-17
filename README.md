***THIS PROJECT IS NO LONGER BEING MANTAINED. The Official Object Detection repo now supports kitti!***

Goal: Glue between tensorflow objection detection models and kitti-2d object detection data
Status: probably won't work out of the box but will save you time vs googling all over

- scripts to fetch convert the kitti 2D objection detection data to TFRecords
- all my code is in the `object_detection/` directory
- more on how stuff works can be found at

### Incomplete List of Dependencies:

    - Download pretrained faster-rcnn https://medium.com/r/?url=http%3A%2F%2Fdownload.tensorflow.org%2Fmodels%2Fobject_detection%2Ffaster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
    - tensorflow
    - a GPU
    - have not tested on anything besides Ubuntu 16.05


### Instructions after cloning

```
cd object_detection
./fetch_kitti.sh  # uncomment python create_dataset.py, or run separately if you get into trouble
./train_rcnn.sh
# open a separate shell and run
./eval.sh rcnn_logs samples/configs/faster_rcnn_inception_resnet_v2_atrous_kitti.config# open yet a third shell and run
tensorboard --logdir rcnn_logs
# go to sleep...in the morning,
./freeze.sh samples/configs/faster_rcnn_inception_resnet_v2_atrous_kitti.config faster_rcnn_logs/model.ckpt-431399  faster_rcnn_frozen
jupyter notebook
# find kitti_inference.ipynb and try to figure out what is going on
```



### References

- `object_detection/vod_converter` is shamelessly stolen from github.com/nghiattran/vod-converter with a few modifications
- tensorflow object detection: https://github.com/tensorflow/models/tree/master/object_detection


2D Object Detection Benchmark Overview (from KITTI)
===================================================

The goal in the 2D object detection task is to train object detectors for the
classes 'Car', 'Pedestrian', and 'Cyclist'. The object detectors must
provide as output the 2D 0-based bounding box in the image using the format
specified above, as well as a detection score, indicating the confidence
in the detection. All other values must be set to their default values
(=invalid), see above. One text file per image must be provided in a zip
archive, where each file can contain many detections, depending on the
number of objects per image. In our evaluation we only evaluate detections/
objects larger than 25 pixel (height) in the image and do not count 'Van' as
false positives for 'Car' or 'Sitting Person' as false positive for 'Pedestrian'
due to their similarity in appearance. As evaluation criterion we follow
PASCAL and require the intersection-over-union of bounding boxes to be
larger than 50% for an object to be detected correctly.



Validation Results (794 valid images, 6900 train images)
========================================================
Category          mAP@0.5IOU
car               0.959948
cyclist           0.846211
dontcare          0.339320
misc              0.844625
pedestrian        0.792805
person_sitting    0.670089
tram              0.940657
truck             0.943405
van               0.936856
Total             0.808213


SSD Mobilenet Valid Results
===========================
Category          mAP@0.5IOU
car               0.723661
cyclist           0.390498
dontcare          0.073786
misc              0.493499
pedestrian        0.257245
person_sitting    0.573592
tram              0.800318
truck             0.641025
van               0.579114
Total             0.503638


Final Total Loss
================

rcnn    0.474066  43 hours   2.90 steps per second
ssd     2.778544  127 hours  1.15 steps per second

