Goal: Glue between tensorflow objection detection models and kitti-2d object detection data
Status: sort of works

- addition of config files to 
- scripts to fetchconvert the kitti 2D objection detection data to a tf friendly format
- all my code is in the `object_detection/` directory

### Instructions

- clone this
- cd into it
- `git clone git@github.com:umautobots/vod-converter.git`
    - if you are using python 2.7, `git clone git@github.com:nghiattran/vod-converter.git`
- `./fetch_kitti.sh` to pull down the kitti data and make `kitti_data` (takes a while!)
- `./make_tf_records.sh` to get into tensorflow format
- `./train_and_freeze.sh` to train ssd_mobilenet_1 on the data and free the inference graph.
- then use the kitti-inference notebook to inspect performance


### References

- `object_detection/vod_converter` is stolen from github.com/nghiattran/vod-converter with a few modifications
- tensorflow object detection: https://github.com/tensorflow/models/tree/master/object_detection


2D Object Detection Benchmark
=============================

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


Train Results
=============

PerformanceByCategory/mAP@0.5IOU/car               0.937117
PerformanceByCategory/mAP@0.5IOU/cyclist           0.828984
PerformanceByCategory/mAP@0.5IOU/dontcare          0.364170
PerformanceByCategory/mAP@0.5IOU/misc              0.800000
PerformanceByCategory/mAP@0.5IOU/pedestrian        0.766581
PerformanceByCategory/mAP@0.5IOU/person_sitting    0.875000
PerformanceByCategory/mAP@0.5IOU/tram              1.000000
PerformanceByCategory/mAP@0.5IOU/truck             0.914141
PerformanceByCategory/mAP@0.5IOU/van               0.878044
Precision/mAP@0.5IOU                               0.818226

Valid Results (794 valid images)
================================

PerformanceByCategory/mAP@0.5IOU/car               0.959948
PerformanceByCategory/mAP@0.5IOU/cyclist           0.846211
PerformanceByCategory/mAP@0.5IOU/dontcare          0.339320
PerformanceByCategory/mAP@0.5IOU/misc              0.844625
PerformanceByCategory/mAP@0.5IOU/pedestrian        0.792805
PerformanceByCategory/mAP@0.5IOU/person_sitting    0.670089
PerformanceByCategory/mAP@0.5IOU/tram              0.940657
PerformanceByCategory/mAP@0.5IOU/truck             0.943405
PerformanceByCategory/mAP@0.5IOU/van               0.936856
Precision/mAP@0.5IOU                               0.808213


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

