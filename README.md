Goal: Glue between tensorflow objection detection models and kitti-2d object detection data
Status: doesn't work yet

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