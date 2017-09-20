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