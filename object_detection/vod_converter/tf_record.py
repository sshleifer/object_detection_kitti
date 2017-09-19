import hashlib
import io

import PIL.Image
import os
import tensorflow as tf

from object_detection.create_dataset import label_map_dict
from object_detection.utils import dataset_util


class TensorflowEgestor(object):

    def expected_labels(self):
        return {}

    def egest(self, image_detections, root=None):
        writer = tf.python_io.TFRecordWriter(os.path.join(root, 'train.tfrecord'))
        for image_detection in image_detections:
            tfrecord = kitti_dict_to_tf_example(image_detection)
            writer.write(tfrecord.SerializeToString())
        writer.close()


def kitti_dict_to_tf_example(data):
    img = data['image']
    img_path = img['path']
    assert os.path.exists(img_path), img_path
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = img['width']
    height = img['height']

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['detections']:
        difficult_obj.append(obj.get('difficult', False))  # TODO(SS): not being parsed
        truncated.append(0)  # not parsed
        poses.append(obj.get('poses', 'Unspecified').encode('utf-8'))  # not parsed
        xmin.append(obj['left'] / width)
        ymin.append(obj['bottom'] / height)
        xmax.append(obj['right'] / width)
        ymax.append(obj['top'] / height)
        classes_text.append(obj['label'].lower().encode('utf8'))
        classes.append(label_map_dict[obj['label'].lower()])
    filename = os.path.basename(img_path).encode('utf8')
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example