from object_detection.create_pascal_tf_record import dict_to_tf_example
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf


writer = tf.python_io.TFRecordWriter('data/train.tfrecord')
label_map_dict = label_map_util.get_label_map_dict('data/kitti_map.pbtxt')
annotations_dir = '/Users/shleifer/voc_kitti/VOC2012/Annotations/'
examples_path = '/Users/shleifer/voc_kitti/VOC2012/ImageSets/Main/trainval.txt'
dataset_directory = 'data/'


def create_records(examples_path):
    labels = {}
    examples_list = dataset_util.read_examples_list(examples_path)
    for i, example in enumerate(examples_list[1:]):
        # TODO(SS): why is first example screwed up?
        path = os.path.join(annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        
        labels[i] = [k['name'] for k in data['object']]
        tf_example = dict_to_tf_example(data, 
                                        dataset_directory,
                                        label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    return labels  # to inspect a bit
