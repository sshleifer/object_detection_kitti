from __future__ import print_function, division
from object_detection.create_pascal_tf_record import dict_to_tf_example
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import hashlib
import io
import logging
import numpy as np
import os
import glob

from lxml import etree
from  PIL import Image
import tensorflow as tf
import os.path as p
import shutil

writer = tf.python_io.TFRecordWriter('data/train.tfrecord')
label_map_dict = label_map_util.get_label_map_dict('data/kitti_map.pbtxt')
annotations_dir = '/Users/shleifer/voc_kitti/VOC2012/Annotations/'
examples_path = '/Users/shleifer/voc_kitti/VOC2012/ImageSets/Main/trainval.txt'
dataset_directory = 'data/'


IMAGES_URL = 'http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip'
LABELS_URL = 'http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip'
NUM_TRAIN = 100
# DET_LINK = 'http://kitti.is.tue.mpg.de/kitti/data_object_det_2.zip'


def strip_leading_zeroes(path):
    'training/image_2/00074.jpg -> training/image_2/74.jpg'
    end = path[-4:]
    new_basename = '{}{}'.format(int(p.basename(path)[:-4]), end)
    new_path = p.join(p.dirname(path), new_basename)
    shutil.move(path, new_path)
    return new_path



def fetch_required_kitti_data():

    raise NotImplementedError


def convert_to_pascal_voc():
    raise NotImplementedError


def convert_to_jpg_and_save(png_path):
    im = Image.open(png_path)
    rgb_im = im.convert('RGB')
    new_path = '{}.jpg'.format(png_path[:-4])
    rgb_im.save(new_path)
    return new_path


def strip_zeroes_and_convert_to_jpg(data_dir='~/kitti_data/training'):
    '''convert images to jpg, strip leading zeroes and write train.txt file'''
    data_dir = os.path.expanduser(data_dir)
    image_paths = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
    label_paths = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
    new_img_paths = []
    for path in image_paths:
        stripped_path = strip_leading_zeroes(path)
        jpg_path = convert_to_jpg_and_save(stripped_path)
        new_img_paths.append(jpg_path)
    for path in label_paths:
        strip_leading_zeroes(path)
    file_contents = ','.join([os.path.basename(x)[:-4]
                              for x in np.random.choice(new_img_paths, NUM_TRAIN)])

    with open('~/kitti_data/train.txt', 'w') as f:
        f.write(file_contents)

def xml_to_dict(path):
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    return dataset_util.recursive_parse_xml_to_dict(xml)['annotation']


def create_records(examples_path):
    labels = {}
    examples_list = dataset_util.read_examples_list(examples_path)
    for i, example in enumerate(examples_list[1:]):
        # TODO(SS): why is first example screwed up?
        path = os.path.join(annotations_dir, example + '.xml')
        data = xml_to_dict(path)
        
        labels[i] = [k['name'] for k in data['object']]
        tf_example = dict_to_tf_example(data, 
                                        dataset_directory,
                                        label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    return labels  # to inspect a bit

def do_kitti_ingest():
    kitti_path = fetch_required_kitti_data()
    convert_to_jpg(kitti_path)
    pascal_path = convert_to_pascal_voc(kitti_path)
    print ('writing to data/train.tfrecord')
    create_records(pascal_path)



if __name__ == '__main__':
    do_kitti_ingest()