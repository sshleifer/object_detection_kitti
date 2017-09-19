# /usr/bin/python
from __future__ import print_function, division

import glob
import shutil
import subprocess

import numpy as np
import os
import tensorflow as tf
from  PIL import Image
from lxml import etree
from tqdm import tqdm

from object_detection.create_pascal_tf_record import dict_to_tf_example
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


label_map_dict = label_map_util.get_label_map_dict('data/kitti_map.pbtxt')
annotations_dir = 'voc_kitti/VOC2012/Annotations/'
DEFAULT_EXAMPLES_PATH = 'voc_kitti/VOC2012/ImageSets/Main/trainval.txt'
dataset_directory = 'data/'


IMAGES_URL = 'http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip'
LABELS_URL = 'http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip'
NUM_TRAIN = 5
# DET_LINK = 'http://kitti.is.tue.mpg.de/kitti/data_object_det_2.zip'


def strip_leading_zeroes(path):
    'training/image_2/00074.jpg -> training/image_2/74.jpg'
    end = path[-4:]
    new_basename = '{}{}'.format(int(os.path.basename(path)[:-4]), end)
    new_path = os.path.join(os.path.dirname(path), new_basename)
    shutil.move(path, new_path)
    return new_path


def convert_to_jpg_and_save(png_path):
    im = Image.open(png_path)
    rgb_im = im.convert('RGB')
    new_path = '{}.jpg'.format(png_path[:-4])
    rgb_im.save(new_path)
    os.remove(png_path)
    return new_path

def get_id(path):
    return os.path.basename(path)[:-4]



def make_directory_if_not_there(path):
    '''makes a directory if not there'''
    if not os.path.exists(path):
        os.makedirs(path)

def get_labels_path(id, data_dir='kitti_data'):
    return os.path.join(data_dir, 'training', 'label_2', '{}.txt'.format(id))


def split_validation_images(data_dir, jpg_paths):
    # TODO: make this work with pascal_converter
    image_paths = glob.glob(os.path.join(data_dir, '*', 'image_2', '*.png'))
    valid_label_dir = os.path.join(data_dir, 'valid', 'label_2')
    valid_image_dir = os.path.join(data_dir, 'valid', 'image_2')
    make_directory_if_not_there(valid_image_dir)
    make_directory_if_not_there(valid_label_dir)
    train_paths = np.random.choice(jpg_paths, NUM_TRAIN)
    train_ids = []; valid_ids = []
    for path in image_paths:
        id = get_id(path)
        if path in train_paths:
            train_ids.append(id)
        else:
            valid_ids.append(id)
            shutil.move(path, valid_image_dir)
            shutil.move(get_labels_path(id), valid_label_dir)

    train_file_contents = ','.join([train_ids])
    valid_file_contents = ','.join([valid_ids])
    make_directory_if_not_there(os.path.join(data_dir, 'valid', 'label_2'))


    with open('kitti_data/train.txt', 'w+') as f:
        f.write(train_file_contents)
    with open('kitti_data/valid.txt', 'w+') as f:
        f.write(valid_file_contents)


def strip_zeroes_and_convert_to_jpg(data_dir='kitti_data'):
    '''convert images to jpg, strip leading zeroes and write train.txt file'''
    # TODO(SS): Split off valid and what about kitti_data/training
    data_dir = os.path.expanduser(data_dir)
    image_paths = glob.glob(os.path.join(data_dir, '*', 'image_2', '*.png'))
    label_paths = glob.glob(os.path.join(data_dir, '*', 'label_2', '*.txt'))
    # TODO(SS): there are no testing labels
    new_img_paths = []
    for path in tqdm(image_paths):
        stripped_path = strip_leading_zeroes(path)
        jpg_path = convert_to_jpg_and_save(stripped_path)
        new_img_paths.append(jpg_path)
    for path in label_paths:
        strip_leading_zeroes(path)


def xml_to_dict(path):
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    return dataset_util.recursive_parse_xml_to_dict(xml)['annotation']


def create_records(examples_path=DEFAULT_EXAMPLES_PATH, to_path='data/train.tfrecord'):
    writer = tf.python_io.TFRecordWriter(to_path)
    labels = {}
    examples_list = dataset_util.read_examples_list(examples_path)
    assert len(examples_list) > 0, examples_path
    for i, example in enumerate(examples_list):
        # TODO(SS): why is first example screwed up?
        path = os.path.join(annotations_dir, example + '.xml')
        data = xml_to_dict(path)
        
        labels[i] = [k['name'] for k in data['object']]
        tf_example = dict_to_tf_example(data, 
                                        'voc_kitti',
                                        label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    return labels  # to inspect a bit


import click
@click.command()
@click.option('--to-path', default='data/train.tfrecord')
def do_kitti_ingest(to_path):
    strip_zeroes_and_convert_to_jpg()
    assert os.path.exists('vod-converter'), 'Must git clone vod-converter'
    subprocess.call("./vod_convert.sh", shell=True)
    create_records(to_path=to_path)


if __name__ == '__main__':
    do_kitti_ingest()