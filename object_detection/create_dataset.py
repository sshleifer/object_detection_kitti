# /usr/bin/python
from __future__ import print_function, division

import click
import glob
from lxml import etree
import numpy as np
import os
import pandas as pd
from PIL import Image
import shutil
import subprocess
import tensorflow as tf
from tqdm import tqdm

from object_detection.kitti_to_voc import kitti_to_voc
from object_detection.create_pascal_tf_record import dict_to_tf_example
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


label_map_dict = label_map_util.get_label_map_dict('data/kitti_map.pbtxt')
DEFAULT_DATA_DIR = 'kitti_data'
VOC_TRAIN_DIR = 'voc_kitti'
VOC_VALID_DIR = 'voc_kitti_valid'
TRAIN_RECORD_PATH = 'data/train.tfrecord'
VALID_RECORD_PATH = 'data/valid.tfrecord'

def get_fun_paths(base_voc_dir):
    annotations_dir = '{}/VOC2012/Annotations/'.format(base_voc_dir)
    examples_path = '{}/VOC2012/ImageSets/Main/trainval.txt'.format(base_voc_dir)
    return annotations_dir, examples_path


def strip_leading_zeroes(path):
    '''training/image_2/00074.jpg -> training/image_2/74.jpg'''
    end = path[-4:]
    new_basename = '{}{}'.format(int(os.path.basename(path)[:-4]), end)
    new_path = os.path.join(os.path.dirname(path), new_basename)
    if not os.path.exists(new_path):
        shutil.move(path, new_path)
    return new_path


def convert_to_jpg_and_save(png_path):
    # TODO(SS): faster version?
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

def get_labels_path(id, data_dir=DEFAULT_DATA_DIR):
    return os.path.join(data_dir, 'training', 'label_2', '{}.txt'.format(id))


def get_image_path(id, data_dir):
    return os.path.join(data_dir, 'training', 'image_2', '{}.jpg'.format(id))


def split_validation_images(data_dir, pct_train=0.9, num_consider='all'):
    '''make valid.txt and train.txt and create valid subtree'''
    label_paths = glob.glob(os.path.join(data_dir, 'training', 'label_2', '*.txt'))
    if isinstance(num_consider, int):
        label_paths = label_paths[:num_consider]
    else:
        num_consider = len(label_paths)
    assert len(label_paths) > 0
    num_train = int(np.floor(num_consider * pct_train))

    valid_label_dir = os.path.join(data_dir, 'valid', 'label_2')
    valid_image_dir = os.path.join(data_dir, 'valid', 'image_2')
    make_directory_if_not_there(valid_image_dir)
    make_directory_if_not_there(valid_label_dir)

    train_paths = np.random.choice(label_paths, num_train, replace=False)
    train_ids = []
    valid_ids = []
    for label_path in label_paths:
        id = get_id(label_path)
        image_path = get_image_path(id, data_dir)
        if not os.path.exists(image_path):
            print('no path {}'.format(image_path))
            continue

        if label_path in train_paths:
            train_ids.append(id)
        else:
            valid_ids.append(id)
            shutil.move(label_path, valid_label_dir)
            shutil.move(image_path, valid_image_dir)

    assert len(valid_ids) > 0
    make_directory_if_not_there(os.path.join(data_dir, 'valid', 'label_2'))
    train_file_contents = ','.join(train_ids)
    valid_file_contents = ','.join(valid_ids)
    with open('kitti_data/train.txt', 'w+') as f:
        f.write(train_file_contents)
    with open('kitti_data/valid.txt', 'w+') as f:
        f.write(valid_file_contents)


def strip_zeroes_and_convert_to_jpg(data_dir):
    '''convert images to jpg, strip leading zeroes and write train.txt file'''
    data_dir = os.path.expanduser(data_dir)
    image_paths = glob.glob(os.path.join(data_dir, 'training', 'image_2', '*.png'))
    label_paths = glob.glob(os.path.join(data_dir, 'training', 'label_2', '*.txt'))
    for path in tqdm(image_paths):
        print(path)
        stripped_path = strip_leading_zeroes(path)
        convert_to_jpg_and_save(stripped_path)
    for path in label_paths:
        strip_leading_zeroes(path)


def xml_to_dict(path):
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    return dataset_util.recursive_parse_xml_to_dict(xml)['annotation']


def create_records(data_dir, to_path='data/train.tfrecord'):
    annotations_dir, examples_path = get_fun_paths(data_dir)
    writer = tf.python_io.TFRecordWriter(to_path)
    labels = {}
    examples_list = dataset_util.read_examples_list(examples_path)
    assert len(examples_list) > 0, examples_path
    for i, example in enumerate(examples_list):
        path = os.path.join(annotations_dir, example + '.xml')
        data = xml_to_dict(path)
        assert 'object' in data, data['filename']
        labels[i] = [k['name'] for k in data['object']]
        try:
            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
        except Exception as e: #TODO(SS): remove me
            print(e)
            import pdb; pdb.set_trace()
        writer.write(tf_example.SerializeToString())
    writer.close()
    return labels  # to inspect a bit

def glob_base(pat):
    return list(map(os.path.basename, glob.glob(pat)))


def assert_non_overlap_and_len():
    valid_ids = glob_base(VOC_VALID_DIR + '/VOC2012/JPEGImages/*.jpg')
    train_ids = glob_base(VOC_TRAIN_DIR+ '/VOC2012/JPEGImages/*.jpg')
    assert len(pd.Index(valid_ids).intersection(train_ids)) == 0


@click.command()
@click.option('--to-path', default=TRAIN_RECORD_PATH)
@click.option('--data-dir', default=DEFAULT_DATA_DIR)
def do_kitti_ingest(to_path, data_dir):
    # strip_zeroes_and_convert_to_jpg(data_dir)
    assert os.path.exists('vod_converter'), 'Must git clone vod-converter'
    # split_validation_images(data_dir)
    assert not os.path.exists(VOC_TRAIN_DIR)
    assert not os.path.exists(VOC_VALID_DIR)
    kitti_to_voc(os.path.join(data_dir, 'training'),
                 VOC_TRAIN_DIR, os.path.join(data_dir, 'train.txt'))
    kitti_to_voc(os.path.join(data_dir, 'valid'),
                 VOC_VALID_DIR, os.path.join(data_dir, 'valid.txt'))
    create_records(VOC_TRAIN_DIR, to_path=to_path)
    create_records(VOC_VALID_DIR, to_path=VALID_RECORD_PATH)
    assert_non_overlap_and_len()
    print('succesfully wrote {} and {}'.format(to_path, VALID_RECORD_PATH))

if __name__ == '__main__':
    do_kitti_ingest()