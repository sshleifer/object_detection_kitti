import unittest

from object_detection.utils.kitti import show_groundtruth, get_annotations
from object_detection.kitti_to_voc import kitti_to_voc

IMAGE_PATH = 'kitti_data/training/image_2/2456.jpg'
IMAGE_VOC_PATH= 'voc_kitti/VOC2012/JPEGImages/2456.jpg'
class TestKitti(unittest.TestCase):


    def test_label_retriever(self):
        labels = get_annotations(IMAGE_VOC_PATH)
        print(labels)
        labeled_image = show_groundtruth(IMAGE_VOC_PATH)

    def test_end_to_end(self):
        pass

    def test_tf_record_is_usable(self):
        pass

    def test_kitti_to_voc(self):
        kitti_to_voc('test_env', 'test_voc', 'test_env/train.txt')