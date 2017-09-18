import os
import numpy as np
from PIL import Image as Image

from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from object_detection.create_dataset import xml_to_dict
from object_detection.kitti_constants import *

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def box_to_matrix(data):
    max_height, max_width = float(data['size']['height']), float(data['size']['width'])
    boxes = [x['bndbox'] for x in data['object']]
    return np.array([[
        float(box['ymin']) / max_height,
        float(box['xmin']) / max_width,
        float(box['ymax']) / max_height,
        float(box['xmax']) / max_width
    ] for box in boxes])


def show_groundtruth(image_path):
    data = get_annotations(image_path)
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    classes = np.array([name_to_id[x['name'].lower()]
                        for x in data['object']])
    scores = np.ones(len(data['object']))
    box_mat = box_to_matrix(data)
    print(box_mat.shape)
    # [ 0.01188183  0.          0.98302406  0.94844353]
    visualize_boxes_and_labels_on_image_array(
        image_np,
        box_mat,
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def get_annotations(image_path):
    img_id = os.path.basename(image_path)[:-4]
    annotation_path = os.path.join(
    os.path.split(os.path.dirname(image_path))[0], 'Annotations',
    '{}.xml'.format(img_id)
    )
    return xml_to_dict(annotation_path)