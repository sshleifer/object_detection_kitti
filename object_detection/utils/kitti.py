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

def matrix_to_box(box):
    coords = ['xmin', 'xmax', 'ymin', 'ymax']
    return dict(zip(coords, box))


def get_boxes_scores_classes(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result immage, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return boxes, scores, classes, num_detections

reverse_category = {
    'car': 1,
    'cyclist': 6,
    'dontcare': 9,
    'misc': 8,
    'pedestrian': 4,
    'person_sitting': 5,
    'tram': 7,
    'truck': 3,
    'van': 2
}


def create_results_list(paths, sess, detection_graph):
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    image_id = []
    groundtruth_boxes = []
    groundtruth_classes = []
    for image_path in paths:
        data = get_annotations(image_path)
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        boxes, scores, classes, num_detections = get_boxes_scores_classes(image_np, sess, detection_graph)
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        # append to lists
        detection_boxes.append(boxes)
        detection_scores.append(scores)
        detection_classes.append(classes)
        image_id.append(data['filename'][:-4])
        groundtruth_boxes.append(box_to_matrix(data))
        groundtruth_classes.append(
            np.array([reverse_category[x['name']] for x in data['object']])
        )
    return dict(image_id=image_id,
                detection_boxes=detection_boxes,
                detection_classes=detection_classes,
                groundtruth_boxes=groundtruth_boxes,
                groundtruth_classes=groundtruth_classes,
                detection_scores=detection_scores
                )


def show_groundtruth(image_path):
    '''Draw bboxes from annotations file on an image'''
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)

    data = get_annotations(image_path)
    classes = np.array([name_to_id[x['name'].lower()]
                        for x in data['object']])
    scores = np.ones(len(data['object']))
    if classes.ndim != 1:
        classes = np.squeeze(classes).astype(np.int32)
    if scores.ndim != 1:
        scores = np.squeeze(scores).astype(np.int32)
    box_mat = box_to_matrix(data)
    visualize_boxes_and_labels_on_image_array(
        image_np,
        box_mat,
        classes,
        scores,
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