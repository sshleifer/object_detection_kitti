from object_detection.utils import label_map_util
import os


NUM_CLASSES = 9



PATH_TO_LABELS = os.path.join('data', 'kitti_map.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
name_to_id = {x['name']: x['id'] for x in categories}