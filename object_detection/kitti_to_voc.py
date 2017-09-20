from object_detection.vod_converter.kitti import KITTIIngestor
from object_detection.vod_converter.voc import VOCEgestor
from object_detection.vod_converter.tf_record import TensorflowEgestor
from object_detection.vod_converter.converter import convert

def kitti_to_voc(from_path, to_path, train_id_path, select_only_known_labels=False,
                 filter_images_without_labels=False):
    success, msg = convert(
        from_path=from_path,
        ingestor=KITTIIngestor(),
        to_path=to_path,
        egestor=VOCEgestor(),
        select_only_known_labels=select_only_known_labels,
        filter_images_without_labels=filter_images_without_labels,
        train_ids=train_id_path)
    print(msg)