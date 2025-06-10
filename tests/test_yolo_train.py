import os
import logging

from torch.utils import data
from cvn.YOLO.detect import train

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_yolo_train.log"),
    ]
)
logger = logging.getLogger(__name__)
dataset_dir = "samples/dataset/"


def test_coco_data_collector():
    data_collector = train.CocoDataCollector(dataset_dir)
    data_collector.collect()

    train_images = data_collector.train_image_files
    train_boxes = data_collector.train_boxes
    train_class_ids = data_collector.train_class_ids
    class_names = data_collector.class_names

    train_dataset = train.Dataset(
        dataset_dir, train_images, train_class_ids, train_boxes, class_names
    )
    num_samples = 0
    for i in range(len(train_dataset)):
        image, label, label2 = train_dataset[i]
        logger.info("Image shape: " + str(image.shape))
        logger.info("Labels " + str(label))
        logger.info("Labels2: " + str(label2) + "\n")
        if num_samples >= 5:
            break
        num_samples += 1
