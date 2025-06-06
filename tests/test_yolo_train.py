import os
import logging


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


def test_coco_data_collector():
    ...
