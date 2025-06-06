#!/bin/env python3
# -*- encoding: utf-8 -*-

"""
===============================================================================
|                        YOLO Training IMPLEMENTATION                         |
===============================================================================


MIT License

Copyright (c) 2025 Dr Mokira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '0.1.0'
__author__ = 'Dr Mokira'

import os
import sys
import logging
import traceback
import argparse
import random

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils import data
from torch.utils.data import Dataset as BaseDataset

import torchvision.transforms.functional as TF

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - -\033[96m %(levelname)s \033[0m - %(message)s',
    handlers=[
        logging.FileHandler("vae_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# MODEL IMPLEMENTATION
###############################################################################


def logits_2_x_min_y_min_x_max_y_max(normalized_boxes, img_size):
    """
    Function to convert [x, y, w, h] to [x_min, y_min, x_max, y_max]

    :param normalized_boxes: The bounding box formatted as x, y, w, h
    :param img_size: The image size
    :returns: The same bounding box formatted as x_min, y_min,
      x_max, y_max.

    :type normalized_boxes: list
    :type img_size: tuple
    :rtype: torch.Tensor
    """
    img_h, img_w = img_size
    center_x = normalized_boxes[0] * img_w
    center_y = normalized_boxes[1] * img_h
    w = normalized_boxes[2] * img_w
    h = normalized_boxes[3] * img_h
    x = center_x - w / 2
    y = center_y - h / 2

    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    xy_min_max = torch.as_tensor([x_min, y_min, x_max, y_max])
    return xy_min_max


def get_bounding_box(logit_values, image_shape):
    """
    :type logit_values: `tuple` of  `float`
    :type image_shape: `tuple` of `int`
    :rtype: `tuple` of `float`
    """
    img_h, img_w = image_shape[:2]
    center_x = logit_values[0] * img_w
    center_y = logit_values[1] * img_h
    w = logit_values[2] * img_w
    h = logit_values[3] * img_h
    x = center_x - w / 2
    y = center_y - h / 2
    return x, y, w, h


###############################################################################
# Training config
###############################################################################

class Config:
    width = 256
    height = 192
    cell_x = 16
    cell_y = 12
    r_x = int(width / cell_x)
    r_y = int(height / cell_y)
    max_object = 60

    anchors = np.array([[3.0, 1.5], [2.0, 2.0], [1.5, 3.0]])
    nbr_boxes = len(anchors)

    batch_size = 16

    lambda_coord = 5
    lambda_noobj = 0.5

    threshold_iou_loss = 0.6


###############################################################################
# METRICS COMPUTATION
###############################################################################

def intersection_over_union(box1, box2):
    """
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: The score of intersection over union.

    :rtype: float
    """
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)


###############################################################################
# DATASET
###############################################################################


def noise(image):
    h, w, c = image.shape
    n = np.random.randn(h, w, c) * random.randint(5, 30)
    return np.clip(image + n, 0, 255).astype(np.uint8)


def gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


class Dataset(BaseDataset):

    def __init__(
        self, dataset_dir, images, class_ids, bboxes, class_names, anchors,
        img_size=(224, 224), coeff=None, cell_x=16, cell_y=12, max_objects=60
    ):
        self.dataset_dir = dataset_dir
        self.images = images
        self.class_ids = class_ids
        self.bboxes = bboxes
        self.class_names = class_names
        self.anchors = anchors
        self.img_size = img_size
        self.coeff = coeff
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.max_objects = max_objects

        self.r_x = int(self.img_size[0] / self.cell_x)
        self.r_y = int(self.img_size[1] / self.cell_y)
        self.num_boxes = len(self.anchors)

        if self.coeff is None:
            self.coeff = random.uniform(1.1, 2.5)

    def is_xmin_xmax(self, boxes):
        """
        Verify if bounding box values are x_min, x_max
        """
        return all(x >= 1 for x in boxes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]  # image file: /path/to/image.png
        class_ids = self.class_ids[item]  # [2, 3, 1, 3, 2 ... ]
        bboxes = self.bboxes[item]  # [[0.3, 0.345, 0.209, 0.503], ...]

        # Open image file
        image = Image.open(image_file).convert('RGB')

        image = np.asarray(image)
        image = torch.as_tensor(image)

        # Perform zooming
        zoom_x = int(self.coeff * self.img_size[0])
        zoom_y = int(self.coeff * self.img_size[1])
        image_r = image.resize(zoom_x, zoom_y)

        image = torch.as_tensor(np.asarray(image))
        image_r = torch.as_tensor(np.asarray(image_r))

        # Add noise into image
        image_r = noise(image_r)

        if self.coeff == 1:
            shift_x = 0
            shift_y = 0
        else:
            shift_x = np.random.randint(image_r.shape[0] - self.img_size[0])
            shift_y = np.random.randint(image_r.shape[1] - self.img_size[1])

        # S: cell_x
        # S: cell_y
        # B: num_boxes == num of anchors
        # C: num_classes
        label_shape = (self.cell_y,
                       self.cell_x,
                       self.num_boxes,
                       5 + len(self.class_names))
        label = torch.zeros(label_shape, dtype=np.float32)
        label2 = torch.zeros((self.max_objects, 7), dtype=np.float32)

        ratio_x = self.coeff * self.img_size[0] / image.shape[0]
        ratio_y = self.coeff * self.img_size[1] / image.shape[1]

        # Class ids
        class_ids = torch.as_tensor(class_ids)  # [p,]

        # Bounding boxes
        obj_id = 0
        for cls_id, bbox in zip(class_ids, bboxes):
            if not self.is_xmin_xmax(bbox):
                bbox = logits_2_x_min_y_min_x_max_y_max(bbox, image.shape[:2])

            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * ratio_x)
            y_min = int(y_min * ratio_y)
            x_max = int(x_max * ratio_x)
            y_max = int(y_max * ratio_y)

            if x_min < shift_x \
                or y_min < shift_y \
                or x_max > (shift_x + self.img_size[0]) \
                or y_max > (shift_y + self.img_size[1]):
                continue

            x_min = (x_min - shift_x) / self.r_x
            y_min = (y_min - shift_y) / self.r_y
            x_max = (x_max - shift_x) / self.r_x
            y_max = (y_max - shift_y) / self.r_y

            area = (x_max - x_min) * (y_max - y_min)
            label2[obj_id] = torch.tensor(
                [x_min, y_min, x_max, y_max, area, 1, cls_id])

            x_centre = int(x_min + (x_max - x_min) / 2)
            y_centre = int(y_min + (y_max - y_min) / 2)
            x_cell = int(x_centre)
            y_cell = int(y_centre)

            a_x_min = x_centre - self.anchors[:, 0] / 2
            a_y_min = y_centre - self.anchors[:, 1] / 2
            a_x_max = x_centre + self.anchors[:, 0] / 2
            a_y_max = y_centre + self.anchors[:, 1] / 2

            id_a = 0
            best_iou = 0
            for i in range(len(self.anchors)):
                iou = intersection_over_union([x_min, y_min, x_max, y_max],
                                              [a_x_min[i], a_y_min[i],
                                               a_x_max[i], a_y_max[i]])
                if iou > best_iou:
                    best_iou = iou
                    id_a = i

            label[y_cell, x_cell, id_a, 0] = (x_max + x_min) / 2
            label[y_cell, x_cell, id_a, 1] = (y_max + y_min) / 2
            label[y_cell, x_cell, id_a, 2] = x_max - x_min
            label[y_cell, x_cell, id_a, 3] = y_max - y_min
            label[y_cell, x_cell, id_a, 4] = 1.
            label[y_cell, x_cell, id_a, 5 + cls_id] = 1.

            obj_id = obj_id + 1
            if obj_id == self.max_objects:
                logger.info("Maximum number of objects reached !!!!!")
                break

        image_r = image_r[shift_y:(shift_y + self.img_size[1]),
                          shift_x:(shift_x + self.img_size[0])]
        return image_r, label, label2


def get_fn_without_ext(s):
    """
    Function to extract name without extension

    :param s: The file name with its extension
    :returns: The name without its extension.

    :type s: `str`
    :rtype: `str`
    """
    if not s:
        return ''
    fp_split = s.split('.')
    fn_split = [x for x in fp_split if x]
    file_name = '.'.join(fn_split[:-1] if len(fn_split) > 1 else fn_split)
    return file_name


class CocoDataCollector:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.config_file = os.path.join(dataset_dir, "data.yaml")
        self.train_images_dir = None
        self.train_labels_dir = None
        self.val_images_dir = None
        self.val_labels_dir = None
        self.test_images_dir = None
        self.test_labels_dir = None

        self.class_names = None
        self.train_image_files = []
        self.train_class_ids = []
        self.train_boxes = []

        self.val_image_files = []
        self.val_class_ids = []
        self.val_boxes = []

        self.test_image_files = []
        self.test_class_ids = []
        self.test_boxes = []

    def load_config_file(self):
        """
        Function to load content of the config file
        """
        with open(self.config_file, mode='r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        return content

    @staticmethod
    def load_label(file_path):
        """
        Function allows to load class IDs and bounding boxes contained
        in the text file located at file_path

        :type file_path: str
        :rtype: typing.Tuple[typing.List[int], numpy.ndarray]
        """
        with open(file_path, mode='r') as f:
            content_file = f.read()
            lines = content_file.split('\n')
            lines = [line for line in lines if line.strip()]

            bboxes = []
            cl_ids = []

            for line in lines:
                try:
                    line_split = line.split(' ')
                    class_id = int(line_split[0])
                    if len(line_split) != 5:
                        continue
                    x = float(line_split[1])
                    y = float(line_split[2])
                    w = float(line_split[3])
                    h = float(line_split[4])
                    if x > 1 or y > 1 or w > 1 or h > 1:
                        continue
                    cl_ids.append(class_id)
                    bboxes.append([x, y, w, h])
                except ValueError as e:
                    logger.warning(f"{e}")
                    continue

            bboxes = np.asarray(bboxes, dtype=np.float32)
            return cl_ids, bboxes

    def _load_data(self, image_files, labels_dir):
        """
        Private method that allows to load labels from text files
        and associate each of them with their image

        :type image_files: `list` of `str`
        :type labels_dir: `str`
        :rtype: typing.Tuple[typing.List[str], typing.List[int], numpy.ndarray]
        """
        if not image_files:
            # The list of image file paths is empty, so we return None
            return
        images = []
        classes = []
        bboxes = []
        for image_file in image_files:
            file_path_without_ext = get_fn_without_ext(image_file)
            label_file = os.path.join(
                labels_dir, f"{file_path_without_ext}.txt"
            )
            if not os.path.isfile(label_file):
                logger.warning(f"No such label file at: {label_file}")
                continue
            class_ids, boxes = self.load_label(label_file)
            images.append(image_file)
            classes.append(class_ids)
            bboxes.append(boxes)

        bboxes = np.asarray(bboxes, dtype=np.float32)
        return images, classes, bboxes

    def collect(self):
        config = self.load_config_file()
        if 'train' not in config:
            raise ValueError(
                "The train directory is not defined in the config file"
                f" located at {self.config_file}."
            )

        if 'val' not in config:
            raise ValueError(
                "The `val` directory representing validation data,"
                " is not defined in the config file"
                f" located at {self.config_file}."
            )

        if 'names' not in config:
            raise ValueError("The list of the class names is not defined.")

        self.train_images_dir = str(
            os.path.join(self.dataset_dir, config['train'])
        )
        self.val_images_dir = str(
            os.path.join(self.dataset_dir, config['val'])
        )
        if 'test' in config:
            self.test_images_dir = str(
                os.path.join(self.dataset_dir, config['test'])
            )
            logger.info(
                "The images of test directory is located at:"
                f" {self.test_images_dir}"
            )

        logger.info(
            "The images of training directory is located at:"
            f" {self.train_images_dir}"
        )
        logger.info(
            "The images of validation directory is located at:"
            f" {self.val_images_dir}"
        )

        self.class_names = config['names']
        logger.info(f"The class names found: {','.join(self.class_names)}")

        train_label_dir = config['train'].replace('images', 'labels')
        val_label_dir = config['val'].replace('images', 'labels')
        self.train_labels_dir = str(
            os.path.join(self.dataset_dir, train_label_dir)
        )
        self.val_labels_dir = str(
            os.path.join(self.dataset_dir, val_label_dir)
        )

        if 'test' in config:
            test_label_dir = config['test'].replace('images', 'labels')
            self.test_labels_dir = str(
                os.path.join(self.dataset_dir, test_label_dir)
            )
            logger.info(
                "The labels of test directory is located at:"
                f" {self.test_labels_dir}"
            )

        logger.info(
            "The labels of training directory is located at:"
            f" {self.train_labels_dir}"
        )
        logger.info(
            "The labels of validation directory is located at:"
            f" {self.val_labels_dir}"
        )

        # Load all file names of images contained on image directory
        # of each sub dataset (train, val, and test set)
        train_image_files = os.listdir(self.train_images_dir)
        val_image_files = os.listdir(self.val_images_dir)
        test_image_files = []
        if self.test_labels_dir:
            test_image_files.extend(os.listdir(self.test_images_dir))

        # Build full file path to each image file listed from sub-datasets
        train_image_files = [
            str(os.path.join(self.train_images_dir, s))
            for s in train_image_files
        ]
        val_image_files = [
            str(os.path.join(self.val_images_dir, s))
            for s in val_image_files
        ]
        test_image_files = [
            str(os.path.join(self.test_images_dir, s))
            for s in test_image_files
        ]

        # load sub-datasets
        returned1 = self._load_data(train_image_files, self.train_labels_dir)
        returned2 = self._load_data(val_image_files, self.val_labels_dir)
        returned3 = self._load_data(test_image_files, self.val_labels_dir)

        self.train_image_files.extend(returned1[0])
        self.train_class_ids.extend(returned1[1])
        self.train_boxes.extend(returned1[2])

        self.val_image_files.extend(returned2[0])
        self.val_class_ids.extend(returned2[1])
        self.val_boxes.extend(returned2[2])

        if returned3 is not None:
            self.test_image_files.extend(returned3[0])
            self.test_class_ids.extend(returned3[1])
            self.test_boxes.extend(returned3[2])


###############################################################################
# TRAINING LOOP
###############################################################################


def get_argument():
    """
    Function to return command line argument parsed

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--data-train',
                        type=str, help="Data train path", required=True)
    parser.add_argument('-dv', '--data-val',
                        type=str, help="Data validation path", required=True)
    args = parser.parse_args()

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    return args


def main():
    """
    Main function to run training process
    """
    args = get_argument()


if __name__ == '__main__':

    def print_err():
        """
        Function of Error tracked printing
        """
        # get traceback error
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tbobj = traceback.extract_tb(exc_traceback)

        # msg += ''.join([' ' for _ in ERRO]) +
        #     "\t%16s %8s %64s\n" % ("FILE NAME", "LINENO", "MODULE/FUNCTION",);
        for tb in tbobj:
            logger.error(
                "\t%16s %8d %64s\n" % (tb.name, tb.lineno, tb.filename,))

    try:
        main()
        exit(0)
    except KeyboardInterrupt as e:
        print("\033[91mCanceled by user!")
        exit(125)
    except FileNotFoundError:
        print_err()
        exit(2)
    except Exception:  # noqa
        print_err()
        exit(1)
