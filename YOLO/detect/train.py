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
from PIL import Image

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
        self, dataset_dir, images, class_ids, bboxes, class_names,
        img_size=(224, 224), coeff=None, num_boxes=2, cell_x=16, cell_y=12,
        max_objects=60
    ):
        self.dataset_dir = dataset_dir
        self.images = images
        self.class_ids = class_ids
        self.bboxes = bboxes
        self.class_names = class_names
        self.img_size = img_size
        self.coeff = coeff
        self.num_boxes = num_boxes
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.max_objects = max_objects

        if self.coeff is None:
            self.coeff = random.uniform(1.1, 2.5)

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
        image_r = TF.resize(image, [zoom_x, zoom_y])

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
        # B: num_boxes
        # C: num_classes
        label_shape = (self.cell_y,
                       self.cell_x,
                       self.num_boxes,
                       5 + len(self.class_names))
        label = np.zeros(label_shape, dtype=np.float32)
        label2 = np.zeros((self.max_objects, 7), dtype=np.float32)

        ratio_x = self.coeff * self.img_size[0] / image.shape[0]
        ratio_y = self.coeff * self.img_size[1] / image.shape[1]

        # Class ids
        class_ids = torch.as_tensor(class_ids)  # [p,]

        # Bounding boxes
        bboxes = torch.as_tensor(bboxes)  # [p, 4]



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
