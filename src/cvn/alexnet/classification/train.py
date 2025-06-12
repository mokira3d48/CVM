#!/usr/bin/env python3
#-*- encoding: utf8 -*-

__version__ = '0.1.0'
__auther__ = 'Doctor Mokira'

import os
import json
import argparse
import logging
from time import time

import yaml
import numpy as np
import matplotlib.pyplot as plt
from jinja2.optimizer import optimize
from tqdm import tqdm
from PIL import Image, ExifTags
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms.functional as TF
# from torchvision import transforms
from torchinfo import summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alex_net_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """
    Set seeds for reproducibility

    :param seed: An integer value to define the seed for random generator
    :type seed: `int`
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
#  ALEX-NET MODEL IMPLEMENTATION
###############################################################################

class ModelConfig:

    def __init__(
        self, img_size=(224, 224), img_channels=3, num_classes=10,
        dropout_prob=0.2,
        mean=[0.485, 0.456, 0.406],  # noqa
        std=[0.229, 0.224, 0.225],  # noqa
    ):
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.mean = mean
        self.std = std

    def state_dict(self):
        """
        Returns the dictionary of variables associated with its values
        :rtype: typing.Dict[str, object]
        """
        return self.__dict__

    def save(self, file_path):
        """
        Save the attribute values into a YAML file

        :param file_path: The path to the YAML file
        :type file_path: str
        """
        config_data = self.state_dict()
        with open(file_path, mode='w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

    def load(self, file_path):
        """
        Load the attribute values from a YAML file

        :param file_path: The YAML file where the model config data is stored
        :type file_path: str
        """
        with open(file_path, mode='r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            self.__dict__.update(config_data)


class AlexNet(nn.Module):
    """
    ALEX-NET implementation

    :type config: ModelConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config if config else ModelConfig()
        in_channels = self.config.img_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        features_map_size = self._evaluate_features_map_shape()
        feature_map_dim = 1
        for dim in features_map_size[1:]:
            feature_map_dim *= dim

        self.post_backbone = nn.Sequential(
            nn.Linear(feature_map_dim, 4096),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.fc = nn.Linear(4096, self.config.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _evaluate_features_map_shape(self):
        x = torch.zeros((1, 3, 96, 128))
        y = self.backbone(x)
        return y.shape

    def _initialize_weights(self):
        """Function of model weights initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_maps = self.backbone(x)
        x = self.post_backbone(feature_maps)
        out = self.fc(x)
        return out


class Input(nn.Module):

    def __init__(
        self, img_channels=3, img_size=(224, 224),
        mean=[0.485, 0.456, 0.406],  # noqa
        std=[0.229, 0.224, 0.225],  # noqa
        device=None
    ):
        super().__init__()
        assert img_channels in (1, 3), (
            "Either image channels is equal to 3 or equal to 1"
            f" Never equal to {img_channels}"
        )
        self.img_channels = img_channels
        self.img_size = list(img_size)
        self.mean = mean
        self.std = std
        self.device = device if device else torch.device('cpu')

    def forward(self, x):
        """
        Preprocessing method
        --------------------

        :param x: [batch_size, w, h, img_channels];
        :returns: [batch_size, img_channels, h, w]

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        assert x.shape[-1] in (1, 3), (
            f"Expected 1 or 3 as image channels, but {x.shape[-1]} is got."
        )

        # [batch_size, img_channels, h, w]
        x = x.permute((0, 3, 2, 1))
        x = x.contiguous()

        if x.shape[-2:] != self.img_size:
            x = TF.resize(x, self.img_size)

        # RGB, Gray scale conversion
        if x.shape[1] == 1 and self.img_channels == 3:
            x = torch.cat([x, x, x], dim=1)
            x = TF.normalize(x, self.mean, self.std)
        elif x.shape[1] == 3 and self.img_channels == 1:
            x = TF.rgb_to_grayscale(x, num_output_channels=1)

        # Normalization
        x = x / 255.0
        x = x.to(torch.float32)
        return x


class Output(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Post-processing method
        ----------------------

        :param x: [batch_size, num_classes];
        :returns: tuple of two tensors of size [batch_size,],
          the first tensor contains the class ids predicted
          and the second tensor contains the softmax confidences.

        :type x: torch.Tensor
        :rtype: typing.Tuple[torch.Tensor, torch.Tensor]
        """
        probs = torch.softmax(x, dim=-1)  # [n, num_classes]
        class_ids = torch.argmax(probs, dim=-1)  # [n,]
        confidences = torch.max(probs, dim=-1).values  # [n,]
        return class_ids, confidences


class Model(AlexNet):
    def __init__(self, config):
        super().__init__(config)
        self.input = Input(
            img_channels=self.config.img_channels,
            img_size=self.config.img_size,
            mean=self.config.mean,
            std=self.config.std,
        )
        self.output = Output()

    def device(self):
        return next(self.parameters()).device

    def summary(self, batch_size=1):
        """
        Method of model summary

        :param batch_size: The batch size

        :type batch_size: int
        """
        model_device = self.device()
        B = batch_size
        img_channels = self.config.img_channels
        img_size = self.config.img_size
        input_image = torch.randn((B, img_channels, img_size))
        input_image = input_image.to(model_device)
        results = summary(self, input_data=input_image)
        return results

    def weights(self):
        """Return the model weights"""
        state_dict = self.state_dict()
        return state_dict

    def save(self, file_path):
        """
        Function to save model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.weights()
        torch.save(model_weights, model_file)
        self.config.save(param_file)

    @classmethod
    def load(cls, file_path):
        """
        Static method to load model weights from file

        :param file_path: The path to folder that contents the model weight
          file and the model config file as YAML format
        :returns: An instance of the model with its weights correctly loaded.

        :type file_path: str
        :rtype: Model
        """
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')
        if not os.path.isfile(param_file):
            raise FileNotFoundError(
                f"No such model config file at {param_file}.")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"No such model file at {model_file}.")

        hparams = ModelConfig()
        hparams.load(param_file)
        instance = cls(hparams)
        weights = torch.load(model_file, weights_only=True, map_location='cpu')
        instance.load_state_dict(weights)
        logger.info("Model weights loaded successfully!")
        return instance


def fine_tune_model(
    pretrained_model_path, num_new_classes, lr, freeze_feature_layers=True,
    weight_decay=0.005
):
    """
    Load a pre-trained model and modify it for fine-tuning
    on a new dataset with different number of classes.

    :param pretrained_model_path: Path to the pre-trained model file path.
    :param num_new_classes: Number of classes in the new dataset.
    :param lr: The base learning rate will be used to config optimizer
        with the deference layers learning rate.
    :param freeze_feature_layers: Whether to freeze the feature
        extraction layers.
    :param weight_decay: The weight decay value.

    :type pretrained_model_path: `str`
    :type num_new_classes: `int`
    :type lr: `float`
    :type freeze_feature_layers: `bool`
    :type weight_decay: float

    :returns:
        model: Modified model ready for fine-tuning
        optimizer: Configured optimizer for fine-tuning
    :rtype: typing.Tuple[Model, torch.optim.Optimizer]
    """
    # Load the checkpoint
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.load(pretrained_model_path)

    model.fc = nn.Linear(4096, num_new_classes)
    model.config.num_classes = num_new_classes

    if freeze_feature_layers:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        backbone_params = list(model.backbone.parameters())
        post_backbone_params = list(model.post_backbone.parameters())
        fc_params = list(model.fc.parameters())
        optimizer = optim.Adam(
            [
                {'params': backbone_params, 'lr': lr * 0.01},
                #: Very low learning rate for frozen feature layers

                {'params': post_backbone_params, 'lr': lr * 0.1},
                #: Medium learning rate for FC1 and FC2

                {'params': fc_params, 'lr': lr}
                #: High learning rate for the new classification layer
            ],
            weight_decay=weight_decay
        )
    return model, optimizer


###############################################################################
# DATASET IMPLEMENTATION
###############################################################################

def read_image(image_path):
    """
    Fix image rotation based on EXIF orientation data

    :param image_path: Path to the input image
    :returns: Corrected image object

    :type image_path: `str`
    :rtype: numpy.ndarray
    """
    try:
        # Open the image
        image = Image.open(image_path)

        # Get EXIF data
        exif = image._getexif()

        if exif is not None:
            # Find orientation tag
            orientation_key = None
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]

                # Apply rotation based on orientation value
                if orientation == 2:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 4:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = image.rotate(90, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 7:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)

        return image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


class Dataset(BaseDataset):
    """
    Dataset implementation

    :arg inputs: The list of features representing the image files
    :arg targets: The list of the targets
    :arg class_names: The list of available class names
    :arg transform: The pipeline of image transformation

    :type inputs: typing.List[str]
    :type targets: typing.List[str]
    :type targets:
    """
    def __init__(
        self, inputs, targets, class_names, img_size=(224, 224), transform=None
    ):
        self.samples = [(x, y) for x, y in zip(inputs, targets)]
        self.class_names = class_names
        self.img_size = img_size
        self.transform = transform

    @classmethod
    def build(cls, data_dir, img_size=(224, 224), transform=None):
        """
        Class method to build a dataset from data samples
        contained on data dir

        :param data_dir: The path to the data directory
        :param img_size: The image size will be used to resize all images
        :param transform: The pipeline of data transformation
        :returns: An instance of the Dataset.

        :type data_dir: `str`
        :type img_size: typing.Tuple[int, int]
        :type transform: torchvision.transforms.Compose
        :rtype: Dataset
        """
        paths = []
        classes = []
        class_names = []
        folder_names = os.listdir(data_dir)
        folder_names = sorted(folder_names)

        loader = tqdm(folder_names, total=len(folder_names))
        for index, folder_name in enumerate(loader):
            class_name = folder_name
            class_idx = index

            class_names.append(class_name)
            folder_path = os.path.join(data_dir, folder_name)
            file_names = os.listdir(folder_path)
            file_count = len(file_names)
            loader.set_description(
                f"Loading of {file_count} from {folder_name}"
            )
            for fid, file_name in enumerate(file_names):
                is_image_file = (
                    file_name.endswith(".png") or file_name.endswith(".jpg")
                )
                if not is_image_file:
                    continue
                file_path = os.path.join(folder_path, file_name)
                paths.append(file_path)
                classes.append(class_idx)
                loader.set_postfix(files=f"{(fid + 1)}/{file_count}")

            loader.write(f"Class {class_name} of {file_count} is processed.")

        loader.set_description("Done")

        instance = cls(
            paths, classes, class_names, img_size=img_size,
            transform=transform
        )
        return instance

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # img = Image.open(f).convert('RGB')

        img = read_image(path)
        img = img.resize(self.img_size)

        if self.transform:
            img = self.transform(img)

        return img, target
