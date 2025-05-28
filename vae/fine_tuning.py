#!/bin/env python3
# -*- encoding: utf-8 -*-

"""
+=============================================================================+
|              SUPERVISED FINE-TUNING TRAINING IMPLEMENTATION                 |
+=============================================================================+

Here is an implementation of training model using supervised fine-tuning
with VAE encoder as backbone.


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
import json
import math
import argparse
import logging
import time
from dataclasses import dataclass
from typing import Mapping, Any
from shutil import copy

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from torch.utils import data
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms.functional as TF
from torchinfo import summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - - \033[95m%(levelname)s\033[0m - %(message)s',
    handlers=[
        logging.FileHandler("vae_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set seeds for reproducibility

    :param seed: An integer value to define the seed for random generator.
    :type seed: `int`
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# MODEL IMPLEMENTATION
###############################################################################


@dataclass
class BaseConfig:

    def data(self):
        raise NotImplementedError

    def save(self, file_path):
        data = self.data()
        with open(file_path, mode='w', encoding='utf-8') as f:
            yaml.dump(data, f)

    def load(self, file_path):
        with open(file_path, mode='r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.__dict__.update(data)


class SelfAttention(nn.Module):
    def __init__(
        self, d_model, n_heads, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            "d_model value is not compatible with n_heads"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)

        self.d_heads = d_model // n_heads

    def forward(self, x, causal_mask=False):
        """
        Forward pass
        ------------
        :param x: [batch_size, seq+len, dim];
        :param causal_mask: Causal mask;

        :type x: torch.Tensor
        :type causal_mask: bool
        :rtype: torch.Tensor
        """
        batch_size, seq_len, d_model = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # change the shape of q, k, and v to match the interim shape
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)

        # Swap the elements within matrix using transpose
        # Take n_heads before seq_len, like that:
        #   [batch_size, n_heads, seq_len, d_heads]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Calculate attention
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper traingle (above the principal diagonal) is 1
            # Fill the upper traingle with -inf
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)

        # [batch_size, n_heads, seq_len, d_model / n_heads]
        output = weight @ v

        # [batch_size, n_heads, seq_len, dd_model / n_heads]
        #   -> [batch_size, seq_len, n_heads, dd_model / n_heads]
        # Change the shape to the shape of out proj
        output = output.transpose(1, 2).contiguous()
        output = output.reshape((batch_size, seq_len, d_model))

        output = self.out_proj(output)
        return output


def test_self_attention():
    self_attn = SelfAttention(d_model=128, n_heads=8)
    inputs = torch.randn((16, 1000, 128))
    outputs = self_attn(inputs)

    assert outputs.shape == (16, 1000, 128)
    logger.info(str(outputs.shape))


class AttentionBlock(nn.Module):
    def __init__(
        self, num_channels, num_groups=32, n_heads=1, in_proj_bias=True,
        out_proj_bias=True
    ):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.attention = SelfAttention(num_channels,
                                       n_heads, in_proj_bias, out_proj_bias)

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, num_channels, h, w];

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        residual = x.clone()

        # [batch_size, num_channels, h, w] -> [batch_size, num_channels, h, w]
        x = self.group_norm(x)

        # Reshape and transpose in [batch_size, h * w, num_channels]
        batch_size, c, h, w = x.shape
        x = x.view((batch_size, c, h * w))
        x = x.transpose(-1, -2).contiguous()

        # Perform self attention without causal mask
        # After this operation, we get: [batch_size, h * w, num_channels]
        x = self.attention(x)

        # Transpose and reshape in [batch_size, num_channels, h, w]
        x = x.transpose(-1, -2).contiguous()
        x = x.view((batch_size, c, h, w))

        out = residual + x
        return out


def test_attention_block():
    attn_block = AttentionBlock(num_channels=256)
    inputs = torch.randn((16, 256, 7, 7))
    outputs = attn_block(inputs)

    assert outputs.shape == (16, 256, 7, 7)
    logger.info(str(outputs.shape))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.group_norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, in_channels, h, w];
        :returns: [batch_size, in_channels, h, w];

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        residue = x.clone()

        x = self.group_norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.group_norm2(x)
        x = self.conv2(x)

        out = x + self.residual_layer(residue)
        return out


def test_residual_block():
    res_block = ResidualBlock(in_channels=128, out_channels=256)
    inputs = torch.randn((16, 128, 14, 14))
    outputs = res_block(inputs)

    assert outputs.shape == (16, 256, 14, 14)
    logger.info(str(outputs.shape))


class Encoder(nn.Sequential):
    def __init__(
        self, img_channels=3, zch=8, num_groups=32, n_heads=1,
        in_proj_bias=True, out_proj_bias=True, mult_factor=0.18215
    ):
        # Conv2D(in_ch, out_ch, ks, s, p)
        attention_block = AttentionBlock(
            512, num_groups, n_heads, in_proj_bias, out_proj_bias
        )
        super().__init__(
            nn.Conv2d(img_channels, 128, 3, 1, 1), # [n, 128, h, w]
            ResidualBlock(128, 128, num_groups),   # [n, 128, h, w]
            nn.Conv2d(128, 128, 3, 2, 0),          # [n, 128, h / 2, w / 2]

            ResidualBlock(128, 256, num_groups), # [n, 256, h / 2, w / 2]
            ResidualBlock(256, 256, num_groups), # [n, 256, h / 2, w / 2]
            nn.Conv2d(256, 256, 3, 2, 0),        # [n, 256, h / 4, w / 4]

            ResidualBlock(256, 512, num_groups), # [n, 512, h / 4, w / 4]
            ResidualBlock(512, 512, num_groups), # [n, 512, h / 4, w / 4]
            nn.Conv2d(512, 512, 3, 2, 0),        # [n, 256, h / 8, w / 8]

            ResidualBlock(512, 512, num_groups),  # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512, num_groups),  # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512, num_groups),  # [n, 512, h / 8, w / 8]

            attention_block,                      # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512, num_groups),  # [n, 512, h / 8, w / 8]
            nn.GroupNorm(num_groups, 512),        # [n, 512, h / 8, w / 8]
            nn.SiLU(),                            # [n, 512, h / 8, w / 8]

            nn.Conv2d(512, zch, 3, 1, 1),  # [n, zch, h / 8, w / 8]
            nn.Conv2d(zch, zch, 1, 1, 0),  # [n, zch, h / 8, w / 8]

        )

        self.mult_factor = mult_factor

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, num_channels, h, w],
          num_channels can be equal to 3, if the images is in RGB,
          or it can be equal to 1, if the images is in Gray scale.
        :returns: tensor with [batch_size, z_channels / 2, h / 8, w / 8]
          representing the latent representation encoded.

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # (left, right, top, bottom)
            x = module(x)

        # We split the tensor x with dim [n, z_channels, h / 8, w / 8]
        #   into two tensors of equal dimensions:
        #   [n, z_channels / 2, h / 8, w / 8]
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp log variance between -30 and 20
        log_variance = torch.clamp(log_variance, -30, 20)

        # Re-parameterization trick
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std

        out = x * self.mult_factor
        return out, (mean, log_variance)


def test_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(img_channels=3)
    encoder = encoder.to(device)

    inputs = torch.randn((1, 3, 224, 224))
    inputs = inputs.to(device)
    summary(encoder, input_data=inputs)

    inputs = torch.randn((4, 3, 224, 224))
    outputs, _ = encoder(inputs)
    assert outputs.shape == (4, 4, 28, 28)
    logger.info(str(outputs.shape))


class BackboneConfig(BaseConfig):
    img_channels = 3
    img_size = [224, 224]
    num_groups = 32
    zch = 8
    n_heads = 1
    in_proj_bias = True
    out_proj_bias = True
    mult_factor = 0.18215

    def data(self):
        return {"img_channels": self.img_channels,
                "img_size": self.img_size,
                "num_groups": self.num_groups,
                "zch": self.zch,
                "n_heads": self.n_heads,
                "in_proj_bias": self.in_proj_bias,
                "out_proj_bias": self.out_proj_bias,
                "mult_factor": self.mult_factor}


class Backbone(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        if not self.config:
            self.config = BackboneConfig()

        self.encoder = Encoder(
            img_channels=self.config.img_channels,
            num_groups=self.config.num_groups,
            zch=self.config.zch,
            n_heads=self.config.n_heads,
            in_proj_bias=self.config.in_proj_bias,
            out_proj_bias=self.config.out_proj_bias,
            mult_factor=self.config.mult_factor)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        out, _ = self.encoder(x)
        return out
        

class FC1Layer(nn.Module):
    """
    Fully connected 1 layer
    -----------------------

    :param in_features: The input features dimension
    :param num_classes: The number of classes

    :type in_features: `int`
    :type num_classes: `int`
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.output = nn.Linear(self.in_features, self.num_classes)

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, in_features]
        :returns: [batch_size, num_classes]

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        out = self.output(x)
        return out


class FC3Layer(nn.Module):
    """
    Fully connected 3 layers
    ------------------------

    :param in_features: The input features dimension
    :param num_classes: The number of classes
    :param hidden_dims: The number of neurons of hidden layer, by default
      this parameter is set to 1024
    :param dropout_prob: The probability of dropout, its default value
      is set to 0.2

    :type in_features: `int`
    :type num_classes: `int`
    :type hidden_dims: `int`
    :type dropout_prob: `float`
    """
    def __init__(
        self, in_features, num_classes, hidden_dims=1024, dropout_prob=0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob

        self.input = nn.Linear(self.in_features, self.hidden_dims)
        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.hidden = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.dropout2 = nn.Dropout(p=self.dropout_prob)
        self.output = nn.Linear(self.hidden_dims, self.num_classes)

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, in_features]
        :returns: [batch_size, num_classes]

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        x = self.input(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.output(x)
        return out


class Input(nn.Module):
    def __init__(self, img_channels=3, img_size=(224, 224)):
        super().__init__()
        assert img_channels in (1, 3), (
            "Either image channels is equal to 3 or equal to 1"
            f" Never equal to {img_channels}"
        )
        self.img_channels = img_channels
        self.img_size = img_size
        # self.gray_transform = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=1),
        # ])

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
            f"Expected 1 or 3 as image channels, but {x.shape[-1]} is got.")
        # [batch_size, img_channels, h, w]
        x = x.permute((0, 3, 2, 1))
        x = x.contiguous()

        # Resize
        x = TF.resize(x, self.img_size)

        # RGB, Gray scale conversion
        if x.shape[1] == 1 and self.img_channels == 3:
            x = torch.cat([x, x, x], dim=1)
            x = TF.normalize(
                x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif x.shape[1] == 3 and self.img_channels == 1:
            x = TF.rgb_to_grayscale(x, num_output_channels=1)
            # x = TF.normalize(x, [0.5], [0.5])

        # Normalization
        # x = x / 255.0
        # TF.normalize()
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
        :rtype: tuple of torch.Tensor
        """
        probs = torch.log_softmax(x, dim=-1)  # [n, num_classes]
        class_ids = torch.argmax(probs, dim=-1)  # [n,]
        confidences = torch.max(probs, dim=-1)  # [n,]
        return class_ids, confidences


class ModelConfig(BaseConfig):
    backbone_config: BackboneConfig = BackboneConfig()
    bottleneck_arch: str = 'fc1layer'
    dropout_prob: float = 0.2
    hidden_dims: int = 1024
    num_classes: int = 10

    def data(self):
        return {
            "backbone_config": self.backbone_config.data(),
            "bottleneck_arch": self.bottleneck_arch,
            "num_classes": self.num_classes}

    def load(self, file_path):
        with open(file_path, mode='r', encoding='utf-8') as f:
            conf_data = yaml.safe_load(f)
            self.load_from_dict(conf_data)

    def load_from_dict(self, conf_data):
        self.backbone_config = BackboneConfig()
        self.backbone_config.__dict__.update(conf_data['backbone_config'])

        for name, value in conf_data.items():
            if name != 'backbone_config':
                self.__dict__[name] = value


class VAEncoderFT(nn.Module):
    """
    Fine-tuning of VAE encoder

    :args config: The model config will be used to build all the architecture
    :type config: ModelConfig
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        if not self.config:
            self.config = ModelConfig()

        img_channels = self.config.backbone_config.img_channels
        img_size = self.config.backbone_config.img_size
        self.input = Input(img_channels=img_channels, img_size=img_size)

        self.backbone = Backbone(self.config.backbone_config)

        zch = self.config.backbone_config.zch
        in_features = (zch // 2) * (img_size[0] // 8) * (img_size[1] // 8)

        bottleneck = self.config.bottleneck_arch
        if bottleneck == 'fc1layer':
            self.bottleneck = FC1Layer(in_features, self.config.num_classes)
        elif bottleneck == 'fc3layer':
            self.bottleneck = FC3Layer(
                in_features, self.config.num_classes,
                hidden_dims=self.config.hidden_dims,
                dropout_prob=self.config.dropout_prob)
        else:
            raise ValueError(
                f"The selected bottleneck named '{bottleneck}'"
                " is not supported.")

    def forward(self, x):
        """
        Forward pass
        ------------
        """
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        out = self.bottleneck(features)
        return out


class Model(VAEncoderFT):

    @classmethod
    def load(cls, file_path):
        """
        Method to load the model pretrained state dict

        :param file_path: The path to file that contents the pretrained
          model weights
        :returns: An instance of this model with state dict loaded

        :type file_path: `str`
        :rtype: Model
        """
        config_file = os.path.join(file_path, 'config.yaml')
        weights_file = os.path.join(file_path, 'weights.pth')
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"No such model config file at {config_file}")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(
                f"No such model weights file at {weights_file}")
        config = ModelConfig()
        config.load(config_file)

        weights = torch.load(
            weights_file, weights_only=True, map_location='cpu')
        instance = cls(config)
        instance.load_state_dict(weights)
        return instance

    def __init__(self, config=None):
        super().__init__(config)

    def device(self):
        return next(self.parameters()).device

    def summary(self, batch_size=1):
        """
        Function to summary

        :param batch_size: The batch size that is used to print
          model architecture
        :type batch_size: int
        """
        model_device = self.device()
        img_channels = self.config.backbone_config.img_channels
        img_size = self.config.backbone_config.img_size

        input_encoder = torch.randn((batch_size, img_channels, *img_size))
        input_encoder = input_encoder.to(model_device)

        model_stats = summary(self, input_data=input_encoder)
        return model_stats

    def save(self, file_path):
        """
        Function to save encoder model weights into file

        :params file_path: The model file path where we want to save
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.state_dict()
        torch.save(model_weights, model_file)
        self.config.save(param_file)


def build_fine_tuning_model(
    backbone_file, num_classes, bottleneck_arch=None, model_class=Model
):
    """
    Function to build model for fine-tuning using a pretrained backbone
    and a pretrained or new bottleneck

    :param backbone_file: The path to the backbone file
    :param num_classes: The number of classes of the fully connected
    :param bottleneck_arch: The name of the bottleneck architecture
      that will be used to build the bottleneck model. By default,
      if this information is not provided, we will use `fc1layer`
    :param model_class: The class of the model will be instanced
    :returns: An new model instance that build using backbone and bottleneck.

    :type backbone_file: `str`
    :type num_classes: `int`
    :type bottleneck_arch: `str`
    :type model_class: type
    :rtype: Model
    """
    if not os.path.isdir(backbone_file):
        raise FileNotFoundError(
            "The file of the pretrained weights of the backbone is not found.")
    model_config = ModelConfig()
    model_config.bottleneck_arch = bottleneck_arch
    if not model_config.bottleneck_arch:
        model_config.bottleneck_arch = 'fc1layer'
    model_config.num_classes = num_classes

    backbone_config_file = os.path.join(backbone_file, 'config.yaml')
    backbone_config = BackboneConfig()
    backbone_config.load(backbone_config_file)
    model_config.backbone_config = backbone_config

    instance = model_class(model_config)

    backbone_weights_file = os.path.join(backbone_file, "weights.pth")
    backbone_weights = torch.load(backbone_weights_file)
    instance.backbone.encoder.load_state_dict(backbone_weights)

    return instance


def test_fine_tuning_model():
    model = build_fine_tuning_model(
        backbone_file="../outputs/plate_digit_encoder",
        num_classes=36)
    model.summary()
    model.save("../outputs/fine_tuning_model")

    new_model = VAEncoderFT.load("../outputs/fine_tuning_model")
    new_model.summary(4)


###############################################################################
# DATASET IMPLEMENTATION
###############################################################################

class Dataset(BaseDataset):
    def __init__(self, dataset_dir, img_size=(224, 224), augment=False):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.img_size = img_size

        self.images = []
        self.labels = []
        self.class_names = []
        self.load_samples()

    def load_samples(self):
        """
        Method to load image files from dataset directory provided
        """
        self.class_names = os.listdir(self.dataset_dir)
        for class_name in self.class_names:
            label_file_path = os.path.join(self.dataset_dir, class_name)
            image_files = os.listdir(label_file_path)
            for image_file in image_files:
                image_file_path = os.path.join(label_file_path, image_file)
                self.images.append(image_file_path)
                self.labels.append(class_name)

    def __len__(self):
        # return 72  # For an example
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        class_name = self.labels[item]

        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize(self.img_size)
        class_id = self.class_names.index(class_name)

        image = np.asarray(image, dtype=np.uint8)
        image = torch.tensor(image)
        label = torch.tensor(class_id, dtype=torch.int64)
        return image, label


###############################################################################
# TRAINING PROCESS
###############################################################################

class AvgMeter:

    def __init__(self, val=0):
        self.total = val
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0

    def __add__(self, other):
        """
        Add function

        :type other: `float`|`int`
        """
        self.total += other
        self.count += 1
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return 0.0

    def __str__(self):
        return str(self.total)


class Trainer(Model):
    """
    Training model
    ==============
    """

    @classmethod
    def load(cls, file_path=None, checkpoint_file=None):
        """
        Static method to load model state dict from files or checkpoints

        :param file_path: The model file contained the weights and config;
        :param checkpoint_file: The file path to the training checkpoint;
        :returns: The instance of the model.

        :type file_path: `str`
        :type checkpoint_file: `str`
        :rtype: Trainer
        """
        instance = None
        if file_path:
            instance = super().load(file_path)

        if not instance and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file,  weights_only=False, map_location='cpu')

            config = ModelConfig()
            if 'model_config' in checkpoint:
                config.load_from_dict(checkpoint['model_config'])
                logger.info("Model config is loaded from checkpoint!")

            instance = cls(config)
            if 'model_state_dict' in checkpoint:
                instance.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state dict is loaded from checkpoint!")

        return instance

    def __init__(self, config=None):
        super().__init__(config)

        self.train_loader = None
        self.val_loader = None

        self.cross_entropy = None
        self.optimizer = None
        self.lr_scheduler = None

        self.train_losses = {}
        self.val_losses = {}
        self.cross_entropy_loss = AvgMeter()

        self.gas = 128  # Gradiant Accumulation Steps
        self.seed = 42  # Seed number for random generation
        self.num_epochs = 1
        self.batch_size = 1

        self.gac = 0  # Gradient Accumulation Count

        self.checkpoint_dir = "checkpoints"
        self.resume_ckpt = None
        self.best_model = None
        self.epoch = 0

    def compile(self, args):
        """
        Initialization of training process
        ----------------------------------

        :type args: `argparse.Namespace`
        :rtype: `None`
        """
        # torch.manual_seed(args.seed)
        # np.random.seed(args.seed)
        set_seed(args.seed)

        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.gas = args.gas

        train_ds_dir = args.train_ds_dir
        val_ds_dir = args.val_ds_dir
        if not os.path.isdir(train_ds_dir):
            raise FileNotFoundError(
                f"No such training set directory at {train_ds_dir}")
        if not os.path.isdir(val_ds_dir):
            raise FileNotFoundError(
                f"No such validation set directory at {val_ds_dir}")
        train_dataset = Dataset(train_ds_dir)
        val_dataset = Dataset(val_ds_dir)

        # Create data loaders
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False)

        # Set loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Set up the optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay)

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1)

        self.resume_ckpt = args.resume
        self.checkpoint_dir = args.checkpoint_dir
        self.best_model = args.best_model

    @staticmethod
    def _update_losses(input_losses, output_losses):
        """
        Update losses value
        """
        for key, val in input_losses.items():
            if key in output_losses:
                output_losses[key] += val
            else:
                new_loss = AvgMeter(val)
                output_losses[key] = new_loss

    @staticmethod
    def print_results(res):
        """
        Function of average meter losses
        """
        string = ''
        for key, val in res.items():
            string += f"{key}: {val:.8f} "
        return string

    @staticmethod
    def add_to_epoch_results(epc_res, new_res):
        for key, val in epc_res.items():
            if key in epc_res:
                epc_res[key].append(new_res[key])
            else:
                epc_res[key] = [new_res[key]]

    def checkpoint(self, **kwargs):
        checkpoint = {
            "model_config": self.config.data(),
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            # "best_performance": self.best_performance,
            **kwargs}

        file_path1 = os.path.join(
            self.checkpoint_dir, "checkpoint.pth")
        file_path2 = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self.epoch}.pth")
        torch.save(checkpoint, file_path1)
        copy(file_path1, file_path2)
        logger.info("Checkpoint done successfully")

        if self.epoch >= 2 and (self.epoch % 2) == 0:
            old_checkpoint_file = os.path.join(
                self.checkpoint_dir, f"checkpoint_{self.epoch - 2}.pth")
            if os.path.isfile(old_checkpoint_file):
                os.remove(old_checkpoint_file)
                logger.info(f"{old_checkpoint_file} checkpoint is removed.")

    def load_checkpoint(self):
        if not self.resume_ckpt:
            return
        if not os.path.isfile(self.resume_ckpt):
            logger.info(f"No such checkpoint file at {self.resume_ckpt}")
            return

        ckpt_data = torch.load(
            self.resume_ckpt, weights_only=False, map_location='cpu')
        self.optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt_data['lr_scheduler_state_dict'])
        self.epoch = ckpt_data['epoch'] + 1
        self.train_losses = ckpt_data['train_losses']
        self.val_losses = ckpt_data['val_losses']
        # self.best_performance = ckpt_data['best_performance']
        logger.info(f"Checkpoint loaded successfully from {self.resume_ckpt}!")

    def train_step(self, images, targets, write_fn, optimize=False):
        """
        Training method on one batch
        """
        # Forward pass
        outputs = self.forward(images)

        # Comput losses
        loss = self.cross_entropy(outputs, targets)

        # Backward pass
        loss.backward()

        self.cross_entropy_loss += loss.item()

        self.gac += len(images)
        if self.gac >= self.gas or optimize:
            self.optimizer.step()
            self.optimizer.zero_grad()

            write_fn(
                "\t* Optim step"
                f" - cross_entropy loss: {self.cross_entropy_loss.avg():.8f}")
            self.gac = 0

    def train_one_epoch(self):
        """
        Method of training on one epoch
        """
        model_device = self.device()
        loss_data = {}

        self.cross_entropy_loss.reset()

        length = len(self.train_loader)
        desc = "\033[44m    TRAINING\033[0m"
        iterator = tqdm(self.train_loader, desc=desc)
        write_fn = iterator.write

        self.train()
        self.optimizer.zero_grad()
        for index, (images, targets) in enumerate(iterator):
            images = images.to(model_device)  # noqa
            targets = targets.to(model_device)

            images = self.input(images)

            # At last iteration, the gradient accumulation count can not
            # be equal to gradient accumulation step, so we must perform
            # optimization step when we are at the last iteration (length - 1)
            is_last_index = index >= (length - 1)
            self.train_step(images, targets, write_fn, is_last_index)

            loss_data = {
                "cross_entropy_loss": self.cross_entropy_loss.avg()}
            iterator.set_postfix(loss_data)

        return loss_data

    def validate(self):
        model_device = self.device()
        loss_data = {}
        self.cross_entropy_loss.reset()

        self.eval()
        with torch.no_grad():
            desc = "\033[43m    VALIDATION\033[0m"
            iterator = tqdm(self.val_loader, desc=desc)

            for images, targets in iterator:
                images = images.to(model_device)  # noqa
                targets = targets.to(model_device)

                images = self.input(images)

                # Forward pass
                outputs = self.forward(images)

                # Comput losses
                loss = self.cross_entropy(outputs, targets)

                self.cross_entropy_loss += loss.item()

                loss_data.update({
                    "cross_entropy": self.cross_entropy_loss.avg()})
                iterator.set_postfix(loss_data)

        return loss_data

    def fit(self):
        """
        Training process
        ----------------

        Run the training loop and returns the results
        formatted as dictionary.

        :rtype: `dict`
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_checkpoint()

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            logger.info(f'Epoch: {epoch + 1} / {self.num_epochs}:')

            train_losses = self.train_one_epoch()

            # Update the learning rate
            self.lr_scheduler.step()

            # Add losses to train losses epochs
            # Save checkpoint with the current model state
            self.add_to_epoch_results(self.train_losses, train_losses)

            logger.info(f'{self.print_results(train_losses)}')

            # Make checkpoint after training
            self.checkpoint()

            val_losses = self.validate()

            self.add_to_epoch_results(self.val_losses, val_losses)

            logger.info(f'{self.print_results(val_losses)}')

            # Make checkpoint after validation
            self.checkpoint()

            if epoch != (self.num_epochs - 1):
                # Epochs are remaining
                logger.info(("-" * 80) + "\n")

        self.save("fine_tuning_model")
        logger.info(
            "Fine-tuning model is saved at\033[92m fine_tuning_mode\033[0m.")


def parse_argument():
    """
    Command line argument parsing
    """
    parser = argparse.ArgumentParser(prog="VAE Train")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-dt', '--train-ds-dir', type=str, required=True)
    parser.add_argument('-dv', '--val-ds-dir', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument('-nc', '--num-classes', type=int, default=1)
    parser.add_argument(
        '--bottleneck', type=str, choices=['fc1layer', 'fc3layer'],
        default='fc1layer'
    )

    parser.add_argument('-n', '--epochs', type=int, default=2)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('-gas', type=int, default=128)

    parser.add_argument('-r', "--resume", type=str)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--backbone-model', type=str, help="Backbone model")
    parser.add_argument('--model-file', type=str, help="Fine-tuning model")
    parser.add_argument('--best-model', type=str, default="best")

    args = parser.parse_args()
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    return args


def main():
    """
    Main function to run train process
    """
    args = parse_argument()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    backbone_model_file = args.backbone_model
    model_file = args.model_file
    checkpoint = args.resume
    model = None
    if checkpoint:
        model = Trainer.load(checkpoint_file=checkpoint)
    if not model and model_file:
        model = Trainer.load(model_file)
    if not model and backbone_model_file:
        num_classes = args.num_classes
        bottleneck = args.bottleneck
        model = build_fine_tuning_model(
            backbone_model_file, num_classes, bottleneck, Trainer)

    if not model:
        logger.info(
            "ERR: No file path of backbone model provided. Please,"
            " provide one.")
        exit(1)

    model = model.to(device)

    model.compile(args)
    model.summary(args.batch_size)
    model.fit()


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\033[91mCanceled by user!\033[0m")
        exit(125)
