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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchinfo import summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - - %(levelname)s - %(message)s',
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
            f"Expected 1 or 3 as image channels, but {x.shape[-1]} is got."
        )
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
            data = yaml.safe_load(f)
            self.backbone_config = BackboneConfig()
            self.backbone_config.__dict__.update(data['backbone_config'])

            for name, value in data.items():
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

        self.backbone = Backbone(self.config.backbone_config)

        zch = self.config.backbone_config.zch
        img_size = self.config.backbone_config.img_size
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

    # def load_state_dict(self, *args, **kwargs):
    #     """
    #     Redefinition of function to load state dict
    #     """
    #     ret = super().load_state_dict(*args, **kwargs)
    #     for param in self.backbone.parameters():
    #         param.requires_grad = False
    #     return ret

    @classmethod
    def load(cls, file_path):
        """
        Method to load the model pretrained state dict

        :param file_path: The path to file that contents the pretrained
          model weights
        :returns: An instance of this model with state dict loaded

        :type file_path: `str`
        :rtype: VAEncoderFT
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

    def forward(self, x):
        """
        Forward pass
        ------------
        """
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        out = self.bottleneck(features)
        return out

    def save(self, file_path):
        """
        Function to save encoder model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.state_dict()
        torch.save(model_weights, model_file)
        self.config.save(param_file)


def build_fine_tuning_model(backbone_file, num_classes, bottleneck_arch=None):
    """
    Function to build model for fine-tuning using a pretrained backbone
    and a pretrained or new bottleneck

    :param backbone_file: The path to the backbone file
    :param num_classes: The number of classes of the fully connected
    :param bottleneck_arch: The name of the bottleneck architecture
      that will be used to build the bottleneck model. By default,
      if this information is not provided, we will use `fc1layer`
    :returns: An new model instance that build using backbone and bottleneck.

    :type backbone_file: `str`
    :type num_classes: `int`
    :type bottleneck_arch: `str`
    :rtype: VAEncoderFT
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

    instance = VAEncoderFT(model_config)

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
