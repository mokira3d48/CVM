#!/bin/env python3
# -*- encoding: utf-8 -*-

__version__ = '0.1.0'
__author__ = 'Dr Mokira'


import os
import math
import logging
from dataclasses import dataclass

import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary

from torch.utils import data
from torch.utils.data import Dataset as BaseDataset
# from torchvision import transforms

import torchvision.transforms.functional as TF

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vae_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """
    Setting the seed for all the random generator
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# MODEL IMPLEMENTATION
###############################################################################

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
        return out


def test_encoder():
    encoder = Encoder(img_channels=3)
    encoder = encoder.to(device)

    inputs = torch.randn((1, 3, 224, 224))
    inputs = inputs.to(device)
    summary(encoder, input_data=inputs)

    inputs = torch.randn((4, 3, 224, 224))
    outputs = encoder(inputs)
    assert outputs.shape == (4, 4, 28, 28)
    logger.info(str(outputs.shape))


class Decoder(nn.Sequential):
    def __init__(
        self, img_channels=3, zch=8, num_groups=32, n_heads=1,
        in_proj_bias=True, out_proj_bias=True, mult_factor=0.18215
    ):
        attention_block = AttentionBlock(
            512, num_groups, n_heads, in_proj_bias, out_proj_bias
        )
        super().__init__(
            nn.Conv2d(zch // 2, 512, 3, 1, 1),  # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512),            # [n, 512, h / 8, w / 8]

            attention_block,          # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512),  # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512),  # [n, 512, h / 8, w / 8]
            ResidualBlock(512, 512),  # [n, 512, h / 8, w / 8]

            nn.Upsample(scale_factor=2),   # [n, 512, h / 4, w / 4]
            nn.Conv2d(512, 512, 3, 1, 1),  # [n, 512, h / 4, w / 4]
            ResidualBlock(512, 512),       # [n, 512, h / 4, w / 4]
            ResidualBlock(512, 512),       # [n, 512, h / 4, w / 4]
            ResidualBlock(512, 512),       # [n, 512, h / 4, w / 4]

            nn.Upsample(scale_factor=2),  # [n, 512, h / 2, w / 2]
            nn.Conv2d(512, 512, 3, 1, 1), # [n, 512, h / 2, w / 2]
            ResidualBlock(512, 256),      # [n, 256, h / 2, w / 2]
            ResidualBlock(256, 256),      # [n, 256, h / 2, w / 2]
            ResidualBlock(256, 256),      # [n, 256, h / 2, w / 2]

            nn.Upsample(scale_factor=2),  # [n, 256, h, w]
            nn.Conv2d(256, 256, 3, 1, 1), # [n, 256, h, w]
            ResidualBlock(256, 128),      # [n, 128, h, w]
            ResidualBlock(128, 128),      # [n, 128, h, w]
            ResidualBlock(128, 128),      # [n, 128, h, w]

            nn.GroupNorm(num_groups, 128),  # [n, 128, h, w]
            nn.SiLU(),                      # [n, 128, h, w]

            nn.Conv2d(128, img_channels, 3, 1, 1)  # [n, img_channels, h, w]
        )

        self.mult_factor = mult_factor

    def forward(self, x):
        """
        Forward pass
        ------------

        :param x: [batch_size, zch, h, w], zch representing the latent
          representation channels, its value is chosen according zch of encoder
          latent representation;
        :returns: tensor with [batch_size, img_channels, h, w]
          representing the reconstructed image.

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        x = x / self.mult_factor  # remove the scaling adding by the encoder;

        for module in self:
            x = module(x)
        return x


def test_decoder():
    decoder = Decoder(img_channels=3)
    decoder = decoder.to(device)

    inputs = torch.randn((1, 4, 28, 28))
    inputs = inputs.to(device)
    summary(decoder, input_data=inputs)

    inputs = torch.randn((4, 4, 28, 28))
    outputs = decoder(inputs)
    assert outputs.shape == (4, 3, 224, 224)
    logger.info(str(outputs.shape))


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
        elif x.shape[1] == 3 and self.img_channels == 1:
            x = TF.rgb_to_grayscale(x, num_output_channels=1)

        # Normalization
        x = x / 255.0
        return x


@dataclass
class ModelConfig:
    img_channels = 3
    img_size = [224, 224]
    num_groups = 32
    zch = 8
    n_heads = 1
    in_proj_bias = True
    out_proj_bias = True
    mult_factor = 0.18215

    def save(self, file_path):
        data = self.__dict__
        with open(file_path, mode='w', encoding='utf-8') as f:
            yaml.dump(data, f)

    def load(self, file_path):
        with open(file_path, mode='r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.__dict__.update(data)


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config if config else ModelConfig()
        self.input_function = Input(
            img_channels=self.config.img_channels,
            img_size=self.config.img_size,
        )
        self.encoder = None
        self.decoder = None

        self.init_encoder()
        self.init_decoder()

    def init_encoder(self):
        self.encoder = Encoder(
            img_channels=self.config.img_channels,
            num_groups=self.config.num_groups,
            zch=self.config.zch,
            n_heads=self.config.n_heads,
            in_proj_bias=self.config.in_proj_bias,
            out_proj_bias=self.config.out_proj_bias,
            mult_factor=self.config.mult_factor,
        )

    def init_decoder(self):
        self.decoder = Decoder(
            img_channels=self.config.img_channels,
            num_groups=self.config.num_groups,
            zch=self.config.zch,
            n_heads=self.config.n_heads,
            in_proj_bias=self.config.in_proj_bias,
            out_proj_bias=self.config.out_proj_bias,
            mult_factor=self.config.mult_factor,
        )


def load_module(file_path, module, map_location='cpu'):
    """
    Function to load state dict from file

    :param file_path: The file path where the module state dict is stored;
    :param module: The module whose state dict we want to load;
    :param map_location: The device name where we want to load state dict;
    :returns: The instance of module with state dict loaded.

    :type file_path: `str`
    :type module: torch.nn.Module
    :type map_location: `str`
    :rtype: torch.nn.Module
    """
    weights = torch.load(
        file_path, weights_only=True, map_location=map_location
    )
    module.load_state_dict(weights)
    logger.info(f"Model weights of {module.__name__} loaded successfully!")
    return module


class Model(VAE):
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
        img_channels = self.config.img_channels
        img_size = self.config.img_size

        input_encoder = torch.randn((batch_size, img_channels, *img_size))
        input_decoder = torch.randn(
            (batch_size, img_channels, img_size[0] // 8, img_size[1] // 8)
        )
        input_encoder = input_encoder.to(model_device)
        input_decoder = input_decoder.to(model_device)
        encoder_model_stats = summary(self.encoder, input_data=input_encoder)
        decoder_model_stats = summary(self.decoder, input_data=input_decoder)
        return encoder_model_stats, decoder_model_stats

    def save_encoder(self, file_path):
        """
        Function to save encoder model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.encoder.state_dict()
        torch.save(model_weights, model_file)
        self.config.save(param_file)

    def save_decoder(self, file_path):
        """
        Function to save decoder model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.decoder.state_dict()
        torch.save(model_weights, model_file)
        self.config.save(param_file)

    @classmethod
    def load(cls, encoder_fp=None, decoder_fp=None):
        """
        Class method to load encoder and decoder model weights from files

        :param encoder_fp: The file path where the encoder model is saved;
        :param decoder_fp: The file path where the decoder model is saved;
        :returns: The instance of VAE model.

        :type encoder_fp: `str`
        :type decoder_fp: `str`
        :rtype: Model
        """
        instance = None
        model_config = None

        if encoder_fp:
            config_file = os.path.join(encoder_fp, 'config.yaml')
            model_config = ModelConfig()
            model_config.load(config_file)

        if decoder_fp:
            config_file = os.path.join(decoder_fp, 'config.yaml')
            model_config = ModelConfig()
            model_config.load(config_file)

        if model_config:
            instance = cls(model_config)
            if encoder_fp:
                model_file = os.path.join(encoder_fp, 'weights.pth')
                instance.encoder = load_module(model_file, instance.encoder)
            if decoder_fp:
                model_file = os.path.join(decoder_fp, 'weights.pth')
                instance.decoder = load_module(model_file, instance.decoder)
        return instance


def test_model_save_load():
    instance = Model()
    instance.save_encoder("encoder_file")
    instance.save_decoder("decoder_file")

    assert os.path.isdir("encoder_file") == True
    assert os.path.isfile("encoder_file/config.yaml")
    assert os.path.isfile("encoder_file/weights.pth")
    assert os.path.isdir("decoder_file") == True
    assert os.path.isfile("decoder_file/config.yaml")
    assert os.path.isfile("decoder_file/weights.pth")

    # Load encoder and decoder from file


###############################################################################
# DATASET IMPLEMENTATION
###############################################################################

class Dataset(BaseDataset):
    def __init__(self, dataset_dir, augment=False):
        self.dataset_dir = dataset_dir
        self.augment = augment

        self.image_files = []
        self.load_image_files()

    def load_image_files(self):
        """
        Method to load image files from dataset directory provided
        """
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                file = os.path.join(root, file)
                self.image_files.append(file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]

        image = Image.open(image_file)
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.uint8)
        image = torch.tensor(image)
        return image


def main():
    """
    Main function to run train process
    """
    ...


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        exit(125)
