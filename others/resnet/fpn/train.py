#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import time
import math
import logging
import argparse
from datetime import datetime

import yaml
import numpy as np
from PIL import Image
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resnet_fpn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
NUM_CLASSES = 20  # COCO has 80 classes, but we'll use 20 for example
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet18 and ResNet34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50, ResNet101 and ResNet152
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet backbone
    ===============

    for the Feature Pyramid Network
    """

    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        """
        Build residual layers
        """
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)  # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution

        return [c2, c3, c4, c5]


class FPN(nn.Module):
    """
    Feature Pyramid Network
    =======================
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        # Top-down layers
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            c1x1_s1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            c3x3_s1_p1 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(c1x1_s1)
            self.fpn_convs.append(c3x3_s1_p1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample and add
            interpol = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
            laterals[i - 1] = laterals[i - 1] + interpol

        # Apply conv on each feature map
        outs = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)]

        return outs


class ResNetFPN(nn.Module):
    """
    ResNet Feature Pyramid Network
    ==============================
    """

    def __init__(
            self, backbone_name='resnet50', pretrained=True, out_channels=256,
            num_classes=1000
    ):
        super().__init__()

        # Create ResNet backbone
        if backbone_name == 'resnet18':
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
            in_channels_list = [64, 128, 256, 512]
        elif backbone_name == 'resnet34':
            self.backbone = ResNet(BasicBlock, [3, 4, 6, 3])
            in_channels_list = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
            in_channels_list = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            self.backbone = ResNet(Bottleneck, [3, 4, 23, 3])
            in_channels_list = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet152':
            self.backbone = ResNet(Bottleneck, [3, 8, 36, 3])
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Load pretrained weights if specified
        if pretrained:
            pretrained_model = getattr(
                torchvision.models, backbone_name)(pretrained=True)
            self.backbone.load_state_dict(
                pretrained_model.state_dict(), strict=False)
            print(f"Loaded pretrained weights for {backbone_name}")

        # FPN
        self.fpn = FPN(in_channels_list, out_channels)

        # Classification head (just for demonstration)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels * 4, num_classes)
        #: Combine all levels for classification

    def forward(self, x):
        # Get backbone features
        backbone_features = self.backbone(x)

        # Get pyramid features from FPN
        pyramid_features = self.fpn(backbone_features)

        # For classification task (just for demonstration)
        out = []
        for feat in pyramid_features:
            out.append(self.avgpool(feat))

        out = torch.cat([x.flatten(1) for x in out], dim=1)
        out = self.fc(out)

        return out, pyramid_features


class DetectionHead(nn.Module):
    """
    Detection head for object detection using FPN features
    """
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification branch
        self.cls_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_output = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=1)

        # Regression branch (for bounding box)
        self.reg_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1)
        self.reg_output = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1)  # 4 for (x, y, w, h)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Classification branch
        cls_feat = self.cls_conv(x)
        cls_feat = torch.relu(cls_feat)
        cls_output = self.cls_output(cls_feat)

        # Reshape output for classification [N, A*C, H, W] -> [N, H, W, A, C]
        N, _, H, W = cls_output.shape
        cls_output = cls_output.view(
            N, self.num_anchors, self.num_classes, H, W)
        cls_output = cls_output.permute(0, 3, 4, 1, 2)
        cls_output = cls_output.contiguous()

        # Regression branch
        reg_feat = self.reg_conv(x)
        reg_feat = torch.relu(reg_feat)
        reg_output = self.reg_output(reg_feat)

        # Reshape output for regression [N, A*4, H, W] -> [N, H, W, A, 4]
        reg_output = reg_output.view(N, self.num_anchors, 4, H, W)
        reg_output = reg_output.permute(0, 3, 4, 1, 2)
        reg_output = reg_output.contiguous()

        return cls_output, reg_output


class ObjectDetector(nn.Module):
    """
    Object detector using ResNetFPN backbone
    """
    def __init__(
            self, backbone_name='resnet50', pretrained=True, num_classes=20
    ):
        super().__init__()

        # FPN backbone
        self.backbone = ResNetFPN(
            backbone_name=backbone_name, pretrained=pretrained,
            out_channels=256, num_classes=num_classes)

        # Detection heads for each FPN level
        self.detection_heads = nn.ModuleList([
            DetectionHead(in_channels=256, num_classes=num_classes)
            for _ in range(4)])

    def forward(self, x):
        # Get features from backbone
        _, fpn_features = self.backbone(x)

        # Apply detection heads to each level of feature pyramid
        cls_outputs = []
        reg_outputs = []

        for feat, head in zip(fpn_features, self.detection_heads):
            cls_out, reg_out = head(feat)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)

        return cls_outputs, reg_outputs


import torch.optim as optim
import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter


class RandomDataset(data.Dataset):
    """
    Random dataset for object detection
    """
    def __init__(self, images, targets, transform=None):
        self.images = images  # Shape: [N, C, H, W]
        self.targets = targets  # List of dictionaries with 'boxes' and 'labels'
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


class Dataset(data.Dataset):
    """
    Dataset for object detection
    """
    def __init__(self, dataset_dir, cfg_file, img_size, transform=None):
        self.dataset_dir = dataset_dir
        self.cfg_file = cfg_file
        self.img_size = img_size
        self.transform = transform

        self.img_dir = os.path.join(self.dataset_dir, "images")
        self.lab_dir = os.path.join(self.dataset_dir, "labels")
       # self.cfg_file = os.path.join(self.dataset_dir, "data.yaml")

        self.images = os.listdir(self.img_dir)
        self.labels = os.listdir(self.lab_dir)

        self.label_names = self._load_class_names(self.cfg_file)

    def _load_class_names(self, file_path):
        """
        Class name loading
        ------------------

        Returns the class names list contained
        in YAML file located at file_path.

        :param file_path: The file path of the YAML file.
        :type file_path: `str`
        :rtype: `list` of `str`
        """
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            if 'names' not in content:
                raise AttributeError(
                    "No class names list is not defined into YAML"
                    f" file at: {file_path}")
            names = content['names']
            return names
        return content

    def _load_bbox(self, label_filepath):
        """
        Bounding box loading
        --------------------

        :type label_filepath: `str`
        :rtype: `tuple` of `np.ndarray`
        """
        with open(label_filepath, mode='r') as f:
            content_file = f.read()
            lines = content_file.split('\n')
            lines = [line for line in lines if line.strip()]
            img_w = self.img_size[0]
            img_h = self.img_size[1]

            class_ids = []
            bboxes = []
            for line in lines:
                try:
                    line_split = line.split(' ')
                    class_id = int(line_split[0])
                    if len(line_split) != 5:
                        continue
                    nx = float(line_split[1])
                    ny = float(line_split[2])
                    nw = float(line_split[3])
                    nh = float(line_split[4])
                    if nx > 1 or ny > 1 or nw > 1 or nh > 1:
                        continue
                    class_ids.append(class_id)

                    center_x = nx * img_w
                    center_y = ny * img_h
                    w = nw * img_w
                    h = nh * img_h
                    x = center_x - w / 2
                    y = center_y - h / 2

                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    bboxes.append([x1, y1, x2, y2])
                except ValueError as e:
                    print(f"{e}")
                    continue

            class_ids = np.asarray(class_ids, dtype=np.float32)
            bboxes = np.asarray(bboxes, dtype=np.float32)
            return class_ids, bboxes

    def _class_id_verify(self, class_ids):
        """
        Class ID verifying
        ------------------

        :type class_ids: `numpy.ndarray`
        :rtype: `None`
        """
        classes_count = len(self.label_names)
        for class_id in class_ids.tolist():
            if not (0 <= class_id < classes_count):
                raise ValueError(
                    f"The class ID {class_id} is not in available classes.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        lab_path = os.path.join(self.lab_dir, self.labels[idx])

        class_ids, bboxes = self._load_bbox(lab_path)
        self._class_id_verify(class_ids)

        # Load image
        # Convert into RGB
        # Resize image
        # Normalize pixels
        # Transpose to [C H W]
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.asarray(image)
        # image = cv.resize(image, self.img_size)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        # Load annotations
        # Albumentations augmentations
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        image = torch.tensor(image, dtype=torch.float32)
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(class_ids, dtype=torch.int64)
        }
        return image, target

def create_random_dataset(
    num_samples=10, img_size=512, num_classes=20, max_objects=10
):
    """
    Create a random dataset for testing
    """
    images = []
    targets = []

    for _ in range(num_samples):
        # Random image
        image = torch.rand(3, img_size, img_size)

        # Random number of objects (1 to max_objects)
        num_objects = np.random.randint(1, max_objects + 1)

        # Random boxes and labels
        boxes = []
        labels = []

        for _ in range(num_objects):
            # Random box coordinates (x1, y1, x2, y2) format
            x1 = np.random.randint(0, img_size // 2)
            y1 = np.random.randint(0, img_size // 2)
            x2 = np.random.randint(x1 + 50, img_size)
            y2 = np.random.randint(y1 + 50, img_size)

            # Random class label
            label = np.random.randint(0, num_classes)

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        images.append(image)
        targets.append(target)

    return images, targets


def collate_fn(batch):
    """
    Custom collate function for batching detection data
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss for bounding box regression
    """

    def __init__(self, beta=0.11, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        diff = torch.abs(inputs - targets)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta,
                           diff - 0.5 * self.beta)

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_losses(cls_outputs, reg_outputs, targets, img_size=128):
    """
    Compute detection losses
    """
    batch_size = len(targets)
    num_levels = len(cls_outputs)

    cls_loss = 0
    reg_loss = 0

    # Simple implementation for demonstration purposes
    # In a real implementation, you would:
    # 1. Generate anchors for each level
    # 2. Match ground truth boxes to anchors
    # 3. Create target tensors
    # 4. Compute losses

    # For this example, we'll create dummy targets
    for level in range(num_levels):
        # Get output size at this level
        _, _, h, w, _ = cls_outputs[level].shape

        for b in range(batch_size):
            # Get ground truth boxes and labels for this image
            gt_boxes = targets[b]['boxes']  # Shape: [N, 4]
            gt_labels = targets[b]['labels']  # Shape: [N]
            num_boxes = len(gt_boxes)

            if num_boxes == 0:
                continue

            # Scale ground truth boxes to feature map size
            scale_factor = h / img_size
            scaled_boxes = gt_boxes * scale_factor

            # Create dummy targets for classification and regression
            cls_target = torch.zeros_like(cls_outputs[level][b])
            reg_target = torch.zeros_like(reg_outputs[level][b])

            # For each ground truth box, assign targets to the closest grid cell
            for i in range(num_boxes):
                box = scaled_boxes[i]
                label = gt_labels[i]

                # Convert to [x, y, w, h] format for regression
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]

                # Find the grid cell
                grid_x = int(center_x)
                grid_y = int(center_y)

                if 0 <= grid_x < w and 0 <= grid_y < h:
                    # Assign classification target (one-hot encoded)
                    cls_target[grid_y, grid_x, 0, label] = 1.0

                    # Assign regression target
                    reg_target[
                        grid_y, grid_x, 0, 0] = center_x - grid_x  # x offset
                    reg_target[
                        grid_y, grid_x, 0, 1] = center_y - grid_y  # y offset
                    reg_target[grid_y, grid_x, 0, 2] = width  # width
                    reg_target[grid_y, grid_x, 0, 3] = height  # height

            # Classification loss
            cls_level_loss = FocalLoss()(cls_outputs[level][b].view(-1),
                                         cls_target.view(-1))
            cls_loss += cls_level_loss

            # Regression loss
            # Create a mask for positive samples
            pos_mask = cls_target.sum(dim=-1, keepdim=True).expand_as(
                reg_target)
            reg_level_loss = SmoothL1Loss()(reg_outputs[level][b], reg_target,
                                            pos_mask)
            reg_loss += reg_level_loss

    # Normalize losses
    cls_loss = 100_000_000 * cls_loss / (batch_size * num_levels)
    reg_loss = 200_000_000 * reg_loss / (batch_size * num_levels)

    return {
        'cls_loss': cls_loss,
        'reg_loss': reg_loss,
        'total_loss': cls_loss + reg_loss
    }


def train_one_epoch(model, optimizer, data_loader, epoch, device):
    """
    Train for one epoch
    """
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    steps = 0

    for images, targets in data_loader:
        images = images.to(device)

        # Forward pass
        cls_outputs, reg_outputs = model(images)

        # Compute losses
        losses = compute_losses(cls_outputs, reg_outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        total_cls_loss += losses['cls_loss'].item()
        total_reg_loss += losses['reg_loss'].item()
        steps += 1

        # Print progress
        if steps % 10 == 0:
            logger.info(
                f" Epoch: {epoch + 1},"
                f" Step: {steps}/{len(data_loader)},"
                f" Loss: {loss.item():.8f},"
                f" Cls Loss: {losses['cls_loss'].item():.8f},"
                f" Reg Loss: {losses['reg_loss'].item():.8f}")

    # Compute average losses
    avg_loss = total_loss / steps
    avg_cls_loss = total_cls_loss / steps
    avg_reg_loss = total_reg_loss / steps

    return avg_loss, avg_cls_loss, avg_reg_loss


def validate(model, data_loader, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    steps = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)

            # Forward pass
            cls_outputs, reg_outputs = model(images)

            # Compute losses
            losses = compute_losses(cls_outputs, reg_outputs, targets)

            # Update statistics
            total_loss += losses['total_loss'].item()
            total_cls_loss += losses['cls_loss'].item()
            total_reg_loss += losses['reg_loss'].item()
            steps += 1

    # Compute average losses
    avg_loss = total_loss / steps
    avg_cls_loss = total_cls_loss / steps
    avg_reg_loss = total_reg_loss / steps

    return avg_loss, avg_cls_loss, avg_reg_loss


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save a checkpoint of the model"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001,
                weight_decay=0.0001,
                patience=5, checkpoint_dir='checkpoints'):
    """Train the model"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1,
                                                     patience=patience // 2)

    # Create tensorboard writer
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Track best model
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print('-' * 100)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        start_time = time.time()
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(
            model, optimizer, train_loader, epoch, DEVICE
        )
        train_time = time.time() - start_time

        # Validate
        val_loss, val_cls_loss, val_reg_loss = validate(model, val_loader,
                                                        DEVICE)

        # Update learning rate
        scheduler.step(val_loss)

        # Print results
        logger.info(
            f" Training loss: {train_loss:.8f}, CLS: {train_cls_loss:.8f}, REG: {train_reg_loss:.8f}")
        logger.info(
            f" Validation loss: {val_loss:.8f}, CLS: {val_cls_loss:.8f}, REG: {val_reg_loss:.8f}")
        logger.info(
            f" Time: {train_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f'checkpoint.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Check if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            logger.info(f"New best model saved at {best_model_path}")
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= patience:
            logger.info(
                f"Early stopping after {patience} epochs without improvement")
            break

    print("Training completed!")

    return model


def main(args):
    """Main function"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Creating model: {args.backbone}")

    # Create dataset
    # print("Creating random datasets for testing...")
    # train_images, train_targets = create_random_dataset(num_samples=50,
    #                                                     img_size=args.img_size,
    #                                                     num_classes=NUM_CLASSES)
    # val_images, val_targets = create_random_dataset(num_samples=10,
    #                                                 img_size=args.img_size,
    #                                                 num_classes=NUM_CLASSES)
    #
    # train_dataset = RandomDataset(train_images, train_targets)
    # val_dataset = RandomDataset(val_images, val_targets)
    train_dataset = Dataset("dataset/train", "dataset/data.yaml", (128, 128))
    val_dataset = Dataset("dataset/val", "dataset/data.yaml", (128, 128))
    num_classes = len(train_dataset.label_names)

    # Create model
    model = ObjectDetector(backbone_name=args.backbone,
                           pretrained=args.pretrained, num_classes=num_classes)
    model = model.to(DEVICE)

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    # Train model
    logger.info("Starting training...")
    _model = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )

    logger.info("Done!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ResNetFPN for object detection')

    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('-n', '--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def test_training():
    """Test function to validate the training setup"""
    print("Testing the training setup with a small dataset...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a small model
    model = ObjectDetector(backbone_name='resnet18', pretrained=False,
                           num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Create small dataset (10 images)
    print("Creating random dataset...")
    images, targets = create_random_dataset(num_samples=10, img_size=224,
                                            num_classes=NUM_CLASSES)
    dataset = RandomDataset(images, targets)

    # Create data loader
    data_loader = data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    # Test forward pass
    print("Testing forward pass...")
    for images, targets in data_loader:
        images = images.to(DEVICE)
        cls_outputs, reg_outputs = model(images)

        # Check outputs
        print(f"Number of feature levels: {len(cls_outputs)}")
        for i, (cls_out, reg_out) in enumerate(zip(cls_outputs, reg_outputs)):
            print(f"Level {i + 1}:")
            print(f"  Classification output shape: {cls_out.shape}")
            print(f"  Regression output shape: {reg_out.shape}")

        # Check loss computation
        losses = compute_losses(cls_outputs, reg_outputs, targets,
                                img_size=224)
        print(f"Classification loss: {losses['cls_loss'].item():.4f}")
        print(f"Regression loss: {losses['reg_loss'].item():.4f}")
        print(f"Total loss: {losses['total_loss'].item():.4f}")

        break

    # Test training for 2 epochs
    print("\nTesting training for 2 epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        start_time = time.time()

        model.train()
        epoch_loss = 0

        for images, targets in data_loader:
            images = images.to(DEVICE)

            # Forward pass
            cls_outputs, reg_outputs = model(images)

            # Compute losses
            losses = compute_losses(cls_outputs, reg_outputs, targets,
                                    img_size=224)
            loss = losses['total_loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Batch loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        print(f"Time: {time.time() - start_time:.2f}s")

    print("\nTraining test completed successfully!")
    return model


if __name__ == "__main__":
    # Parse arguments and run main
    args = parse_args()
    main(args)
