#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import os
import argparse
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import transforms

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


def generate_anchors(feature_size, scales, aspect_ratios, stride):
    """
    Generate anchors for a feature map of the given size
    """
    anchors = []
    for y in range(feature_size[0]):
        for x in range(feature_size[1]):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for scale in scales:
                for ratio in aspect_ratios:
                    h = scale * ratio ** 0.5
                    w = scale / ratio ** 0.5
                    anchors.append([cx, cy, w, h])
    return torch.tensor(anchors)


def generate_anchor_boxes(feature_sizes, img_size=128):
    """Generate anchors for all feature pyramid levels"""
    anchor_boxes = []

    # Define scales and aspect ratios for anchors
    scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
    aspect_ratios = [0.5, 1.0, 2.0]

    # Calculate strides for each feature level
    strides = [img_size // size[0] for size in feature_sizes]

    for i, size in enumerate(feature_sizes):
        anchors = generate_anchors(size, scales, aspect_ratios, strides[i])
        anchor_boxes.append(anchors)

    return anchor_boxes


def decode_box_predictions(cls_outputs, reg_outputs, anchor_boxes, img_size,
                           score_threshold=0.5):
    """Decode model outputs to bounding boxes"""
    all_boxes = []
    all_scores = []
    all_labels = []

    batch_size = cls_outputs[0].shape[0]
    num_classes = cls_outputs[0].shape[-1]

    # Process each image in the batch
    for b in range(batch_size):
        boxes = []
        scores = []
        labels = []

        # Process each feature pyramid level
        for level, (cls_output, reg_output, level_anchors) in enumerate(
                zip(cls_outputs, reg_outputs, anchor_boxes)):
            # Get classification scores
            cls_preds = cls_output[
                b].detach().sigmoid()  # Apply sigmoid to get probabilities

            # Get regression predictions
            reg_preds = reg_output[b].detach()

            # Get feature map size
            H, W = cls_preds.shape[:2]
            A = cls_preds.shape[2]  # Number of anchors per location

            # Reshape tensors for processing
            cls_preds = cls_preds.reshape(H * W * A, num_classes)
            reg_preds = reg_preds.reshape(H * W * A, 4)
            level_anchors = level_anchors.reshape(-1, 4).to(cls_preds.device)

            # Find detections above threshold
            max_scores, max_labels = torch.max(cls_preds, dim=1)
            keep = max_scores > score_threshold

            if keep.sum() == 0:
                continue

            # Get boxes, scores and labels for detections above threshold
            level_boxes = reg_preds[keep]
            level_scores = max_scores[keep]
            level_labels = max_labels[keep]
            level_anchors = level_anchors[keep]

            # Convert regression outputs to boxes in [x1, y1, x2, y2] format
            # For this example, we assume the model predicts [dx, dy, dw, dh]
            # where (dx, dy) are offsets to the anchor center and (dw, dh)
            # are log-scale factors
            anchor_x = level_anchors[:, 0]
            anchor_y = level_anchors[:, 1]
            anchor_w = level_anchors[:, 2]
            anchor_h = level_anchors[:, 3]

            pred_cx = level_boxes[:, 0] * anchor_w + anchor_x
            pred_cy = level_boxes[:, 1] * anchor_h + anchor_y
            pred_w = torch.exp(level_boxes[:, 2]) * anchor_w
            pred_h = torch.exp(level_boxes[:, 3]) * anchor_h

            pred_x1 = pred_cx - 0.5 * pred_w
            pred_y1 = pred_cy - 0.5 * pred_h
            pred_x2 = pred_cx + 0.5 * pred_w
            pred_y2 = pred_cy + 0.5 * pred_h

            # Clip boxes to image bounds
            pred_x1 = torch.clamp(pred_x1, 0, img_size)
            pred_y1 = torch.clamp(pred_y1, 0, img_size)
            pred_x2 = torch.clamp(pred_x2, 0, img_size)
            pred_y2 = torch.clamp(pred_y2, 0, img_size)

            # Stack boxes
            level_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2],
                                      dim=1)

            boxes.append(level_boxes)
            scores.append(level_scores)
            labels.append(level_labels)

        # Concatenate detections from all levels
        if boxes:
            boxes = torch.cat(boxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            boxes = torch.zeros((0, 4), device=DEVICE)
            scores = torch.zeros(0, device=DEVICE)
            labels = torch.zeros(0, dtype=torch.long, device=DEVICE)

        # Apply NMS
        keep = apply_nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply non-maximum suppression to remove overlapping boxes
    """
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert boxes to (x1, y1, x2, y2) if they're not already
    if boxes.shape[1] == 4:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    else:
        raise ValueError("Boxes should be in [x1, y1, x2, y2] format")

    # Compute areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score
    order = torch.argsort(scores, descending=True)

    keep = []
    while order.numel() > 0:
        # The index of the highest scoring box
        i = order[0].item()
        keep.append(i)

        # If only one box is left, break
        if order.numel() == 1:
            break

        # Compute IoU of the highest scoring box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU < threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def get_coco_class_names():
    """
    Return example class names (simplified COCO classes)
    """
    # return [
    #     'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #     'bus',
    #     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    #     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    #     'cow',
    #     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    #     'handbag'
    # ]
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_color_map(num_classes):
    """
    Generate color map for visualizing different classes
    """
    colors = []
    for i in range(num_classes):
        # Generate a distinct color for each class
        r = int((i * 5 + 50) % 255)
        g = int((i * 3 + 100) % 255)
        b = int((i * 11 + 150) % 255)
        colors.append((r, g, b))
    return colors


def preprocess_image(image_path, img_size=128):
    """Preprocess image for model input"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()

    # Resize
    img = img.resize((img_size, img_size), Image.BILINEAR)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img_tensor, original_img


def predict_and_draw(model, image_path, output_path=None, img_size=128,
                     score_threshold=0.5, device=DEVICE):
    """Predict objects in image and draw bounding boxes"""
    # Load and preprocess image
    img_tensor, original_img = preprocess_image(image_path, img_size)
    img_tensor = img_tensor.to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        cls_outputs, reg_outputs = model(img_tensor)

    # Get feature map sizes
    feature_sizes = [(output.shape[1], output.shape[2]) for output in
                     cls_outputs]

    # Generate anchors
    anchor_boxes = generate_anchor_boxes(feature_sizes, img_size)

    # Decode predictions
    boxes, scores, labels = decode_box_predictions(
        cls_outputs, reg_outputs, anchor_boxes, img_size, score_threshold
    )

    # Get class names and color map
    class_names = get_coco_class_names()
    colors = get_color_map(len(class_names))

    # Convert to numpy for visualization
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    labels = labels[0].cpu().numpy()

    # Scale boxes to original image size
    orig_w, orig_h = original_img.size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    boxes[:, 0] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 2] *= scale_x
    boxes[:, 3] *= scale_y

    # Draw boxes
    draw = ImageDraw.Draw(original_img)

    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[label] if label < len(
            class_names) else f"class_{label}"
        color = colors[label % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        text = f"{class_name}: {score:.2f}"

        # Use the newer method for getting text dimensions
        if hasattr(font, "getbbox"):
            # Newer versions of Pillow
            bbox = font.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(draw, "textbbox"):
            # Alternative method available in some versions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            # Fallback to approximate dimensions
            text_w, text_h = len(text) * 8, 15

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h), text, fill=(255, 255, 255), font=font)

    # Save or show the image
    if output_path:
        original_img.save(output_path)
        print(f"Saved annotated image to {output_path}")

    # Display using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(original_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return boxes, scores, labels, original_img


def plot_detections(image, boxes, scores, labels, class_names=None):
    """Plot detections using matplotlib"""
    if class_names is None:
        class_names = get_coco_class_names()

    colors = get_color_map(len(class_names))

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    ax = plt.gca()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        width = x2 - x1
        height = y2 - y1

        class_name = class_names[label] if label < len(
            class_names) else f"class_{label}"
        color = colors[label % len(colors)]

        # Convert RGB to matplotlib format (0-1)
        color_norm = [c / 255 for c in color]

        # Create rectangle patch
        rect = Rectangle((x1, y1), width, height, linewidth=2,
                         edgecolor=color_norm, facecolor='none')
        ax.add_patch(rect)

        # Add label
        plt.text(
            x1, y1 - 5, f"{class_name}: {score:.2f}",
            color='white', fontsize=10,
            bbox=dict(facecolor=color_norm, alpha=0.8)
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main_prediction(args):
    """
    Main function for prediction
    """
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = ObjectDetector(backbone_name=args.backbone, pretrained=False,
                           num_classes=36)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    print(f"Model loaded. Running inference on {args.image}")

    # Make prediction and visualize
    boxes, scores, labels, annotated_img = predict_and_draw(
        model, args.image, args.output,
        img_size=args.img_size,
        score_threshold=args.threshold,
        device=DEVICE
    )

    # Print detection results
    class_names = get_coco_class_names()
    print("\nDetection Results:")
    print("------------------")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_idx = int(label)
        class_name = class_names[class_idx] if class_idx < len(
            class_names) else f"class_{class_idx}"
        print(f"Detection {i + 1}: {class_name} ({score:.2f}) at {box}")


def test_prediction():
    """
    Test function to validate the prediction setup with random data
    """
    print("Testing the prediction setup with random data...")

    # Create a small model
    model = ObjectDetector(backbone_name='resnet18', pretrained=False,
                           num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Generate a random image
    img_size = 128
    random_img = torch.rand(3, img_size, img_size)

    # Convert tensor to PIL Image for visualization
    img_np = (random_img * 255).byte().permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img_np)

    # Add random boxes to the image
    draw = ImageDraw.Draw(pil_img)
    ground_truth_boxes = []

    for _ in range(5):  # Add 5 random boxes
        x1 = np.random.randint(50, img_size - 100)
        y1 = np.random.randint(50, img_size - 100)
        w = np.random.randint(50, 150)
        h = np.random.randint(50, 150)
        x2 = min(x1 + w, img_size - 10)
        y2 = min(y1 + h, img_size - 10)

        label = np.random.randint(0, NUM_CLASSES)
        color = get_color_map(NUM_CLASSES)[label]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        ground_truth_boxes.append([x1, y1, x2, y2, label])

    # Save the random image
    temp_img_path = "temp_random_image.jpg"
    pil_img.save(temp_img_path)

    # Run model prediction
    img_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        cls_outputs, reg_outputs = model(img_tensor)

    # Get feature map sizes
    feature_sizes = [(output.shape[1], output.shape[2]) for output in
                     cls_outputs]

    # Generate anchors
    anchor_boxes = generate_anchor_boxes(feature_sizes, img_size)

    # Decode predictions
    boxes, scores, labels = decode_box_predictions(
        cls_outputs, reg_outputs, anchor_boxes, img_size, score_threshold=0.3
    )

    print("\nModel prediction successful!")
    print(f"Found {len(boxes[0])} detections")

    # Let's visualize the inference and ground truth
    print("Visualizing the results...")

    # Convert model outputs to numpy for visualization
    pred_boxes = boxes[0].cpu().numpy()
    pred_scores = scores[0].cpu().numpy()
    pred_labels = labels[0].cpu().numpy()

    # Display ground truth
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title("Ground Truth")
    plt.axis('off')

    # Display model predictions
    ax = plt.subplot(1, 2, 2)
    plt.imshow(pil_img)
    plt.title("Model Predictions")

    colors = get_color_map(NUM_CLASSES)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.3:  # Only show high-confidence detections
            x1, y1, x2, y2 = box.astype(int)
            width = x2 - x1
            height = y2 - y1

            color = colors[int(label) % len(colors)]
            color_norm = [c / 255 for c in color]

            rect = Rectangle((x1, y1), width, height, linewidth=2,
                             edgecolor=color_norm, facecolor='none')
            ax.add_patch(rect)

            # Add label
            plt.text(
                x1, y1 - 5, f"class_{int(label)}: {score:.2f}",
                color='white', fontsize=10,
                bbox=dict(facecolor=color_norm, alpha=0.8)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Clean up
    os.remove(temp_img_path)
    print("Test completed successfully!")

    return model


def parse_args_predict():
    """Parse command line arguments for prediction"""
    parser = argparse.ArgumentParser(
        description='Run object detection with ResNetFPN')

    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'],
                        help='Backbone architecture')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size for processing')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')

    return parser.parse_args()


if __name__ == "__main__":
    # If no args provided, run test function
    if not len(os.sys.argv) > 1:
        test_prediction()
    else:
        # Parse arguments and run main
        args = parse_args_predict()
        main_prediction(args)
