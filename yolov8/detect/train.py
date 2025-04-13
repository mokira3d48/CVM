# -*- encoding: utf-8 -*-

__version__ = '0.1.0'
__author__ = 'Arnold Mokira'

import os
import math
import random
import logging
from argparse import ArgumentParser, FileType
from dataclasses import dataclass

import yaml
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch import nn
from torch.utils import data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolov8_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_seed(s=42):
    """
    Setup random seed.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Conv(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        groups=1, activation=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.act(x)
        return y


class Bottleneck(nn.Module):
    """Bottleneck: staack of 2 COnv with shortcut connnection (True/False)"""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        x_in = x # for residual connection
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in
        return x


class C2f(nn.Module):
    """C2f: Conv + bottleneck*N+ Conv"""

    def __init__(
        self, in_channels, out_channels, num_bottlenecks, shortcut=True
    ):
        super().__init__()
        
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks
        self.conv1=Conv(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # sequence of bottleneck layers
        self.m = nn.ModuleList(
            [Bottleneck(self.mid_channels, self.mid_channels)
             for _ in range(num_bottlenecks)]
        )
        self.conv2 = Conv(
            in_channels=(num_bottlenecks + 2) * out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)

        # split x along channel dimension
        x1, x2 = x[:, :(x.shape[1] // 2), :, :], x[:, (x.shape[1] // 2):, :, :]

        # list of outputs
        outputs = [x1, x2] # x1 is fed through the bottlenecks

        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)    # [bs, 0.5c_out, w, h]
            outputs.insert(0, x1)

        outputs = torch.cat(outputs,dim=1)
        #: [bs, 0.5c_out(num_bottlenecks + 2), w, h]
        out = self.conv2(outputs)
        return out


class SPPF(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5):
        # kernel_size = size of maxpool
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # concatenate outputs of maxpool and feed to conv2
        self.conv2 = Conv(
            4 * hidden_channels, out_channels, kernel_size=1, stride=1,
            padding=0)

        # maxpool is applied at 3 different sacles
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
            dilation=1, ceil_mode=False)

    def forward(self,x):
        x = self.conv1(x)

        # apply maxpooling at diffent scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # concantenate 
        y = torch.cat([x,y1,y2,y3], dim=1)

        # final conv
        y = self.conv2(y)
        return y


def yolo_params(version):
    """Returns d,w,r based on version"""
    if version == 'n':
        return 1/3, 1/4, 2.0
    elif version == 's':
        return 1/3, 1/2, 2.0
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0


class Backbone(nn.Module):
    """backbone = DarkNet53"""

    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        d, w, r = yolo_params(version)

        # conv layers
        self.conv_0 = Conv(
            in_channels, int(64 * w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(
            int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(
            int(128 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(
            int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(
            int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1)

        # c2f layers
        self.c2f_2 = C2f(
            int(128 * w), int(128 * w), num_bottlenecks=int(3 * d),
            shortcut=True)
        self.c2f_4 = C2f(
            int(256 * w), int(256 * w), num_bottlenecks=int(6 * d),
            shortcut=True)
        self.c2f_6 = C2f(
            int(512 * w), int(512 * w), num_bottlenecks=int(6 * d),
            shortcut=True)
        self.c2f_8 = C2f(
            int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d),
            shortcut=True)

        # sppf
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))
    
    def forward(self,x):
        x = self.conv_0(x)
        x = self.conv_1(x)

        x = self.c2f_2(x)
        x = self.conv_3(x)

        out1 = self.c2f_4(x) # keep for output
        x = self.conv_5(out1)

        out2 = self.c2f_6(x) # keep for output

        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)

        return out1, out2, out3


class Upsample(nn.Module):
    """
    Upsample = nearest-neighbor interpolation with scale_factor=2.
    It doesn't have trainable paramaters.
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
 
    def forward(self,x):
        out = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return out


class Neck(nn.Module):
    """The neck comprises of Upsample + C2f with"""

    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample() # no trainable parameters
        self.c2f_1 = C2f(
            in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w),
            num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_2 = C2f(
            in_channels=int(768 * w), out_channels=int(256 * w),
            num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_3 = C2f(
            in_channels=int(768 * w), out_channels=int(512 * w),
            num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_4 = C2f(
            in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w * r),
            num_bottlenecks=int(3 * d), shortcut=False)

        self.cv_1 = Conv(
            in_channels=int(256 * w), out_channels=int(256 * w),
            kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(
            in_channels=int(512 * w), out_channels=int(512 * w),
            kernel_size=3, stride=2, padding=1)

    def forward(self,x_res_1,x_res_2,x):    
        # x_res_1, x_res_2, x = output of backbone
        res_1 = x  # for residual connection

        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)

        res_2 = self.c2f_1(x)  # for residual connection

        x = self.up(res_2)

        x = torch.cat([x, x_res_1], dim=1)

        out_1 = self.c2f_2(x)
        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out_2)

        x = torch.cat([x,res_1],dim=1)
        out_3 = self.c2f_4(x)
        return out_1, out_2, out_3


class DFL(nn.Module):
    """
    DFL considers the predicted bbox coordinates as a probability distribution.
    At inference time, it samples from the distribution to get refined
    coordinates (x, y, w, h). For example, to predict coordinate
    x in the normalized range [0, 1]:

    1. DFL uses 16 bins which are equally spaced in [0, 1] bin length 1/16.
    2. The model outputs 16 numbers which corresponds to probabilities
        that x falls in these bins, for example, [0, 0, ..., 9/10, 1/10].
    3. Prediction for x = mean value = 9/10 x 1/16 + 1/10 x 1 = 0.94375.
    """
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch        

        self.conv = nn.Conv2d(in_channels=ch,
                              out_channels=1,
                              kernel_size=1,
                              bias=False)
        self.conv = self.conv.requires_grad_(False)

        # initialize conv with [0,...,ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
        #: DFL only has ch parameters

    def forward(self, x):
        # x must have num_channels = 4 * ch: x = [bs, 4 * ch, c]
        b, c, a = x.shape  # c = 4 * ch
        x = x.view(b, 4, self.ch, a).transpose(1, 2)  # [bs, ch, 4, a]

        # take softmax on channel dimension
        # to get distribution probabilities
        x = x.softmax(1)  # [b, ch, 4, a]
        x = self.conv(x)  # [b, 1, 4, a]
        out = x.view(b, 4, a)  # [b, 4, a]
        return out


class Head(nn.Module):
    """
    Consist of 3 modules: (1) bbox coordinates, (2) classification scores,
    (3) distribution focal loss (DFL).
    """
    def __init__(self, version, ch=16, num_classes=80):

        super().__init__()
        self.ch = ch                            # dfl channels
        self.coordinates = self.ch * 4          # number of bounding box coordinates 
        self.nc = num_classes                   # 80 for COCO
        self.no = self.coordinates + self.nc    # number of outputs per anchor box

        self.stride = torch.zeros(3)          # strides computed during build

        d, w, r = yolo_params(version=version)
        
        # for bounding boxes
        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(
                    int(256 * w), self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                Conv(
                    self.coordinates, self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                nn.Conv2d(
                    self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(
                    int(512 * w), self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                Conv(
                    self.coordinates, self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                nn.Conv2d(
                    self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(
                    int(512 * w * r), self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                Conv(
                    self.coordinates, self.coordinates, kernel_size=3, stride=1,
                    padding=1),
                nn.Conv2d(
                    self.coordinates, self.coordinates, kernel_size=1, stride=1)
            )
        ])

        # for classification
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),

            nn.Sequential(
                Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),

            nn.Sequential(
                Conv(
                    int(512 * w * r), self.nc, kernel_size=3, stride=1,
                    padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1))
        ])

        # dfl
        self.dfl = DFL()

    def make_anchors(self, x, strides, offset=0.5):
        """
        x = list of feature maps: x=[x[0],...,x[N-1]],
        in our case N= num_detection_heads=3 each having shape [bs, ch, w, h]
        each feature map x[i] gives output[i] = w * h anchor
        coordinates + w * h stride values.

        strides = list of stride values indicating how much
        the spatial resolution of the feature map is reduced
        compared to the original image.
        """
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            # x coordinates of anchor centers
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            # y coordinates of anchor centers
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            # all anchor centers
            sy, sx = torch.meshgrid(sy, sx) 
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)

    def forward(self, x):
        # x = output of Neck = list of 3 tensors with different resolution
        # and different channel dim
        # x[0]=[bs, ch0, w0, h0], x[1]=[bs, ch1, w1, h1], x[2]=[bs,ch2, w2, h2] 

        for i in range(len(self.box)):         # detection head i
            box = self.box[i](x[i])            # [bs, num_coordinates,w,h]
            cls = self.cls[i](x[i])            # [bs, num_classes,w,h]
            x[i] = torch.cat((box,cls),dim=1)
            #: [bs, num_coordinates + num_classes, w, h]

        # in training, no dfl output
        if self.training:
            return x  # [3, bs, num_coordinates + num_classes, w, h]

        # in inference time, dfl produces refined bounding box coordinates
        anchors, strides = (
            i.transpose(0, 1) for i in self.make_anchors(x, self.stride)
        )

        # concatenate predictions from all detection layers
        #: [bs, 4*self.ch + self.nc, sum_i(h[i]w[i])]
        x = torch.cat(
            [i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)

        # split out predictions for box and cls
        #   box = [bs, 4 × self.ch, sum_i(h[i] w[i])]
        #   cls = [bs, self.nc, sum_i(h[i] w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2×self.ch,sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)


class YOLOv8(nn.Module):

    def __init__(self, version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version)

    def forward(self, x):
        x = self.backbone(x)              # return out1, out2, out3
        x = self.neck(x[0], x[1], x[2])   # return out_1, out_2,out_3
        out = self.head(list(x))
        return out


def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.7):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)
            x = x[conf.view(-1) > confidence_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    """
    Convert nx4 boxes
    from [x, y, w, h] normalized to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right
    """
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized
    # where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


################################################################################
# DATASET
################################################################################

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

def resample():
    choices = (cv.INTER_AREA,
               cv.INTER_CUBIC,
               cv.INTER_LINEAR,
               cv.INTER_NEAREST,
               cv.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv.merge((cv.LUT(h, lut_h), cv.LUT(s, lut_s), cv.LUT(v, lut_v)))
    cv.cvtColor(hsv, cv.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv.resize(
            image, dsize=pad,
            interpolation=resample() if augment else cv.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    #: aspect ratio

    out = (w2 > 2) \
          & (h2 > 2) \
          & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) \
          & (aspect_ratio < 100)
    return out


def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    center = np.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = np.eye(3)

    # Rotation and Scale
    rotate = np.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = np.eye(3)
    x1 = random.uniform(-params['shear'], params['shear']) * math.pi / 180
    x2 = random.uniform(-params['shear'], params['shear']) * math.pi / 180
    shear[0, 1] = math.tan(x1)
    shear[1, 0] = math.tan(x2)

    # Translation
    translate = np.eye(3)
    x1 = random.uniform(
        0.5 - params['translate'], 0.5 + params['translate']) * w
    x2 = random.uniform(
        0.5 - params['translate'], 0.5 + params['translate']) * h
    translate[0, 2] = x1
    translate[1, 2] = x2

    # Combined rotation matrix, order of operations (right to left)
    # is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():
        # image changed
        image = cv.warpAffine(
            image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(label)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        #: x1y1, x2y2, x1y2, x2y1

        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
        box = box.reshape(4, n).T

        # clip
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        # filter candidates
        indices = candidates(box1=label[:, 1:5].T * s, box2=box.T)

        label = label[indices]
        label[:, 1:5] = box[indices]

    return image, label


def mix_up(image1, label1, image2, label2):
    """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf"""
    alpha = np.random.beta(a=32.0, b=32.0)
    #: mix-up ratio, alpha=beta=32.0

    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = np.concatenate((label1, label2), 0)
    return image, lab


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations

            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]

            bbox = albumentations.BboxParams('yolo', ['class_labels'])
            self.transform = albumentations.Compose(transforms, bbox)

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = numpy.array(x['bboxes'])
            cls = numpy.array(x['class_labels'])
        return image, box, cls


class Dataset(data.Dataset):

    def __init__(self, images_dir, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size

        # Read labels
        samples = self.load_label(images_dir)
        print(samples)
        self.labels = list(samples.values())
        self.filenames = list(samples.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)

        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            # Load MOSAIC
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = mix_up(
                    mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(
                    label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        return (
            torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl))

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            interp = resample() if self.augment else cv2.INTER_LINEAR
            image = cv.resize(
                image, dsize=(int(w * r), int(h * r)),
                interpolation=interp)
        return image, (h, w)

    def load_mosaic(self, index, params):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = np.full(
            (self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = \
             (None, None, None, None, None, None, None, None)

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            if i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            if i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            if i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            pad_w = x1a - x1b
            pad_h = y1a - y1b
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # Labels
            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(
                    label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        # Concat/clip labels
        label4 = np.concatenate(label4, 0)
        for x in label4[:, 1:]:
            numpy.clip(x, 0, 2 * self.input_size, out=x)

        # Augment
        image4, label4 = random_perspective(image4, label4, params, border)
        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets

    @staticmethod
    def load_label(images_dir):
        ds_dir = os.path.dirname(images_dir)
        labels_dir = os.path.join(ds_dir, 'labels')

        path = f'{os.path.dirname(images_dir)}.cache'
        if os.path.exists(path):
            logger.info(f"Cache file found at {path}.")
            samples = torch.load(path, weights_only=False)
            return samples
        x = {}
        filenames = os.listdir(images_dir)
        for filename in filenames:
            filepath = os.path.join(images_dir, filename)
            try:
                # verify images
                with open(filepath, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), (
                    f'image size {shape} <10 pixels')
                assert image.format.lower() in FORMATS, (
                    f'invalid image format {image.format}')

                # verify labels
                filename_split = filename.split('.')
                filename = '.'.join(filename_split[:-1])
                label_file = filename + '.txt'
                label_file = os.path.join(labels_dir, label_file)

                if os.path.isfile(label_file):
                    with open(label_file) as f:
                        label = f.read().strip().splitlines()
                        label = [x.split() for x in label if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                            logger.warning(
                                f"Label duplicated removed for {filename}")
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
            except FileNotFoundError:
                label = np.zeros((0, 5), dtype=np.float32)
            except AssertionError:
                continue
            x[filepath] = label
        torch.save(x, path)
        return x


################################################################################
# TRAINING PROCESS
################################################################################

@dataclass
class Config:
    min_lr = 0.000100000000    # initial learning rate
    max_lr = 0.010000000000    # maximum learning rate
    momentum = 0.9370000000    # SGD momentum/Adam beta1
    weight_decay = 0.000500    # optimizer weight decay
    warmup_epochs = 3.00000    # warmup epochs
    box = 7.500000000000000    # box loss gain
    cls = 0.500000000000000    # cls loss gain
    dfl = 1.500000000000000    # dfl loss gain
    hsv_h = 0.0150000000000    # image HSV-Hue augmentation (fraction)
    hsv_s = 0.7000000000000    # image HSV-Saturation augmentation (fraction)
    hsv_v = 0.4000000000000    # image HSV-Value augmentation (fraction)
    degrees = 0.00000000000    # image rotation (+/- deg)
    translate = 0.100000000    # image translation (+/- fraction)
    scale = 0.5000000000000    # image scale (+/- gain)
    shear = 0.0000000000000    # image shear (+/- deg)
    flip_ud = 0.00000000000    # image flip up-down (probability)
    flip_lr = 0.50000000000    # image flip left-right (probability)
    mosaic = 1.000000000000    # image mosaic (probability)
    mix_up = 0.000000000000    # image mix-up (probability)

    def save(self, file_path):
        data = self.__dict__
        with open(file_path, mode='w', encoding='utf-8') as f:
            yaml.dump(data, f)

    def load(self, file):
        data = yaml.safe_load(file)
        self.__dict__.update(data)


def dataset_prepare(args):
    with open(args.dataset, mode='r', encoding='utf-8') as f:
        ds_config = yaml.safe_load(f)

    ds_root_dir = os.path.dirname(args.dataset)
    train_dir = os.path.join(ds_root_dir, ds_config['train'])
    val_dir = os.path.join(ds_root_dir, ds_config['val'])
    test_dir = os.path.join(ds_root_dir, ds_config['test'])

    config = Config()
    if args.config:
        config.load(args.config)
    params = config.__dict__

    logger.info(f"Images of training set: {train_dir}")
    logger.info(f"Images of validation set: {val_dir}")
    logger.info(f"Images of test set: {test_dir}")
    
    train_dataset = Dataset(
        train_dir, input_size=640, params=params, augment=False)
    val_dataset = Dataset(
        val_dir, input_size=640, params=params, augment=False)
    test_dataset = Dataset(
        test_dir, input_size=640, params=params, augment=False)


def model_train(args):
    """Main function to train model"""
    dataloaders = dataset_prepare(args)


def main():
    """Main function to start model training"""
    parser = ArgumentParser(prog="YOLOv8 Training")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="The path to the YAML file of the dataset.")
    parser.add_argument('-c', '--config', type=FileType('r'),
                        help="The path to the YAML file of training config.")
    args = parser.parse_args()
    model_train(args)

if __name__ == '__main__':
    main()
