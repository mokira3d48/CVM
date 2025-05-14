import os
import logging
from dataclasses import dataclass
from argparse import ArgumentParser
from shutil import copy
from collections import defaultdict

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchinfo import summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("faster_rcnn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # Number of classes (your dataset classes + 1 for background)
        self.label_names.insert(0, '')

    def _load_class_names(self, file_path):
        """
        Class name loading
        ------------------

        Returns the class names list contained
        in YAML file located at file_path.

        :param file_path: The file path of the YAML file
        :type file_path: `str`
        :rtype: `list` of `str`
        """
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            if 'names' not in content:
                raise AttributeError(
                    "No class names list is not defined into YAML"
                    f" file at: {file_path}")
            if isinstance(content['names'], dict):
                names = list(content['names'].values())
            else:
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
            img_w = self.img_size
            img_h = self.img_size
            # num_classes = len(self.label_names)

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
                    class_ids.append(class_id + 1)

                    center_x = nx * img_w
                    center_y = ny * img_h
                    w = nw * img_w
                    h = nh * img_h
                    x = center_x - w / 2
                    y = center_y - h / 2

                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    bboxes.append([x1, y1, x2, y2])
                    # bboxes.append([nx, ny, nw, nh])
                except ValueError as e:
                    print(f"{e}")
                    continue

            class_ids = np.asarray(class_ids, dtype=np.int64) if class_ids \
                        else np.asarray([0,], dtype=np.int64)
            bboxes = np.asarray(bboxes, dtype=np.float32) if bboxes \
                     else np.asarray([[0., 0., img_w, img_h]], dtype=np.float32)
            return class_ids, bboxes

    def _class_id_verify(self, class_ids):
        """
        Class ID verifying
        ------------------

        :type class_ids: `np.ndarray`
        :rtype: `None`
        """
        classes_count = len(self.label_names)
        for class_id in class_ids.tolist():
            if not (0 <= class_id < classes_count):
                raise ValueError(
                    f"The class ID {class_id} is not in available classes.")

    def __len__(self):
        # return len(self.images)
        return 100

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
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
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
            'labels': torch.tensor(class_ids, dtype=torch.int64),
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
        }
        return image, target


class Model(nn.Module):
    _default_file_name = 'faster_rcnn'
    _backbones = {'resnet50': fasterrcnn_resnet50_fpn}

    @dataclass
    class Config:
        num_classes: int = 1000
        backbone: str = 'resnet50'
        im_size: int = 224

        def save(self, file_path):
            data = self.__dict__
            with open(file_path, mode='w', encoding='utf-8') as f:
                yaml.dump(data, f)

        def load(self, file_path):
            with open(file_path, mode='r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.__dict__.update(data)

    @classmethod
    def load(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such saved model file at {file_path}")
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')
        if not os.path.isfile(param_file):
            raise FileNotFoundError(
                f"No such model config file at {param_file}.")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"No such model file at {model_file}.")
        hparams = cls.Config()
        hparams.load(param_file)
        instance = cls(hparams)
        weights = torch.load(
            model_file, weights_only=True, map_location='cpu')
        instance.estimator.load_state_dict(weights)
        logger.info("Model weights loaded successfully!")
        return instance

    def __init__(self, config=None, pretrained=False):
        super().__init__()
        self.config = self.Config() if not config else config
        if self.config.backbone not in self._backbones:
            raise ValueError(
                f"The backbone named {self.config.backbone} is not supported.")
        estimator_function = self._backbones[self.config.backbone]
        if not pretrained:
            self.estimator = estimator_function(
                weights_backbone=None, weights=None)
        else:
            self.estimator = estimator_function(pretrained=True)

        # Get the number of input features for the classifier
        in_feat = self.estimator.roi_heads.box_predictor.cls_score.in_features

        # Replace the head of the model with a new one
        # (for the number of classes in your dataset)
        self.estimator.roi_heads.box_predictor = \
            FastRCNNPredictor(in_feat, self.config.num_classes + 1)

    @property
    def num_classes(self):
        """
        int: Number of the classes taking into account the background
        """
        return self.config.num_classes + 1

    def forward(self, inputs, target=None):
        """
        Forward pass
        ------------
        """
        return self.estimator(inputs, target)

    def device(self):
        return next(self.parameters()).device

    def summary(self, batch_size=1):
        """
        Model summary
        -------------
        """
        device = self.device()
        samples = torch.zeros(
            size=(batch_size, 3, self.config.im_size, self.config.im_size),
            device=device)
        summary(self.estimator, input_data=samples)

    def weights(self):
        return self.estimator.state_dict()

    def save(self, file_path):
        """
        Function to save model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        if not file_path:
            file_path = self._default_file_name
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.weights()
        torch.save(model_weights, model_file)
        self.config.save(param_file)


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


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union  (IoU) between two bounding boxes

    :param box1: Tensor of shape (4,) with [x1, y1, x2, y2] coordinates
    :param box2: Tensor of shape (4,) with [x1, y1, x2, y2] coordinates

    :type box1: `torch.Tensor`
    :type box2: `torch.Tensor`

    :returns: IoU value as a scalar tensor
    :rtype: `torch.Tensor`
    """
    # Get the coordinates of the intersection rectangle
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate Union area by using Formula: Union = A + B - Intersection
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area

    return iou


def calculate_precision_recall(
    pred_boxes,
    pred_labels,
    pred_scores,
    true_boxes,
    true_labels,
    iou_threshold,
    num_classes=None
):
    """
    Calculate precision and recall values for each class

    Args:
        pred_boxes (torch.Tensor): Tensor of shape (N, 4) with predicted
            bounding boxes [x1, y1, x2, y2]
        pred_labels: Tensor of shape (N,) with predicted class labels
        pred_scores: Tensor of shape (N,) with confidence scores
            for predictions
        true_boxes: Tensor of shape (M, 4) with ground truth bounding boxes
            [x1, y1, x2, y2]
        true_labels: Tensor of shape (M,) with ground truth class labels
        iou_threshold: IoU threshold to determine if detection is correct
        num_classes: Number of classes in the dataset

    Returns:
        precision_dict: Dictionary mapping class_id to list of precision values
        recall_dict: Dictionary mapping class_id to list of recall values
        gt_count_dict: Dictionary mapping class_id to number
            of ground truth instances
    """
    if num_classes is None:
        num_classes = max(torch.max(pred_labels).item(),
                          torch.max(true_labels).item()) + 1

    # Sort predictions by decreasing confidence
    indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[indices]
    pred_labels = pred_labels[indices]
    pred_scores = pred_scores[indices]

    # Initialize dictionaries to store precision-recall values for each class
    precision_dict = defaultdict(list)
    recall_dict = defaultdict(list)
    gt_count_dict = defaultdict(int)

    # Count ground truth instances per class
    for label in true_labels:
        gt_count_dict[label.item()] += 1

    # Initialize detection status for ground truth boxes
    detection_status = torch.zeros(len(true_boxes), dtype=torch.bool)

    # Initialize counter for true positives per class
    true_positives_per_class = defaultdict(int)

    # Evaluate each prediction
    cumulative_tps = defaultdict(list)
    cumulative_fps = defaultdict(list)

    for i, (box, label, score) in enumerate(
            zip(pred_boxes, pred_labels, pred_scores)):
        matched_gt = -1
        max_iou = iou_threshold

        # Find the ground truth box with the highest IoU
        for j, (gt_box, gt_label) in enumerate(zip(true_boxes, true_labels)):
            # Only compare boxes of the same class
            if label == gt_label and not detection_status[j]:
                iou = calculate_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt = j

        # Update statistics for the current class
        class_id = label.item()

        # If match found, count it as true positive
        if matched_gt >= 0:
            detection_status[matched_gt] = True
            true_positives_per_class[class_id] += 1

        # Count cumulative true positives and false positives
        cumulative_tps[class_id].append(true_positives_per_class[class_id])
        cumulative_fps[class_id].append(
            i + 1 - true_positives_per_class[class_id])

    # Calculate precision and recall for each class
    for class_id in range(num_classes):
        if class_id not in gt_count_dict:
            gt_count_dict[class_id] = 0

        if class_id not in cumulative_tps:
            precision_dict[class_id] = [1.0]  # No detections = 100% precision
            recall_dict[class_id] = [0.0]  # No detections = 0% recall
            continue

        # Convert cumulative TP and FP to precision and recall values
        tp_array = cumulative_tps[class_id]
        fp_array = cumulative_fps[class_id]

        if gt_count_dict[class_id] > 0:
            recalls = [tp / gt_count_dict[class_id] for tp in tp_array]
            precisions = [tp / (tp + fp) for tp, fp in zip(tp_array, fp_array)]
        else:
            recalls = [0.0] * len(tp_array)
            precisions = [0.0] * len(tp_array)

        # Add point at (0, 1) to make the PR curve start from precision=1
        precision_dict[class_id] = [1.0] + precisions
        recall_dict[class_id] = [0.0] + recalls

    return precision_dict, recall_dict, gt_count_dict


def calculate_average_precision(precision, recall):
    """
    Calculate Average Precision using the 11-point interpolation
    (or use the area under the PR curve directly)

    Args:
        precision (typing.List[float]): List of precision values
        recall (typing.List[float]): List of recall values

    Returns:
        float: Average Precision value
    """
    # Make sure precision and recall are paired correctly
    assert len(precision) == len(recall), (
        "Precision and recall lists must have the same length")

    # Ensure precision and recall are in the correct order (sorted by recall)
    if len(recall) > 1 and recall[0] > recall[1]:
        sorted_pairs = sorted(zip(recall, precision))
        recall = [r for r, _ in sorted_pairs]
        precision = [p for _, p in sorted_pairs]

    # Calculate area under PR curve using all points
    average_precision = 0
    for i in range(len(recall) - 1):
        average_precision += (recall[i + 1] - recall[i]) * precision[i + 1]

    return average_precision



class MAP:
    """
    Mean Average Precision
    ======================

    Args:
        threshold (float): The IOU threshold
        num_classes (int): Number of classes
    """
    def __init__(self, threshold, num_classes=None):
        self.threshold = threshold
        self.num_classes = num_classes

        self.pred_boxes_list = []
        self.pred_labels_list = []
        self.pred_scores_list = []
        self.true_boxes_list = []
        self.true_labels_list = []

    def calculate(self):
        """
        Calculate mAP across all classes and images

        Returns:
            float: mAP50 Mean Average Precision at IoU threshold of 0.5
            typing.Dict[int, float]: class_aps Dictionary mapping class_id
                to AP value
        """
        pred_boxes_list = self.pred_boxes_list
        pred_labels_list = self.pred_labels_list
        pred_scores_list = self.pred_scores_list
        true_boxes_list = self.true_boxes_list
        true_labels_list = self.true_labels_list

        if not pred_boxes_list:
            return 0.0, {}

        # Determine the number of classes if not provided
        if self.num_classes is None:
            max_pred_class = max(
                [torch.max(labels).item() if len(labels) > 0 else -1
                 for labels in pred_labels_list])
            max_true_class = max(
                [torch.max(labels).item() if len(labels) > 0 else -1
                 for labels in true_labels_list])
            self.num_classes = max(max_pred_class, max_true_class) + 1

        # Aggregate precision and recall dictionaries across all images
        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)
        gt_counts = defaultdict(int)

        # Process each image
        iteration = zip(
            pred_boxes_list, pred_labels_list, pred_scores_list,
            true_boxes_list, true_labels_list)
        for elm in iteration:
            pred_boxes, pred_labels, pred_scores, true_boxes, true_labels = elm

            # Skip if there are no predictions or ground truth
            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                continue

            # Calculate precision and recall for this image
            ret = calculate_precision_recall(
                pred_boxes, pred_labels, pred_scores,
                true_boxes, true_labels,
                iou_threshold=self.threshold,
                num_classes=self.num_classes
            )
            precision_dict, recall_dict, gt_count_dict = ret

            # Update global dictionaries
            for class_id in range(self.num_classes):
                if class_id in precision_dict:
                    all_precisions[class_id].extend(precision_dict[class_id])
                    all_recalls[class_id].extend(recall_dict[class_id])
                gt_counts[class_id] += gt_count_dict[class_id]

        # Calculate AP for each class
        class_aps = {}
        valid_classes = 0
        sum_ap = 0.0

        for class_id in range(self.num_classes):
            # Skip classes with no ground truth instances
            if gt_counts[class_id] == 0:
                class_aps[class_id] = 0.0
                continue

            # Sort precision and recall values by recall
            if len(all_recalls[class_id]) > 0:
                paired = sorted(
                    zip(all_recalls[class_id], all_precisions[class_id]))
                recalls_sorted = [r for r, _ in paired]
                precisions_sorted = [p for _, p in paired]

                # Calculate AP for this class
                ap = calculate_average_precision(precisions_sorted,
                                                 recalls_sorted)
                class_aps[class_id] = ap
                sum_ap += ap
                valid_classes += 1
            else:
                class_aps[class_id] = 0.0

        # Calculate mAP
        mapx = sum_ap / valid_classes if valid_classes > 0 else 0.0
        return mapx, class_aps

    def update(self, predictions, targets):
        pred_boxes_list = [predict['boxes'] for predict in predictions]
        pred_labels_list = [predict['labels'] for predict in predictions]
        pred_scores_list = [predict['scores'] for predict in predictions]
        true_boxes_list = [target['boxes'] for target in targets]
        true_labels_list = [target['labels'] for target in targets]

        self.pred_boxes_list.extend(pred_boxes_list)
        self.pred_labels_list.extend(pred_labels_list)
        self.pred_scores_list.extend(pred_scores_list)
        self.true_boxes_list.extend(true_boxes_list)
        self.true_labels_list.extend(true_labels_list)

        return (
            pred_boxes_list, pred_labels_list, pred_scores_list,
            true_boxes_list, true_labels_list)

    def clear(self):
        self.pred_boxes_list.clear()
        self.pred_labels_list.clear()
        self.pred_scores_list.clear()
        self.true_boxes_list.clear()
        self.true_labels_list.clear()

    def __call__(self):
        maps, class_aps = self.calculate()
        return maps, class_aps


class InferenceStorage:

    @classmethod
    def load(cls, file_path):
        data = torch.load(file_path, weights_only=False, map_location='cpu')
        if "format" not in data or data['format'] != 'faster_rcnn':
            raise ValueError("Not supported formatting.")
        instance = cls(file_path)
        instance.label_predict = data['label_predict']
        instance.label_target = data['label_target']
        instance.bbox_predict = data['bbox_predict']
        instance.bbox_target = data['bbox_target']

    def __init__(self, file_path):
        self.file_path = file_path
        self.label_predict = None
        self.label_target = None
        self.bbox_predict = None
        self.bbox_target = None

    def _add_inf(self, label_predict, label_target, bbox_predict, bbox_target):
        """
        Adding function
        ---------------

        """
        if self.label_predict is not None:
            self.label_predict = torch.cat([self.label_predict, label_predict])
            self.label_target = torch.cat([self.label_target, label_target])
            self.bbox_predict = torch.cat([self.bbox_predict, bbox_predict])
            self.bbox_target = torch.cat([self.bbox_target, bbox_target])
        else:
            self.label_predict = label_predict
            self.label_target = label_target
            self.bbox_predict = bbox_predict
            self.bbox_target = bbox_target

    def add(self, predictions, targets):
        batch_size = len(predictions)
        for b in range(batch_size):
            self._add_inf(
                predictions[b]['labels'], targets[b]['labels'],
                predictions[b]['boxes'], targets[b]['boxes'])

    def save(self):
        """
        Saving function
        ---------------

        """
        if self.label_predict is not None:
            data = {
                "label_predict": self.label_predict,
                "label_target": self.label_target,
                "bbox_predict": self.bbox_predict,
                "bbox_target": self.bbox_target,
                "format": "faster_rcnn",
            }
            torch.save(data, self.file_path)

    def clear(self):
        self.label_predict = None
        self.label_target = None
        self.bbox_predict = None
        self.bbox_target = None


class Trainer(Model):
    """
    Training model
    ==============
    """
    @classmethod
    def load(cls, model_file=None, checkpoint_file=None):
        instance = None
        if model_file:
            instance = super().load(model_file)

        if not instance and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file, map_location='cpu', weights_only=False)
            config = cls.Config()
            if 'model_config' in checkpoint:
                config.__dict__.update(checkpoint['model_config'])
            instance = cls(config)
            if 'model_state_dict' in checkpoint:
                instance.estimator.load_state_dict(
                    checkpoint['model_state_dict'])

        return instance

    def __init__(self, config=None):
        super().__init__(config)

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None

        self.train_losses = {}
        self.loss_classifier = AvgMeter()
        self.loss_box_reg = AvgMeter()
        self.loss_objectness = AvgMeter()
        self.loss_rpn_box_reg = AvgMeter()

        self.inference_store = None
        self.mAP50 = None
        self.mAP75 = None
        self.mAP95 = None

        self.gas = 128  # Gradiant Accumulation Steps
        self.seed = 42  # Seed number for random generation
        self.num_epochs = 1
        self.batch_size = 1

        self.gac = 0  # Gradient Accumulation Count

        self.checkpoint_dir = "checkpoints"
        self.resume_ckpt = None
        self.epoch = 1

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

        dataset_dir = args.dataset_dir
        train_ds_dir = os.path.join(dataset_dir, 'train')
        val_ds_dir = os.path.join(dataset_dir, 'val')
        ds_config_file = os.path.join(dataset_dir, 'data.yaml')
        if not os.path.isdir(train_ds_dir):
            raise FileNotFoundError(
                f"No such training set directory at {train_ds_dir}")
        if not os.path.isdir(val_ds_dir):
            raise FileNotFoundError(
                f"No such validation set directory at {val_ds_dir}")
        if not os.path.isfile(ds_config_file):
            raise FileNotFoundError(
                f"No such dataset config file at {ds_config_file}")
        image_size = args.image_size
        train_dataset = Dataset(train_ds_dir, ds_config_file, image_size)
        val_dataset = Dataset(val_ds_dir, ds_config_file, image_size)

        # Create data loaders
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)))
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)))

        # Set up the optimizer
        params = [p for p in self.estimator.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay)

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1)

        self.resume_ckpt = args.resume
        self.checkpoint_dir = args.checkpoint_dir

        self.inference_store = InferenceStorage(args.inference_store)
        self.mAP50 = MAP(0.50, len(val_dataset.label_names) + 1)
        self.mAP75 = MAP(0.75, len(val_dataset.label_names) + 1)
        self.mAP95 = MAP(0.95, len(val_dataset.label_names) + 1)

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
    def _print_losses(avg_meters):
        """
        Function of average meter losses
        """
        string = ''
        for key, val in avg_meters.items():
            string += f"{key}: {val:.8f} "
        return string

    @staticmethod
    def _add_to_epoch_results(epc_res, new_res):
        for key, val in epc_res.items():
            if key in epc_res:
                epc_res[key].append(new_res[key])
            else:
                epc_res[key] = [new_res[key]]

    def checkpoint(self, **kwargs):
        checkpoint = {
            "model_config": self.config.__dict__,
            "model_state_dict": self.estimator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "epoch": self.epoch,
            "train_losses": self.train_losses,
            **kwargs
        }

        file_path1 = os.path.join(
            self.checkpoint_dir, "checkpoint.pth")
        file_path2 = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self.epoch}.pth")
        torch.save(checkpoint, file_path1)
        copy(file_path1, file_path2)
        logger.info("Checkpoint done")

        if self.epoch > 2:
            old_checkpoint_file = os.path.join(
                self.checkpoint_dir, f"checkpoint_{self.epoch - 2}.pth")
            if os.path.isfile(old_checkpoint_file):
                os.remove(old_checkpoint_file)

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
        self.epoch = ckpt_data['epoch']
        self.train_losses = ckpt_data['train_losses']
        logger.info(f"Checkpoint loaded successfully from {self.resume_ckpt}!")

    def train_step(self, images, targets, write_fn, optimize=False):
        """
        Training method on one batch
        """
        # Forward pass and loss compute
        loss_dict = self.forward(images, targets)
        losses = list(loss_dict.values())
        loss = losses[0]
        for i in range(1, len(losses)):
            loss += losses[i]

        # Backward pass
        loss.backward()

        self.loss_classifier += loss_dict['loss_classifier'].item()
        self.loss_box_reg += loss_dict['loss_box_reg'].item()
        self.loss_objectness += loss_dict['loss_objectness'].item()
        self.loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        self.gac += len(images)
        if self.gac >= self.gas or optimize:
            self.optimizer.step()
            self.optimizer.zero_grad()

            write_fn(
                "Optim step"
                f" - loss classifier: {self.loss_classifier.avg():.8f}"
                f" - loss box reg: {self.loss_box_reg.avg():.8f}"
                f" - loss objectness: {self.loss_objectness.avg():.8f}"
                f" - loss rpn box reg: {self.loss_rpn_box_reg.avg():.8f}")
            self.gac = 0

    def train_one_epoch(self):
        """
        Method of training on one epoch
        """
        device = self.device()
        loss_data = {}

        length = len(self.train_loader)
        iterator = tqdm(self.train_loader, desc="train")
        write_fn = iterator.write

        self.train()
        self.optimizer.zero_grad()
        for index, (images, targets) in enumerate(iterator):
            images = list(image.to(device) for image in images)  # noqa
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets]

            # At last iteration, the gradient accumulation count can not
            # be equal to gradient accumulation step, so we must perform
            # optimization step when we are at the last iteration (length - 1)
            last_index = index >= (length - 1)
            self.train_step(images, targets, write_fn, last_index)

            loss_data = {
                "loss_classifier": self.loss_classifier.avg(),
                "loss_box_reg": self.loss_box_reg.avg(),
                "loss_objectness": self.loss_objectness.avg(),
                "loss_rpn_box_reg": self.loss_rpn_box_reg.avg()}
            iterator.set_postfix(loss_data)

        return loss_data

    def validate(self):
        device = self.device()
        self.inference_store.clear()
        self.mAP50.clear()
        self.mAP95.clear()

        metrics = {}

        self.eval()
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="val")
            # boxes_predictions = []
            # labels_predictions = []

            for images, targets in iterator:
                images = list(img.to(device) for img in images)

                # Forward pass
                predictions = self.forward(images)
                # iterator.write(str(predictions))

                # self.inference_store.add(predictions, targets)
                self.mAP50.update(predictions, targets)
                self.mAP75.update(predictions, targets)
                self.mAP95.update(predictions, targets)

                map50, _ = self.mAP50()
                map75, _ = self.mAP75()
                map95, _ = self.mAP95()
                metrics = {"mAP50": map50,
                           "mAP75": map75,
                           "mAP95": map95}
                iterator.set_postfix(metrics)

                # boxes_predictions.append(predictions[0]['boxes'])
                # labels_predictions.append(predictions[0]['labels'])

        return metrics

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

        for epoch in range(self.epoch, (self.num_epochs + 1)):
            self.epoch = epoch
            logger.info(f'Epoch: {epoch} / {self.num_epochs}')

            train_losses = self.train_one_epoch()

            # Update the learning rate
            self.lr_scheduler.step()

            # Add losses to train losses epochs
            self._add_to_epoch_results(self.train_losses, train_losses)

            logger.info(f'{self._print_losses(train_losses)}')

            val_losses = self.validate()
            # self.inference_store.save()

            self.checkpoint(train_losses=train_losses)

            if epoch != self.num_epochs:
                # Epochs are remaining
                logger.info("-" * 80)


def parse_argument():
    """
    Command line argument parsing
    """
    parser = ArgumentParser(prog="ResNet FasterRCNN Train")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-d', '--dataset-dir', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument('-imsz', '--image-size', type=int, default=640)
    parser.add_argument(
        '--backbone', type=str, choices=['resnet50'], default='resnet50')

    parser.add_argument('-n', '--epochs', type=int, default=2)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('-gas', type=int, default=128)

    parser.add_argument('-r', "--resume", type=str)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('-m', '--model-file', type=str)
    parser.add_argument('--inference-store', type=str, default="store.pth")

    args = parser.parse_args()
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    return args


def main():
    args = parse_argument()

    model_file = args.model_file
    checkpoint = args.resume
    model = None
    if checkpoint:
        model = Trainer.load(checkpoint_file=checkpoint)
    if not model and model_file:
        model = Trainer.load(model_file)
    if not model:
        hparams = Trainer.Config()
        hparams.backbone = args.backbone
        hparams.im_size = args.image_size

        # Open dataset config
        # and get number of class names
        ds_config_file = os.path.join(args.dataset_dir, 'data.yaml')
        if not os.path.isfile(ds_config_file):
            logger.error(f"No such config file at {ds_config_file}")
            exit(2)
        with open(ds_config_file, mode='r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            nc = len(config['names'])
            hparams.num_classes = nc

        model = Trainer(hparams)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.compile(args)
    model.summary()
    model.fit()


if __name__ == '__main__':
    main()
