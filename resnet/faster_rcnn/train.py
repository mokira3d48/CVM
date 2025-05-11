import os
import logging
from dataclasses import dataclass

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
            names = list(content['names'].values())
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
        return 10

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


class Model(nn.Module):
    _backbones = {'resnet50': fasterrcnn_resnet50_fpn}

    @dataclass
    class Config:
        class_names = [0, 1, 3, 4, 5]
        backbone = 'resnet50'
        im_size = 224

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
        num_classes = len(self.config.class_names)
        self.estimator.roi_heads.box_predictor = \
            FastRCNNPredictor(in_feat, num_classes)

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


class Trainer(Model):

    def __init__(self, config=None):
        super().__init__(config)

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None

        self.train_losses = {}

        self.gas = 128  # Gradiant Accumulation Steps
        self.seed = 42  # Seed number for random generation


    def compile(self, args):
        """
        Initialization of training process
        ----------------------------------

        :type args: `argparse.Namespace`
        :rtype: `None`
        """
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
        image_size = args.im_size
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

    def fit(self):
        """
        Training process
        ----------------

        Run the training loop and returns the results
        formatted as dictionary.

        :rtype: `dict`
        """
        ...


def main():
    """
    Main function to run training proceess
    """
    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = Dataset("dataset/train", "dataset/data.yaml", (128, 128))
    val_dataset = Dataset("dataset/val", "dataset/data.yaml", (128, 128))

    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                   collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                   collate_fn=lambda x: tuple(zip(*x)))

    config = Model.Config()
    config.class_names = train_dataset.label_names
    config.im_size = 128
    model = Model(config)

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')
    model.to(device)
    model.summary()

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        iterator = tqdm(train_loader, desc="train")

        # Training loop
        for index, (images, targets) in enumerate(iterator):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = list(loss_dict.values())
            loss = losses[0]
            for i in range(1, len(losses)):
                loss += losses[i]

            # print("loss:", loss)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterator.set_postfix(
                {name:value.item() for name, value in loss_dict.items()})

        # Update the learning rate
        lr_scheduler.step()
        print(
            f'Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_loader)}')

        # Test on a new image
        model.eval()
        with torch.no_grad():
            iterator = tqdm(valid_loader, desc="val")
            val_loss = 0
            for images, targets in iterator:
                images = list(img.to(device) for img in images)

                # Forward pass
                loss_dict = model(images, targets)
                # iterator.write(str(loss_dict))
                # losses = list(loss_dict.values())
                # loss = losses[0]
                # for i in range(1, len(losses)):
                #     loss += losses[i]
                # val_loss += loss.item()
                #
                # iterator.set_postfix(
                #     {name: value.item() for name, value in loss_dict.items()})

                # Example: print the bounding boxes and labels
                # for the first image
                # print(predictions[0]['boxes'])
                # print(predictions[0]['labels'])
            print(
                f'Epoch: {epoch + 1}, Val Loss: {val_loss / len(valid_loader)}')

        torch.save(model.state_dict(), 'frcnn_numpr.pth')
    print("Training complete!")


def parse_argument():
    """
    Command line argument parsing
    """
    ...


if __name__ == '__main__':
    main()
