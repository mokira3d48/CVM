import os
import logging
from dataclasses import dataclass
from argparse import ArgumentParser
from shutil import copy

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
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(class_ids, dtype=torch.int64)
        }
        return image, target


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
        instance.load_state_dict(weights)
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
            FastRCNNPredictor(in_feat, self.config.num_classes)

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
            if 'model_hparams' in checkpoint:
                config.__dict__.update(checkpoint['model_hparams'])
            instance = cls(config)
            if 'model_state_dict' in checkpoint:
                instance.load_state_dict(checkpoint['model_state_dict'])

        return instance

    def __init__(self, config=None):
        super().__init__(config)

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None

        self.train_losses = {}
        self.val_losses = {}

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
    def _print_avg_meter_losses(avg_meters):
        """
        Function of average meter losses
        """
        string = ''
        for key, val in avg_meters.items():
            string += f"{key}: {val.avg():.8f} "
        return string

    @staticmethod
    def _add_to_epoch_results(epc_res, new_res):
        for key, val in epc_res.items():
            if key in epc_res:
                epc_res[key].append(new_res[key].avg())
            else:
                epc_res[key] = [new_res[key].avg()]

    def checkpoint(self, **kwargs):
        checkpoint = {
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

    def train_step(self, images, targets):
        """
        Training method on one batch
        """
        # Zero the gradients
        self.optimizer.zero_grad()

        # Forward pass
        loss_dict = self.forward(images, targets)
        losses = list(loss_dict.values())
        loss = losses[0]
        for i in range(1, len(losses)):
            loss += losses[i]

        # Backward pass
        loss.backward()

        self.gac += len(images)
        self.optimizer.step()
        return {key: value.item()
                for key, value in loss_dict.items()}

    def train_one_epoch(self):
        """
        Method of training on one epoch
        """
        device = self.device()
        train_losses = {}

        self.train()
        iterator = tqdm(self.train_loader, desc="train")
        for index, (images, targets) in enumerate(iterator):
            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets]

            losses = self.train_step(images, targets)
            iterator.set_postfix(losses)

            self._update_losses(losses, train_losses)

        return train_losses

    def validate(self):
        device = self.device()
        self.eval()
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="val")
            boxes_predictions = []
            labels_predictions = []

            for images, _ in iterator:
                images = list(img.to(device) for img in images)

                # Forward pass
                predictions = self.forward(images)
                # iterator.write(str(predictions))

                boxes_predictions.append(predictions[0]['boxes'])
                labels_predictions.append(predictions[0]['labels'])

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

            logger.info(f'{self._print_avg_meter_losses(train_losses)}')

            val_losses = self.validate()

            self.checkpoint(train_losses=train_losses)

            logger.info("-" * 80)

def main_():
    """
    Main function to run training process
    """
    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = Dataset("dataset/train", "dataset/data.yaml", (128, 128))
    val_dataset = Dataset("dataset/val", "dataset/data.yaml", (128, 128))

    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                   collate_fn=(lambda x: tuple(zip(*x))))
    valid_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                   collate_fn=(lambda x: tuple(zip(*x))))

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
