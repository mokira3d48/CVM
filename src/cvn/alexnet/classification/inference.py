#!/usr/bin/env python3
#-*- encoding: utf8 -*-

__version__ = '0.1.0'
__auther__ = 'Arnold Mokira'

import os
import json
import argparse
from time import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchinfo import summary

IMAGE_CHANNELS = 3
IMAGE_SIZE = (96, 128)
NUM_CLASSES = 10

valid_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


class AlexNet(nn.Module):
    def __init__(
        self, in_channels=3, input_shape=(224, 224), num_classes=10,
        dropout=0.5, feature_map_dim=None
    ):
        super(AlexNet, self).__init__()
        self.in_channels = in_channels if in_channels else IMAGE_CHANNELS
        self.input_shape = input_shape if input_shape else IMAGE_SIZE
        self.num_classes = num_classes if num_classes else NUM_CLASSES

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        if not feature_map_dim:
            features_map_size = self._evaluate_features_map_shape()
            feature_map_dim = 1
            for dim in features_map_size[1:]:
                feature_map_dim *= dim

        self.fc1 = nn.Sequential(
            nn.Linear(feature_map_dim, 4096),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes))

        # Initialize weights
        self._initialize_weights()

    def _evaluate_features_map_shape(self):
        x = torch.zeros((1, IMAGE_CHANNELS, *IMAGE_SIZE))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.layer5(x)
        return y.shape

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

    def predict(self, images):
        images = valid_transform(images)
        logits = self.forward(images)
        probs = torch.softmax(data, dim=-1)
        indexes = torch.argmax(probs, dim=-1)  #: dim(indexes) = B

        sample_idx = torch.arange(probs.size(0))
        max_vals = probs[sample_idx, indexes]

        indexes = indexes.cpu().detach()
        max_vals = max_vals.cpu().detach()

        predictions = []
        for i in range(indexes.size(0)):
            predictions.append((indexes[i], max_vals[i]))
        return predictions


def model_load(file_path):
    """Model loading function

    :param file_path: The model weights file path.
    :type file_path: `str`
    :rtype: `AlexNet`
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such model file at: {file_path}.")
    instance = AlexNet(IMAGE_CHANNELS, IMAGE_SIZE, NUM_CLASSES)
    weights = torch.load(file_path)
    instance.load_state_dict(weights)
    return instance


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val  # val * n
        self.count += n
        self.avg = self.sum / self.count


def load_dataset_samples(data_dir):
    paths = []
    classes = []
    class_names = []
    folder_names = sorted(os.listdir(data_dir))
    loader = tqdm(folder_names, total=len(folder_names))
    for index, folder_name in enumerate(loader):
        class_name = folder_name
        class_idx = index

        class_names.append(class_name)
        folder_path = os.path.join(data_dir, folder_name)
        file_names = os.listdir(folder_path)
        file_count = len(file_names)
        loader.set_description(
             f"Loading of {file_count} from {folder_name}")
        for fid, file_name in enumerate(file_names):
            if not file_name.endswith(".png") \
               and not file_name.endswith(".jpg"):
                continue
            file_path = os.path.join(folder_path, file_name)
            paths.append(file_path)
            classes.append(class_idx)
            loader.set_postfix(files=f"{(fid + 1)}/{file_count}")
        loader.write(f"Class {class_name} of {file_count} is processed.")
    loader.set_description("Done")
    return paths, classes, class_names


class ImagesDataset(Dataset):

    def __init__(self, inputs, targets, transform):
        self.samples = [(x, y) for x, y in zip(inputs, targets)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        # if self.transform is not None:
        with open(path, mode='rb') as f:
            img = Image.open(f).convert('RGB')
            img = self.transform(img)
        return img, target


def get_data_loaders(args, train_set_percent=0.85):
    x_base, y_base, class_names = load_dataset_samples(args.data_dir)
    dataset = ImagesDataset(x_base, y_base, valid_transform)

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return data_loader class_names


def accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k]
            # correct_k = correct_k.reshape(-1)
            # correct_k = correct_k.float()
            # correct_k = correct_k.sun(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def inference(model, criterion, data_loader, device):
    """Run inference on test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_targets = []

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time()
        loader = tqdm(test_loader, desc="Inference")
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

            # Measure accuracy
            losses.update(loss.item(), inputs.size(0))
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            top1.update(acc1, inputs.size(0))

            # Store predictions and targets
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Measure elapsed time
            batch_time.update(time() - end)
            end = time()
            loader.set_postfix(time=batch_time.avg, losses=losses.avg)

    return losses.avg, top1.avg, all_preds, all_targets


def _run_model_inference(args):
    """Main training function for license plate digit classification"""
    set_seed(args.seed)

    # Set device
    device_name = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu >= 0
        else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Get dataloaders
    data_loader, class_names = get_data_loaders(args)
    num_class_names = len(class_names)

    model = model_load(args.model)
    model = model.to(device)

    input_data = torch.randn(args.batch_size, IMAGE_CHANNELS, *IMAGE_SIZE)
    summary(model, input_data=input_data)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Run inference on test set
    print("\n")
    print("Inference start...")
    test_loss, test_acc, predictions, targets = \
        inference(model, criterion, test_loader, device)
    print(f'Inference Loss: {test_loss:.8f},  Acc: {test_acc.item():.2f}%')

    # Plot confusion matrix and classification report
    if args.plot_results:
        print("\n")
        print("Results plotting...", end=' ')
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = [str(i) for i in range(num_class_names)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, 'confusion_matrix.png'))
        print("Done!")

        # Print classification report
        classes = np.unique(np.concatenate((targets, predictions)))
        classes = [class_names[i] for i in classes]
        report = classification_report(
            targets, predictions, target_names=classes, zero_division=0)
        print("Classification Report:")
        print(report)

        # Save classification report to file
        rfile = os.path.join(args.checkpoint_dir, 'report.txt')
        with open(rfile, 'w') as f:
            f.write(report)

    return model, test_acc


def main():
    parser = argparse.ArgumentParser(description='AlexNet inference')

    # Dataset parameters
    parser.add_argument('-d', '--data-dir', type=str, help='data directory.')

    # Model parameters
    parser.add_argument(
        '-m', '--model', type=str, required=True, help="Existing model file.")

    # Inference parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')

    # Misc parameters
    parser.add_argument('--seed', default=781227, type=int,
                        help='random seed (default: 781227)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use (default: 0, -1 for CPU)')
    parser.add_argument('--num-workers', default=2, type=int,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose output')
    parser.add_argument('--plot-results', action='store_true',
                        help='plot results')

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Train model
    model, test_acc = _run_model_inference(args)
    print(f'Final Inference Accuracy: {test_acc.item():.2f}%')


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\033[91mCanceled by user\033[0m")
        exit(125)
    except FileNotFoundError as e:
        print("[FileNotFoundError]", e.args[0])
        exit(2)
