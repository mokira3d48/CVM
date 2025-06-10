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

train_transform = transforms.Compose([
    transforms.Resize((96, 128)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((96, 128)),
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
        x = torch.zeros((1, 3, 96, 128))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.layer5(x)
        return y.shape

    def _initialize_weights(self):
        """Function of model weights initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

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

    def replace_classifier(self, num_classes):
        """
        Replace the final classification layer with a new one
        that has the specified number of output classes.

        :param num_classes: Number of classes for the new classification
            layer.
        :type num_classes: `int`
        :rtype: `AlexNet`
        """
        # Create a new fully connected layer for classification
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

        # Initialize the weights of the new layer
        for m in self.fc3.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

        return self

    def optimizer(self, lrs, weight_decay=1e-4):
        # Configure optimizer with different learning rates
        # for different parts of the network.
        feature_params = []
        fc1_fc2_params = []
        fc3_params = []

        for name, param in self.named_parameters():
            if 'fc3' in name:
                fc3_params.append(param)
            elif 'fc1' in name or 'fc2' in name:
                fc1_fc2_params.append(param)
            else:
                feature_params.append(param)

        optimizer = optim.Adam([
            {'params': feature_params, 'lr': lrs[0]},
            #: Very low learning rate for frozen feature layers

            {'params': fc1_fc2_params, 'lr': lrs[1]},
            #: Medium learning rate for FC1 and FC2

            {'params': fc3_params, 'lr': lrs[2]}
            #: High learning rate for the new classification layer
        ], weight_decay=weight_decay)
        return optimizer


def fine_tune_model(
    pretrained_model_path, num_new_classes, lr, freeze_feature_layers=True
):
    """
    Load a pre-trained model and modify it for fine-tuning
    on a new dataset with different number of classes.

    :param pretrained_model_path: Path to the pre-trained model file path.
    :param num_new_classes: Number of classes in the new dataset.
    :param lr: The base learning rate will be used to config optimizer
        with the deference layers learning rate.
    :param freeze_feature_layers: Whether to freeze the feature
        extraction layers.

    :type pretrained_model_path: `str`
    :type num_new_classes: `int`
    :type lr: `float`
    :type freeze_feature_layers: `bool`

    :returns:
        model: Modified model ready for fine-tuning
        optimizer: Configured optimizer for fine-tuning
    :rtype: `tuple`
    """
    # Load the checkpoint
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_state_dict = torch.load(pretrained_model_path, map_location='cpu')

    # Get the original number of classes
    # Assuming the last layer in state_dict has
    # shape [original_num_classes, 4096]
    original_num_classes = None

    # Find the final classification layer
    for key in weights_state_dict.keys():
        if 'fc3.0.weight' in key:
            original_num_classes = weights_state_dict[key].shape[0]

    if original_num_classes is None:
        raise ValueError(
            "Could not determine the number of classes"
            " in the pre-trained model")

    print(f"Original model had {original_num_classes} output classes")
    print(f"New model will have {num_new_classes} output classes")

    # Create a model with the original number of classes first
    model = AlexNet(num_classes=original_num_classes)

    # Load all the weights from the checkpoint
    model.load_state_dict(weights_state_dict)

    # Now replace the classifier with a new one
    # for the target number of classes
    model.replace_classifier(num_new_classes)

    # Freeze feature extraction layers if specified
    if freeze_feature_layers:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in
                   ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']):
                param.requires_grad = False

    optimizer = model.optimizer((lr * 0.01, lr * 0.1, lr))
    return model, optimizer


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

    def update(self, val):
        self.val = val
        self.sum += val  # val * n
        self.count += 1
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


def dataset_cache_save(train, valid, test, class_names, ckpt_dir):
    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test

    train_file_cache = os.path.join(ckpt_dir, 'train.cach')
    valid_file_cache = os.path.join(ckpt_dir, 'valid.cach')
    test_file_cache = os.path.join(ckpt_dir, 'test.cach')

    with open(train_file_cache, mode='w', encoding='utf-8') as file1:
        train_dict = {
            'features': x_train,
            'labels': y_train,
            'classes': class_names}
        train_json = json.dumps(train_dict, indent=4)
        file1.write(train_json)

    with open(valid_file_cache, mode='w', encoding='utf-8') as file2:
        valid_dict = {
            'features': x_valid, 'labels': y_valid, 'classes': class_names}
        valid_json = json.dumps(valid_dict, indent=4)
        file2.write(valid_json)

    with open(test_file_cache, mode='w', encoding='utf-8') as file3:
        test_dict = {
            'features': x_test, 'labels': y_test, 'classes': class_names}
        test_json = json.dumps(test_dict, indent=4)
        file3.write(test_json)


def dataset_cache_load(ckpt_dir):
    train_file_cache = os.path.join(ckpt_dir, 'train.cach')
    valid_file_cache = os.path.join(ckpt_dir, 'valid.cach')
    test_file_cache = os.path.join(ckpt_dir, 'test.cach')

    for file in [train_file_cache, valid_file_cache, test_file_cache]:
        if not os.path.isfile(file):
            print(f"No dataset cache found at {ckpt_dir}")
            return None

    train = tuple()
    valid = tuple()
    test = tuple()
    class_names = []

    with open(train_file_cache, mode='r', encoding='utf-8') as file1:
        train_dict = json.load(file1)
        train = (train_dict['features'], train_dict['labels'])
        class_names = train_dict['classes']

    with open(valid_file_cache, mode='r', encoding='utf-8') as file2:
        valid_dict = json.load(file2)
        valid = (valid_dict['features'], valid_dict['labels'])

    with open(test_file_cache, mode='r', encoding='utf-8') as file3:
        test_dict = json.load(file3)
        test = (test_dict['features'], test_dict['labels'])

    return (train, valid, test), class_names


def get_data_loaders(args, train_set_percent=0.85):
    returned = dataset_cache_load(args.checkpoint_dir)
    if not returned:
        x_base, y_base, class_names = load_dataset_samples(args.data_dir)
        # base_dataset = ImagesDataset(base_samples)

        test_probs = 1.0 - train_set_percent
        data = train_test_split(x_base, y_base, test_size=test_probs)
        x_train = data[0]
        x_valid = data[1]
        y_train = data[2]
        y_valid = data[3]

        # test_size = int(0.5 * val_size)
        # val_size = val_size - test_size
        data = train_test_split(x_valid, y_valid, test_size=0.5)
        x_valid = data[0]
        x_test = data[1]
        y_valid = data[2]
        y_test = data[3]

        dataset_cache_save(
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test),
            class_names, args.checkpoint_dir)
    else:
        samples, class_names = returned
        train, valid, test = samples
        x_train, y_train = train
        x_valid, y_valid = valid
        x_test, y_test = test

    train_dataset = ImagesDataset(x_train, y_train, train_transform)
    valid_dataset = ImagesDataset(x_valid, y_valid, valid_transform)
    test_dataset = ImagesDataset(x_test, y_test, valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    return (train_loader, valid_loader, test_loader), class_names


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


def train_one_epoch(
    model, train_loader, criterion, optimizer, epoch, device
):
    """Train the model for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f" Training epoch {epoch + 1}")
        for i, (inputs, targets) in enumerate(tepoch):
            start = time()
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))
            acc1 = acc1[0]
            losses.update(loss.item())
            top1.update(acc1.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time() - start)

            tepoch.set_postfix(
                loss=f"{losses.avg:.8f}",
                acc=top1.avg,
                lr=optimizer.param_groups[0]['lr'])

    return losses.avg, top1.avg


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        loader = tqdm(val_loader, desc=" Validation")
        for i, (inputs, targets) in enumerate(loader):
            start = time()
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item())
            top1.update(acc1.item())

            # Measure elapsed time
            batch_time.update(time() - start)

            loader.set_postfix(
                loss=f"{losses.avg:.8f}", acc=top1.avg)

    return losses.avg, top1.avg


def inference(model, criterion, test_loader, device):
    """Run inference on test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_preds = []
    all_targets = []

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        loader = tqdm(test_loader, desc="Inference")
        for i, (inputs, targets) in enumerate(loader):
            start = time()
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

            # Measure accuracy
            losses.update(loss.item())
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            top1.update(acc1.item())

            # Store predictions and targets
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Measure elapsed time
            batch_time.update(time() - start)
            loader.set_postfix(time=batch_time.avg, losses=losses.avg)

    return losses.avg, top1.avg, batch_time.avg, all_preds, all_targets


def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint to disk"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # filename = os.path.join(
    #     checkpoint_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    filename = os.path.join(checkpoint_dir, f'checkpoint.pth')
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best.pt')
        torch.save(state['state_dict'], best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model and optimizer from checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(
            f"=> Loaded checkpoint '{checkpoint_path}'"
            f" (epoch {checkpoint['epoch'] + 1})")
        return checkpoint
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return None


def model_train(args):
    """Main training function for image classification"""
    set_seed(args.seed)

    # Set device
    device_name = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu >= 0
        else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Get dataloaders
    loaders, class_names = get_data_loaders(args)
    num_class_names = len(class_names)
    train_loader = loaders[0]
    valid_loader = loaders[1]
    test_loader = loaders[2]

    # Create model - for digits we use 35 classes (0-9) and (A-Z)
    model = AlexNet(num_classes=num_class_names, dropout=args.dropout)
    # optimizer = optim.Adam(
    #    model.parameters(),
    #    lr=args.learning_rate,
    #    weight_decay=args.weight_decay
    # )
    optimizer = model.optimizer(
        (args.learning_rate, args.learning_rate, args.learning_rate),
        weight_decay=args.weight_decay)
    if args.model and os.path.isfile(args.model):
        # weights = torch.load(args.weights)
        # model.load_state_dict(weights)
        # load_checkpoint(args.weights, model)
        # model.fc3 = nn.Sequential(nn.Linear(4096, 37))
        model, optimizer = fine_tune_model(
            args.model, num_class_names, args.learning_rate,
            args.feeze_feature_layers)

    model = model.to(device)
    input_data = torch.randn(args.batch_size, IMAGE_CHANNELS, *IMAGE_SIZE)
    summary(model, input_data=input_data)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=args.lr_patience,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        try:
            print(
                "\n"
                f"Start training epoch: {epoch + 1}/{args.epochs}"
                f" with lr: {scheduler.get_last_lr()[0]}")
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch, device
            )

            # Evaluate on validation set
            val_loss, val_acc = validate(
                model, valid_loader, criterion, device)

            # Adjust learning rate
            scheduler.step(val_acc)

            # Remember best accuracy and save checkpoint
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.checkpoint_dir)

            # Print epoch statistics
            # print(f'Epoch: {epoch+1}/{args.epochs}')
            print("")
            print(
                f'Train Loss: {train_loss:.8f},'
                f'Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.8f}, Val Acc: {val_acc:.2f}%')
            print(f'* Best Val Acc: \033[92m{best_acc:.2f}%\033[0m')
            print('-' * 80)
        except KeyboardInterrupt:
            print("Training process is canceled by user.")
            break

    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, 'best.pt')
    # load_checkpoint(best_model_path, model)
    weights = torch.load(best_model_path)
    model.load_state_dict(weights)

    # Run inference on test set
    print("\n")
    print("Inference start...")
    test_loss, test_acc, duration, predictions, targets = \
        inference(model, criterion, test_loader, device)
    print(
        f'Inference Loss: {test_loss:.8f},  Acc: {test_acc:.2f}%,'
        f' Time: {duration:.4f}s')

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

    return model, test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(
        description='License Plate Digit Classification with AlexNet')

    # Dataset parameters
    parser.add_argument('-d', '--data-dir', type=str,
                        help='data directory (default: ./license_plate_digits)')

    # Model parameters
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('-m', '--model', type=str, help="Existing model file.")
    parser.add_argument(
        '-ffl', '--feeze-feature-layers', action="store_true",
        help="Existing model file.")

    # Training parameters
    parser.add_argument('-n', '--epochs', default=50, type=int,
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-patience', default=5, type=int,
                        help='patience for learning rate reduction (default: 5)')

    # Checkpoint parameters
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument(
        '-ckpt', '--checkpoint-dir', default='./checkpoints', type=str,
        help='checkpoint directory (default: ./checkpoints)')

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
    model, test_loss, test_acc = model_train(args)
    print(f'Final Test Loss: {test_loss:.8f}, Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\033[91mCanceled by user\033[0m")
        exit(125)
