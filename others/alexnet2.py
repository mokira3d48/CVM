import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlexNetLicensePlate(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(AlexNetLicensePlate, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                3, 96, kernel_size=11, stride=4, padding=0),
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

        features_map_size = self._evaluate_features_map_shape()
        classifier_input_size = 1
        for dim in features_map_size[1:]:
            classifier_input_size *= dim

        self.fc1 = nn.Sequential(
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LicensePlateDigitDataset(Dataset):
    """Dataset for license plate digit classification"""

    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        self._make_dataset()

    def _make_dataset(self):
        folder_names = sorted(os.listdir(self.root_dir))
        loader = tqdm(folder_names, total=len(folder_names))
        for index, folder_name in enumerate(loader):
            class_name = folder_name
            class_idx = index

            self.class_to_idx[class_name] = class_idx
            folder_path = os.path.join(self.root_dir, folder_name)
            file_names = os.listdir(folder_path)
            file_count = len(file_names)
            loader.set_description(
                 f"Loading of {file_count} from {folder_name}")
            for index, file_name in enumerate(file_names):
                if not file_name.endswith(".png") \
                   and not file_name.endswith(".jpg"):
                    continue
                file_path = os.path.join(folder_path, file_name)
                self.samples.append((file_path, class_idx))
                loader.set_postfix(files=f"{(index + 1)}/{file_count}")
            loader.write(f"Class {class_name} of {file_count} is processed.")
        loader.set_description("Done")
        # self.samples = self.samples[:1000]
             
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        # Load image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_dataloaders(args):
    """
    Create train, validation, and test dataloaders for license plate digits
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((96, 128)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    base_dataset = LicensePlateDigitDataset(
        root_dir=args.data_dir,
        transform=train_transform
    )
    
    train_size = int(0.9 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        base_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_one_epoch(
    model, train_loader, criterion, optimizer, epoch, device, args
):
    """Train the model for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f" Training epoch {epoch + 1}")
        for i, (inputs, targets) in enumerate(tepoch):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            tepoch.set_postfix(
                loss=f"{losses.avg:.8f}",
                acc=top1.avg.item(),
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
        end = time.time()
        loader = tqdm(val_loader, desc=" Validation")
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            loader.set_postfix(
                loss=f"{losses.avg:.8f}", acc=top1.avg.item())
    
    return losses.avg, top1.avg

def inference(model, test_loader, device):
    """Run inference on test set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()
    
    # Switch to evaluation mode
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        end = time.time()
        loader = tqdm(test_loader, desc="Inference")
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Measure accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            top1.update(acc1, inputs.size(0))
            
            # Store predictions and targets
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    return top1.avg, all_preds, all_targets

def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint to disk"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        # torch.save(state, best_filename)
        torch.save(state['state_dict'], best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model and optimizer from checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return None

def train_license_plate_model(args):
    """Main training function for license plate digit classification"""
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create model - for digits we use 35 classes (0-9) and (A-Z)
    model = AlexNetLicensePlate(num_classes=35, dropout=args.dropout)
    if args.weights and os.path.isfile(args.weights):
        # weights = torch.load(args.weights)
        # model.load_state_dict(weights)
        load_checkpoint(args.weights, model)
        model.fc3 = nn.Sequential(nn.Linear(4096, 37))

    model = model.to(device)
    summary(model, input_shape=(args.batch_size, 3, 96, 128))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=args.lr_patience,
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
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
        print(
            "\n"
            f"Start training epoch: {epoch + 1}/{args.epochs}"
            f" with lr: {scheduler.get_last_lr()[0]}")
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device, args
        )

        # Evaluate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
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
        print(f'Train Loss: {train_loss:.8f}, Train Acc: {train_acc.item():.2f}%')
        print(f'Val Loss: {val_loss:.8f}, Val Acc: {val_acc.item():.2f}%')
        print(f'Best Val Acc: {best_acc.item():.2f}%')
        print('-' * 80)
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
    # load_checkpoint(best_model_path, model)
    weights = torch.load(best_model_path)
    model.load_state_dict(weights)
    
    # Run inference on test set
    test_acc, predictions, targets = inference(model, test_loader, device)
    print(f'Test Acc: {test_acc.item():.2f}%')
    
    # Plot confusion matrix and classification report
    if args.plot_results:
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = [str(i) for i in range(37)]
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

        # Print classification report
        report = classification_report(targets, predictions, target_names=classes)
        print("Classification Report:")
        print(report)
        
        # Save classification report to file
        with open(os.path.join(args.checkpoint_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    return model, test_acc

def main():
    parser = argparse.ArgumentParser(description='License Plate Digit Classification with AlexNet')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, required=True,
                        help='data directory (default: ./license_plate_digits)')
    
    # Model parameters
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--weights', type=str, help="Existing model file.")

    # Training parameters
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run (default: 50)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-patience', default=5, type=int,
                        help='patience for learning rate reduction (default: 5)')
    
    # Checkpoint parameters
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-dir', default='./license_plate_checkpoints', type=str,
                        help='checkpoint directory (default: ./license_plate_checkpoints)')
    
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
    model, test_acc = train_license_plate_model(args)
    print(f'Final Test Accuracy: {test_acc.item():.2f}%')

if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\033[91mCanceled by user\033[0m")
        exit(125)
