import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(781227)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        
        # First convolutional layer
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            # Layer 6
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            # Layer 7
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # Layer 8 (Output layer)
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
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

def get_dataloaders(args):
    """
    Create train, validation, and test dataloaders
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths
    data_dir = args.data_dir
    
    # Load datasets
    if args.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'test'),
            transform=val_transform
        )
    else:
        # For CIFAR-10 (demonstration purposes)
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        # Split train into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
    
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
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler, epoch, device, args
):
    """Train the model for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch training {epoch + 1}")
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
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            curr_lr = scheduler.get_last_lr()[0]
            tepoch.set_postfix(
                loss=f"{losses.avg:.8f}", acc=top1.avg.item(),
                lr=curr_lr)
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    return losses.avg, top1.avg, top5.avg

def inference(model, test_loader, device):
    """Run inference on test set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Switch to evaluation mode
    model.eval()
    
    preds = []
    targets_list = []
    
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            # Store predictions
            _, pred = outputs.topk(1, 1, True, True)
            preds.extend(pred.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    return top1.avg, top5.avg, preds, targets_list

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
        torch.save(state, best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model and optimizer from checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return None

def train_alexnet(args):
    """Main training function"""
    # set_seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if args.dataset == 'imagenet':
        num_classes = 1000
    else:  # CIFAR-10
        num_classes = 10
    
    model = AlexNet(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # AlexNet paper uses SGD with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=args.lr_patience,
        # verbose=True
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
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, device, args
        )
        print("OK")
        # Evaluate on validation set
        val_loss, val_acc, val_acc5 = validate(model, val_loader, criterion, device)
        
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
        print(f'Epoch: {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Acc@5: {val_acc5:.2f}%')
        print(f'Best Val Acc: {best_acc:.2f}%')
        print('-' * 60)

    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
    load_checkpoint(best_model_path, model)
    
    # Run inference on test set
    test_acc, test_acc5, predictions, targets = inference(model, test_loader, device)
    
    print(f'Test Acc: {test_acc:.2f}%, Test Acc@5: {test_acc5:.2f}%')
    
    # Plot confusion matrix if it's a small dataset like CIFAR-10
    if args.dataset != 'imagenet' and args.plot_results:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(args.checkpoint_dir, 'confusion_matrix.png'))

    return model, test_acc

def main():
    parser = argparse.ArgumentParser(description='AlexNet Training')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='cifar10', choices=['imagenet', 'cifar10'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='data directory (default: ./data)')
    
    # Model parameters
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout probability (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--epochs', default=90, type=int,
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-patience', default=5, type=int,
                        help='patience for learning rate reduction (default: 5)')
    
    # Checkpoint parameters
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str,
                        help='checkpoint directory (default: ./checkpoints)')
    
    # Misc parameters
    parser.add_argument('--seed', default=781227, type=int,
                        help='random seed (default: 781227)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use (default: 0, -1 for CPU)')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose output')
    parser.add_argument('--plot-results', action='store_true',
                        help='plot results (only for CIFAR-10)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Train model
    model, test_acc = train_alexnet(args)
    print(f'Final Test Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()
