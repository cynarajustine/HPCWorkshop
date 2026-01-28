#!/usr/bin/env python3
"""
Production-ready DDP training script for HPC environments
Usage:
    Single GPU: python train_ddp.py
    Multiple GPUs: torchrun --nproc_per_node=4 train_ddp.py
    HPC Cluster: sbatch submit_job.slurm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
import argparse
import os
import time
import json
from pathlib import Path


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    torch.cuda.set_device(gpu)
    if world_size > 1:
        dist.init_process_group('nccl')
    
    return rank, world_size, gpu


def cleanup():
    """Clean up distributed training"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_logger(rank):
    """Create logger that only prints on rank 0"""
    class Logger:
        def __init__(self, rank):
            self.rank = rank
        
        def info(self, msg):
            if self.rank == 0:
                print(f"[INFO] {msg}")
        
        def warning(self, msg):
            if self.rank == 0:
                print(f"[WARN] {msg}")
    
    return Logger(rank)


def create_model(num_classes=10):
    """Create ResNet-18 model"""
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    return model


def create_dataloaders(batch_size, num_workers, world_size, rank):
    """Create train and test dataloaders with DistributedSampler"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, 
                                transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, 
                               transform=transform_test)
    
    # DistributedSampler handles data sharding across GPUs
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    test_sampler = DistributedSampler(
        testset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )
    
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler


def train_epoch(epoch, train_loader, model, criterion, optimizer, scaler, 
                device, logger, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    # Important for DistributedSampler
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Epoch {epoch+1} Batch [{batch_idx+1}/{len(train_loader)}] "
                       f"Loss: {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, elapsed


@torch.no_grad()
def evaluate(test_loader, model, criterion, device, logger):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='DDP Training on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size per GPU (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}')
    logger = get_logger(rank)
    
    logger.info(f"Training on rank {rank}/{world_size}, GPU {gpu}")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.lr}, AMP: {args.use_amp}")
    
    # Create model
    model = create_model(num_classes=10).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], output_device=gpu)
    
    # Create dataloaders
    train_loader, test_loader, train_sampler = create_dataloaders(
        args.batch_size, args.num_workers, world_size, rank
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                         weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.use_amp else None
    
    # Training loop
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    logger.info("Starting training...")
    overall_start = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            epoch, train_loader, model, criterion, optimizer, scaler, 
            device, logger, use_amp=args.use_amp
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(test_loader, model, criterion, 
                                      device, logger)
        
        # Log metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        metrics['epoch_times'].append(epoch_time)
        
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                   f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
                   f"Time: {epoch_time:.2f}s")
        
        scheduler.step()
        
        # Save checkpoint
        if rank == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(args.checkpoint_dir) / f'epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if world_size > 1 
                                    else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
    
    overall_time = time.time() - overall_start
    
    # Save final metrics
    if rank == 0:
        metrics['total_time'] = overall_time
        metrics['samples_per_sec'] = len(train_loader.dataset) * args.epochs / overall_time
        
        metrics_path = Path(args.checkpoint_dir) / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Total time: {overall_time:.2f}s")
        logger.info(f"Samples/sec: {metrics['samples_per_sec']:.2f}")
        logger.info(f"Final Test Accuracy: {metrics['test_acc'][-1]:.2f}%")
    
    cleanup()


if __name__ == '__main__':
    main()
