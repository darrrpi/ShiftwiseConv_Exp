import torch
from torchvision import datasets, transforms
import os

def get_dataloaders(dataset='cifar10', batch_size=128, num_workers=4, 
                    use_augmentation=True):
    """
    Setup CIFAR-10 or CIFAR-100 dataloaders
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: batch size for training
        num_workers: number of data loading workers
        use_augmentation: whether to use data augmentation
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Training transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=val_transform
        )
    else:
        train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=val_transform
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, num_classes


def get_dataloaders_tiny_imagenet(data_root='./data/tiny-imagenet-200', 
                                   batch_size=128, num_workers=4):
    """
    Setup Tiny ImageNet dataloaders (if needed for additional experiments)
    
    Args:
        data_root: path to Tiny ImageNet dataset
        batch_size: batch size
        num_workers: number of workers
    
    Returns:
        train_loader, val_loader, num_classes
    """
    from torchvision.datasets import ImageFolder
    
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    num_classes = 200
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = ImageFolder(
        os.path.join(data_root, 'train'),
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        os.path.join(data_root, 'val'),
        transform=val_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, num_classes