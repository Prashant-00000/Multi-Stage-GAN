import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from a directory"""
    def __init__(self, img_dir, transform=None, img_size=28):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        
        # Collect all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for ext in valid_extensions:
            self.img_paths.extend(Path(img_dir).rglob(f'*{ext}'))
            self.img_paths.extend(Path(img_dir).rglob(f'*{ext.upper()}'))
        
        self.img_paths = list(set(self.img_paths))  # Remove duplicates
        
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img

def get_data_loader(dataset_name, batch_size, img_size=28, num_workers=4, root='./data'):
    """Get data loader for specified dataset"""
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
        
    elif dataset_name == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_path = os.path.join(root, 'celeba')
        os.makedirs(data_path, exist_ok=True)
        dataset = CustomImageDataset(data_path, transform=transform, img_size=img_size)
        
    elif dataset_name == 'monet':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_path = os.path.join(root, 'monet')
        os.makedirs(data_path, exist_ok=True)
        dataset = CustomImageDataset(data_path, transform=transform, img_size=img_size)
        
    elif dataset_name == 'photos':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_path = os.path.join(root, 'photos')
        os.makedirs(data_path, exist_ok=True)
        dataset = CustomImageDataset(data_path, transform=transform, img_size=img_size)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return data_loader

def get_mnist_loaders(batch_size, num_workers=4, root='./data'):
    """Get MNIST train and test loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader

class ConditionalMNIST(datasets.MNIST):
    """MNIST dataset that returns both image and label"""
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label

def get_conditional_mnist_loader(batch_size, num_workers=4, root='./data'):
    """Get conditional MNIST loader with labels"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = ConditionalMNIST(root=root, train=True, transform=transform, download=True)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return loader

class UnpairedImageDataset(Dataset):
    """Dataset for unpaired image-to-image translation (CycleGAN)"""
    def __init__(self, dir_a, dir_b, transform=None, max_dataset_size=None):
        self.transform = transform
        self.img_paths_a = sorted([str(p) for p in Path(dir_a).rglob('*.*')])
        self.img_paths_b = sorted([str(p) for p in Path(dir_b).rglob('*.*')])
        
        if max_dataset_size:
            self.img_paths_a = self.img_paths_a[:max_dataset_size]
            self.img_paths_b = self.img_paths_b[:max_dataset_size]
        
        self.len_a = len(self.img_paths_a)
        self.len_b = len(self.img_paths_b)
    
    def __getitem__(self, idx):
        idx_a = idx % self.len_a
        idx_b = np.random.randint(0, self.len_b)
        
        img_a = Image.open(self.img_paths_a[idx_a]).convert('RGB')
        img_b = Image.open(self.img_paths_b[idx_b]).convert('RGB')
        
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b
    
    def __len__(self):
        return max(self.len_a, self.len_b)
