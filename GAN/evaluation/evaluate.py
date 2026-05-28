"""
Comprehensive Evaluation Script
Evaluates generated images using FID and other metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.stage1_gan.generator import Generator
from src.stage2_dcgan.generator import DCGANGenerator
from src.stage2b_cgan.generator import ConditionalGenerator
from evaluation.fid import FIDCalculator
from src.common.utils import get_device

class GANEvaluator:
    """Evaluator for GAN models"""
    def __init__(self, device=None):
        self.device = device or get_device()
        self.fid_calculator = FIDCalculator(self.device)
    
    def evaluate_stage1(self, checkpoint_path=None, num_samples=2048):
        """Evaluate Stage 1 GAN"""
        print("\n" + "="*60)
        print("Evaluating Stage 1 - Basic GAN")
        print("="*60)
        
        z_dim = 100
        img_dim = 784
        
        # Load generator
        gen = Generator(z_dim, img_dim).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            gen.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        gen.eval()
        
        # Calculate real statistics if not exists
        real_stats_path = "outputs/real_mnist_stats.npz"
        if not os.path.exists(real_stats_path):
            print("Computing real image statistics...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            loader = DataLoader(
                datasets.MNIST(root="data/", train=True, transform=transform, download=True),
                batch_size=64,
                shuffle=False
            )
            self.fid_calculator.save_statistics(loader, real_stats_path)
        
        # Generate fake images
        print(f"Generating {num_samples} fake images...")
        fake_images = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                z = torch.randn(current_batch, z_dim).to(self.device)
                fake = gen(z).reshape(current_batch, 1, 28, 28)
                fake_images.append(fake.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)
        
        # Calculate FID
        print("Calculating FID score...")
        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, images):
                self.images = images
            def __len__(self):
                return len(self.images)
            def __getitem__(self, idx):
                return self.images[idx]
        
        fake_loader = DataLoader(FakeDataset(fake_images), batch_size=64)
        fake_features = self.fid_calculator.get_features(fake_loader)
        fid_score = self.fid_calculator.calculate_fid_from_stats(fake_features, real_stats_path)
        
        print(f"\nStage 1 Results:")
        print(f"  FID Score: {fid_score:.2f}")
        print(f"  (Lower is better - typical range: 5-50)")
        
        return {'fid': fid_score}
    
    def evaluate_stage2(self, checkpoint_path=None, num_samples=2048):
        """Evaluate Stage 2 DCGAN"""
        print("\n" + "="*60)
        print("Evaluating Stage 2 - DCGAN")
        print("="*60)
        
        z_dim = 100
        channels = 1
        feature_maps = 64
        
        # Load generator
        gen = DCGANGenerator(z_dim, channels, feature_maps).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            gen.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        gen.eval()
        
        # Calculate real statistics if not exists
        real_stats_path = "outputs/real_mnist_stats.npz"
        if not os.path.exists(real_stats_path):
            print("Computing real image statistics...")
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            loader = DataLoader(
                datasets.MNIST(root="data/", train=True, transform=transform, download=True),
                batch_size=64,
                shuffle=False
            )
            self.fid_calculator.save_statistics(loader, real_stats_path)
        
        # Generate fake images
        print(f"Generating {num_samples} fake images...")
        fake_images = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                z = torch.randn(current_batch, z_dim).to(self.device)
                fake = gen(z)
                # Crop if necessary
                if fake.size(2) > 28:
                    fake = fake[:, :, :28, :28]
                fake_images.append(fake.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)
        
        # Calculate FID
        print("Calculating FID score...")
        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, images):
                self.images = images
            def __len__(self):
                return len(self.images)
            def __getitem__(self, idx):
                return self.images[idx]
        
        fake_loader = DataLoader(FakeDataset(fake_images), batch_size=64)
        fake_features = self.fid_calculator.get_features(fake_loader)
        fid_score = self.fid_calculator.calculate_fid_from_stats(fake_features, real_stats_path)
        
        print(f"\nStage 2 Results:")
        print(f"  FID Score: {fid_score:.2f}")
        print(f"  (Lower is better - DCGAN should show improvement over Stage 1)")
        
        return {'fid': fid_score}
    
    def evaluate_stage2b(self, checkpoint_path=None, num_samples=2048):
        """Evaluate Stage 2b Conditional GAN"""
        print("\n" + "="*60)
        print("Evaluating Stage 2b - Conditional GAN")
        print("="*60)
        
        z_dim = 100
        num_classes = 10
        embedding_dim = 100
        channels = 1
        feature_maps = 64
        
        # Load generator
        gen = ConditionalGenerator(z_dim, num_classes, embedding_dim, channels, feature_maps).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            gen.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        gen.eval()
        
        # Calculate real statistics if not exists
        real_stats_path = "outputs/real_mnist_stats.npz"
        if not os.path.exists(real_stats_path):
            print("Computing real image statistics...")
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            loader = DataLoader(
                datasets.MNIST(root="data/", train=True, transform=transform, download=True),
                batch_size=64,
                shuffle=False
            )
            self.fid_calculator.save_statistics(loader, real_stats_path)
        
        # Generate fake images (balanced across classes)
        print(f"Generating {num_samples} fake images (balanced across classes)...")
        fake_images = []
        batch_size = 64
        samples_per_class = num_samples // num_classes
        
        with torch.no_grad():
            for class_idx in range(num_classes):
                for i in range(0, samples_per_class, batch_size):
                    current_batch = min(batch_size, samples_per_class - i)
                    z = torch.randn(current_batch, z_dim).to(self.device)
                    labels = torch.full((current_batch,), class_idx, dtype=torch.long).to(self.device)
                    fake = gen(z, labels)
                    # Crop if necessary
                    if fake.size(2) > 28:
                        fake = fake[:, :, :28, :28]
                    fake_images.append(fake.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)
        
        # Calculate FID
        print("Calculating FID score...")
        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, images):
                self.images = images
            def __len__(self):
                return len(self.images)
            def __getitem__(self, idx):
                return self.images[idx]
        
        fake_loader = DataLoader(FakeDataset(fake_images), batch_size=64)
        fake_features = self.fid_calculator.get_features(fake_loader)
        fid_score = self.fid_calculator.calculate_fid_from_stats(fake_features, real_stats_path)
        
        print(f"\nStage 2b Results:")
        print(f"  FID Score: {fid_score:.2f}")
        print(f"  (Conditional generation maintains class-specific quality)")
        
        return {'fid': fid_score}

def main():
    """Run evaluations on all stages"""
    device = get_device()
    evaluator = GANEvaluator(device)
    
    results = {}
    
    # Evaluate Stage 1
    try:
        results['stage1'] = evaluator.evaluate_stage1()
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
    
    # Evaluate Stage 2
    try:
        results['stage2'] = evaluator.evaluate_stage2()
    except Exception as e:
        print(f"Stage 2 evaluation failed: {e}")
    
    # Evaluate Stage 2b
    try:
        results['stage2b'] = evaluator.evaluate_stage2b()
    except Exception as e:
        print(f"Stage 2b evaluation failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for stage, metrics in results.items():
        print(f"\n{stage}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    main()
