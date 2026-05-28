"""
Stage 1 - Basic GAN Demo Script
Demonstrates training a basic GAN on MNIST and evaluating with FID
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Import from project
from src.stage1_gan.generator import Generator
from src.stage1_gan.discriminator import Discriminator
from src.training_engine.gan_loop import train_step
from evaluation.fid import FIDCalculator

def stage1_demo():
    """Run Stage 1 GAN demo"""
    print("=" * 60)
    print("Stage 1 - Basic GAN Demo")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    z_dim = 100
    img_dim = 784
    batch_size = 64
    lr = 2e-4
    epochs = 10  # Short demo
    
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    loader = DataLoader(
        datasets.MNIST(root="data/", train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True
    )
    print(f"Dataset loaded: {len(loader)} batches\n")
    
    # Initialize models
    print("Initializing models...")
    gen = Generator(z_dim, img_dim).to(device)
    disc = Discriminator(img_dim).to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}\n")
    
    # Training
    print("Starting training...")
    os.makedirs("outputs/images", exist_ok=True)
    
    for epoch in range(epochs):
        gen.train()
        disc.train()
        loss_d_avg = 0
        loss_g_avg = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real, _ in pbar:
            loss_D, loss_G = train_step(
                real, gen, disc, opt_gen, opt_disc, criterion, device, z_dim
            )
            loss_d_avg += loss_D
            loss_g_avg += loss_G
            num_batches += 1
            pbar.set_postfix({'D': f'{loss_D:.4f}', 'G': f'{loss_G:.4f}'})
        
        avg_d = loss_d_avg / num_batches
        avg_g = loss_g_avg / num_batches
        print(f"Epoch {epoch+1} - Loss D: {avg_d:.4f}, Loss G: {avg_g:.4f}")
        
        # Generate samples
        if (epoch + 1) % 5 == 0:
            gen.eval()
            with torch.no_grad():
                z = torch.randn(16, z_dim).to(device)
                fake = gen(z).cpu().reshape(16, 1, 28, 28)
                
                # Visualize
                fig, axes = plt.subplots(4, 4, figsize=(6, 6))
                for i, ax in enumerate(axes.flat):
                    img = (fake[i, 0] + 1) / 2.0
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"outputs/images/stage1_epoch_{epoch+1}.png", dpi=100)
                plt.close()
            gen.train()
    
    print("\nTraining completed!")
    print(f"Sample images saved to outputs/images/")
    
    # FID Calculation (skip on CPU - too slow)
    if str(device) == 'cpu':
        print("\n⚠ Skipping FID calculation (too slow on CPU - use GPU for full evaluation)")
        print("✓ Training complete! Check outputs/images/ for generated samples")
    else:
        try:
            print("\nCalculating FID Score...")
            fid_calculator = FIDCalculator(device)
            
            # Load real stats (compute if not exists)
            real_stats_path = "outputs/real_mnist_stats.npz"
            if not os.path.exists(real_stats_path):
                print("Computing real image statistics...")
                full_loader = DataLoader(
                    datasets.MNIST(root="data/", train=True, transform=transform, download=True),
                    batch_size=batch_size,
                    shuffle=False
                )
                fid_calculator.save_statistics(full_loader, real_stats_path)
            
            # Generate fake samples and compute FID
            print("Generating fake images for FID calculation...")
            gen.eval()
            fake_images = []
            num_samples = 2048
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    current_batch = min(batch_size, num_samples - i)
                    z = torch.randn(current_batch, z_dim).to(device)
                    fake = gen(z).reshape(current_batch, 1, 28, 28)
                    fake_images.append(fake.cpu())
            
            fake_images = torch.cat(fake_images, dim=0)
            
            class FakeDataset(torch.utils.data.Dataset):
                def __init__(self, images):
                    self.images = images
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    return self.images[idx]
            
            fake_loader = DataLoader(FakeDataset(fake_images), batch_size=batch_size)
            fake_features = fid_calculator.get_features(fake_loader)
            fid_score = fid_calculator.calculate_fid_from_stats(fake_features, real_stats_path)
            
            print(f"\nFID Score: {fid_score:.2f}")
            print("(Lower is better - indicates better quality generated images)")
        except Exception as e:
            print(f"⚠ FID calculation failed: {e}")
            print("✓ Training complete! Check outputs/images/ for generated samples")

if __name__ == "__main__":
    stage1_demo()
