"""
Stage 2 - DCGAN Demo Script
Demonstrates training a convolutional GAN on MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.stage2_dcgan.generator import DCGANGenerator
from src.stage2_dcgan.discriminator import DCGANDiscriminator
from src.training_engine.gan_loop import train_step_conv
from src.common.utils import init_weights

def stage2_demo():
    """Run Stage 2 DCGAN demo"""
    print("=" * 60)
    print("Stage 2 - DCGAN Demo")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    z_dim = 100
    channels = 1
    feature_maps = 64
    batch_size = 64
    lr = 2e-4
    epochs = 10  # Short demo
    img_size = 28
    
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
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
    print("Initializing DCGAN models...")
    gen = DCGANGenerator(z_dim, channels, feature_maps).to(device)
    disc = DCGANDiscriminator(channels, feature_maps).to(device)
    
    # Initialize weights
    init_weights(gen, init_type='normal', gain=0.02)
    init_weights(disc, init_type='normal', gain=0.02)
    
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}\n")
    
    # Training
    print("Starting training...")
    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        gen.train()
        disc.train()
        loss_d_avg = 0
        loss_g_avg = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real, _ in pbar:
            loss_D, loss_G = train_step_conv(
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
                fake = gen(z).cpu()
                
                # Handle size variation
                if fake.size(2) > 28:
                    fake = fake[:, :, :28, :28]
                
                # Visualize
                fig, axes = plt.subplots(4, 4, figsize=(6, 6))
                for i, ax in enumerate(axes.flat):
                    img = (fake[i, 0].clamp(-1, 1) + 1) / 2.0
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"outputs/images/stage2_dcgan_epoch_{epoch+1}.png", dpi=100)
                plt.close()
            gen.train()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
            }
            torch.save(checkpoint, f"outputs/checkpoints/dcgan_epoch_{epoch}.pt")
    
    print("\nTraining completed!")
    print(f"Sample images saved to outputs/images/")
    print(f"Checkpoints saved to outputs/checkpoints/")

if __name__ == "__main__":
    stage2_demo()
