"""
Stage 2b - Conditional GAN Demo Script
Demonstrates class-conditioned generation on MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.stage2b_cgan.generator import ConditionalGenerator
from src.stage2b_cgan.discriminator import ConditionalDiscriminator
from src.common.utils import init_weights

def stage2b_demo():
    """Run Stage 2b Conditional GAN demo"""
    print("=" * 60)
    print("Stage 2b - Conditional GAN Demo")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    z_dim = 100
    num_classes = 10
    embedding_dim = 100
    channels = 1
    feature_maps = 64
    batch_size = 64
    lr = 2e-4
    epochs = 10  # Short demo
    img_size = 28
    
    # Data loading
    print("Loading MNIST dataset with labels...")
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
    print("Initializing Conditional GAN models...")
    gen = ConditionalGenerator(z_dim, num_classes, embedding_dim, channels, feature_maps).to(device)
    disc = ConditionalDiscriminator(num_classes, embedding_dim, channels, feature_maps).to(device)
    
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
        
        for real, labels in pbar:
            real = real.to(device)
            labels = labels.to(device)
            batch_size_current = real.size(0)
            
            # Train Discriminator
            z = torch.randn(batch_size_current, z_dim).to(device)
            fake = gen(z, labels).detach()
            
            disc_real = disc(real, labels)
            disc_fake = disc(fake, labels)
            
            loss_d = criterion(disc_real, torch.full_like(disc_real, 0.9)) + \
                     criterion(disc_fake, torch.full_like(disc_fake, 0.1))
            
            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()
            
            # Train Generator
            z = torch.randn(batch_size_current, z_dim).to(device)
            fake = gen(z, labels)
            output = disc(fake, labels)
            loss_g = criterion(output, torch.ones_like(output))
            
            opt_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()
            
            loss_d_avg += loss_d.item()
            loss_g_avg += loss_g.item()
            num_batches += 1
            pbar.set_postfix({'D': f'{loss_d.item():.4f}', 'G': f'{loss_g.item():.4f}'})
        
        avg_d = loss_d_avg / num_batches
        avg_g = loss_g_avg / num_batches
        print(f"Epoch {epoch+1} - Loss D: {avg_d:.4f}, Loss G: {avg_g:.4f}")
        
        # Generate samples per class
        if (epoch + 1) % 5 == 0:
            gen.eval()
            with torch.no_grad():
                from torchvision.utils import save_image
                
                # Generate one sample per digit
                z = torch.randn(10, z_dim).to(device)
                labels_gen = torch.arange(10).to(device)
                fake_images = gen(z, labels_gen).cpu()
                
                # Handle size variation
                if fake_images.size(2) > 28:
                    fake_images = fake_images[:, :, :28, :28]
                
                save_image((fake_images + 1) / 2.0, f"outputs/images/stage2b_cgan_epoch_{epoch+1}.png", nrow=5)
            gen.train()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
            }
            torch.save(checkpoint, f"outputs/checkpoints/cgan_epoch_{epoch}.pt")
    
    print("\nTraining completed!")
    print(f"Sample images saved to outputs/images/")
    print(f"Checkpoints saved to outputs/checkpoints/")
    print("\nConditional GAN generates specific digit classes based on labels!")

if __name__ == "__main__":
    stage2b_demo()
