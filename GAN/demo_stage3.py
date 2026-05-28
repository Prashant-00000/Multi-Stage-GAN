#!/usr/bin/env python3
"""
Stage 3 - CycleGAN Demo: Unpaired Image-to-Image Translation
Demonstrates style transfer and domain adaptation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import CycleGAN components
from src.stage3_cyclegan.generator_resnet import ResNetGenerator
from src.stage3_cyclegan.discriminator_patchgan import PatchGANDiscriminator
from src.stage3_cyclegan.replay_buffer import ReplayBufferFixed
from src.common.losses import CycleLoss, IdentityLoss
from src.common.utils import init_weights

print("=" * 60)
print("Stage 3 - CycleGAN Demo")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ==================== Configuration ====================
img_size = 256
channels = 3
feature_maps = 64
num_residual_blocks = 9
batch_size = 1  # CycleGAN uses batch_size=1 typically
lr = 0.0002
epochs = 10  # Demo: shorter training
checkpoint_interval = 2

lambda_cycle = 10.0
lambda_identity = 0.5

# Create output directories
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)

# ==================== Create Simple Dataset ====================
class SimpleImageDataset(Dataset):
    """
    Simple dataset for demo - generates random images in two domains
    Domain A: Random images (simulating photos)
    Domain B: Random images (simulating paintings/sketches)
    """
    def __init__(self, num_samples=100, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image in domain A
        img_a = torch.randn(3, self.img_size, self.img_size)
        # Generate random image in domain B
        img_b = torch.randn(3, self.img_size, self.img_size)
        
        # Normalize to [-1, 1]
        img_a = (img_a - img_a.min()) / (img_a.max() - img_a.min()) * 2 - 1
        img_b = (img_b - img_b.min()) / (img_b.max() - img_b.min()) * 2 - 1
        
        return img_a, img_b

# Create data loader
print(f"Creating dataset... (synthetic for demo)")
dataset = SimpleImageDataset(num_samples=100, img_size=img_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Dataset created: {len(dataset)} image pairs\n")

# ==================== Initialize Models ====================
print("Initializing CycleGAN models...")

# Generators (A->B and B->A)
gen_AB = ResNetGenerator(channels, channels, num_residual_blocks, feature_maps).to(device)
gen_BA = ResNetGenerator(channels, channels, num_residual_blocks, feature_maps).to(device)

# Discriminators (for A and B domains)
disc_A = PatchGANDiscriminator(channels, feature_maps).to(device)
disc_B = PatchGANDiscriminator(channels, feature_maps).to(device)

# Initialize weights
init_weights(gen_AB, init_type='normal', gain=0.02)
init_weights(gen_BA, init_type='normal', gain=0.02)
init_weights(disc_A, init_type='normal', gain=0.02)
init_weights(disc_B, init_type='normal', gain=0.02)

# Count parameters
gen_params = sum(p.numel() for p in gen_AB.parameters()) + sum(p.numel() for p in gen_BA.parameters())
disc_params = sum(p.numel() for p in disc_A.parameters()) + sum(p.numel() for p in disc_B.parameters())
print(f"Generator params (both): {gen_params:,}")
print(f"Discriminator params (both): {disc_params:,}")
print(f"Total params: {gen_params + disc_params:,}\n")

# ==================== Setup Optimizers & Loss ====================
opt_gen = optim.Adam(
    list(gen_AB.parameters()) + list(gen_BA.parameters()),
    lr=lr, betas=(0.5, 0.999)
)
opt_disc = optim.Adam(
    list(disc_A.parameters()) + list(disc_B.parameters()),
    lr=lr, betas=(0.5, 0.999)
)

cycle_loss = CycleLoss(lambda_cycle=lambda_cycle)
identity_loss = IdentityLoss(lambda_identity=lambda_identity)
adversarial_loss = nn.MSELoss()

# Replay buffers
buffer_fake_A = ReplayBufferFixed(capacity=50)
buffer_fake_B = ReplayBufferFixed(capacity=50)

print("Starting CycleGAN training...\n")

# ==================== Training Loop ====================
for epoch in range(epochs):
    gen_AB.train()
    gen_BA.train()
    disc_A.train()
    disc_B.train()
    
    total_loss_gen = 0
    total_loss_disc = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch_idx, (real_A, real_B) in enumerate(pbar):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        
        # ========== Generator Step ==========
        opt_gen.zero_grad()
        
        # Generate fake images
        fake_B = gen_AB(real_A)
        fake_A = gen_BA(real_B)
        
        # Cycle reconstruction
        recon_A = gen_BA(fake_B)
        recon_B = gen_AB(fake_A)
        
        # Identity preservation
        identity_A = gen_BA(real_A)
        identity_B = gen_AB(real_B)
        
        # Losses
        loss_cycle = cycle_loss(real_A, recon_A, real_B, recon_B)
        loss_identity = identity_loss(real_A, identity_A, real_B, identity_B)
        
        # Adversarial losses
        fake_B_pred = disc_B(fake_B)
        fake_A_pred = disc_A(fake_A)
        
        real_label = torch.ones_like(fake_B_pred)
        loss_adv = adversarial_loss(fake_B_pred, real_label) + adversarial_loss(fake_A_pred, real_label)
        
        loss_gen = loss_cycle + loss_identity + loss_adv
        loss_gen.backward()
        opt_gen.step()
        
        # ========== Discriminator Step ==========
        opt_disc.zero_grad()
        
        # Push to replay buffers
        buffer_fake_B.push(fake_B.detach())
        buffer_fake_A.push(fake_A.detach())
        
        # Discriminate real images
        real_A_pred = disc_A(real_A)
        real_B_pred = disc_B(real_B)
        
        # Sample from replay buffer
        fake_B_sampled = buffer_fake_B.sample(batch_size, device)
        fake_A_sampled = buffer_fake_A.sample(batch_size, device)
        
        # Discriminate fake images
        fake_B_pred = disc_B(fake_B_sampled)
        fake_A_pred = disc_A(fake_A_sampled)
        
        real_label = torch.ones_like(real_A_pred)
        fake_label = torch.zeros_like(fake_B_pred)
        
        loss_disc_A = adversarial_loss(real_A_pred, real_label) + adversarial_loss(fake_A_pred, fake_label)
        loss_disc_B = adversarial_loss(real_B_pred, real_label) + adversarial_loss(fake_B_pred, fake_label)
        
        loss_disc = loss_disc_A + loss_disc_B
        loss_disc.backward()
        opt_disc.step()
        
        total_loss_gen += loss_gen.item()
        total_loss_disc += loss_disc.item()
        
        pbar.set_postfix({
            'L_G': f'{loss_gen.item():.4f}',
            'L_D': f'{loss_disc.item():.4f}'
        })
    
    avg_gen = total_loss_gen / len(data_loader)
    avg_disc = total_loss_disc / len(data_loader)
    
    print(f"Epoch {epoch+1} - Loss G: {avg_gen:.4f}, Loss D: {avg_disc:.4f}")
    
    # Save samples and checkpoint periodically
    if (epoch + 1) % checkpoint_interval == 0:
        # Save checkpoint
        torch.save({
            'gen_AB': gen_AB.state_dict(),
            'gen_BA': gen_BA.state_dict(),
            'disc_A': disc_A.state_dict(),
            'disc_B': disc_B.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_disc.state_dict(),
        }, f"outputs/checkpoints/cyclegan_epoch_{epoch+1}.pt")
        
        # Generate sample image
        with torch.no_grad():
            gen_AB.eval()
            gen_BA.eval()
            
            sample_A, sample_B = next(iter(data_loader))
            sample_A = sample_A.to(device)
            sample_B = sample_B.to(device)
            
            fake_B = gen_AB(sample_A)
            fake_A = gen_BA(sample_B)
            
            # Save visualization
            vis_img = torch.cat([sample_A, fake_B, sample_B, fake_A], dim=3)
            vis_img = (vis_img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            vis_img = vis_img.clamp(0, 1)
            
            # Convert to PIL and save
            img_np = (vis_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(f"outputs/images/cyclegan_epoch_{epoch+1}.png")

print("\n" + "=" * 60)
print("Training completed!")
print(f"Checkpoints saved to outputs/checkpoints/")
print(f"Sample images saved to outputs/images/")
print("=" * 60)
print("\nCycleGAN Features Demonstrated:")
print("  ✓ Unpaired image-to-image translation")
print("  ✓ Domain A ↔ Domain B transformation")
print("  ✓ Cycle consistency loss (A→B→A preservation)")
print("  ✓ Identity loss (domain similarity preservation)")
print("  ✓ PatchGAN discriminator (70×70 local patches)")
print("  ✓ ResNet generators (9 residual blocks)")
print("  ✓ Replay buffer for discriminator stability")
print("\nReal-world applications:")
print("  • Photo ↔ Painting style transfer")
print("  • Day ↔ Night domain adaptation")
print("  • Horse ↔ Zebra translation")
print("  • Medical imaging (CT ↔ MRI)")
print("  • Thermal ↔ RGB image translation")
print("=" * 60)
