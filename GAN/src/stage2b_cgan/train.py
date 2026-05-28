import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.stage2b_cgan.generator import ConditionalGenerator
from src.stage2b_cgan.discriminator import ConditionalDiscriminator
from src.common.utils import save_checkpoint, init_weights
from tqdm import tqdm
import os

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
z_dim = 100
num_classes = 10
embedding_dim = 100
img_size = 28
channels = 1
feature_maps = 64
batch_size = 64
lr = 2e-4
epochs = 50
checkpoint_interval = 5

# Create output directories
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)

# Data Loading
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

# Initialize Models
gen = ConditionalGenerator(z_dim, num_classes, embedding_dim, channels, feature_maps).to(device)
disc = ConditionalDiscriminator(num_classes, embedding_dim, channels, feature_maps).to(device)

# Initialize weights
init_weights(gen, init_type='normal', gain=0.02)
init_weights(disc, init_type='normal', gain=0.02)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss Function
criterion = nn.BCELoss()

print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")

# Training Loop
print("\nStarting training...")
for epoch in range(epochs):
    gen.train()
    disc.train()
    
    loss_d_total = 0
    loss_g_total = 0
    batch_count = 0
    
    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{epochs}]")
    
    for batch_idx, (real, labels) in enumerate(pbar):
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
        
        loss_d_total += loss_d.item()
        loss_g_total += loss_g.item()
        batch_count += 1
        
        pbar.set_postfix({
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item()
        })
    
    avg_loss_d = loss_d_total / batch_count
    avg_loss_g = loss_g_total / batch_count
    
    print(f"Epoch [{epoch}/{epochs}] - Loss D: {avg_loss_d:.4f}, Loss G: {avg_loss_g:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"outputs/checkpoints/cgan_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'generator': gen.state_dict(),
            'discriminator': disc.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_disc.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Generate sample images per class
    if (epoch + 1) % checkpoint_interval == 0:
        gen.eval()
        with torch.no_grad():
            from torchvision.utils import save_image
            
            # Generate samples for each class
            z = torch.randn(10, z_dim).to(device)
            labels_gen = torch.arange(10).to(device)
            fake_images = gen(z, labels_gen)
            
            save_image(fake_images * 0.5 + 0.5, f"outputs/images/cgan_epoch_{epoch}.png", nrow=5)
        gen.train()

print("Training completed!")
