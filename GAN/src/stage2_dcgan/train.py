import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.stage2_dcgan.generator import DCGANGenerator
from src.stage2_dcgan.discriminator import DCGANDiscriminator
from src.training_engine.gan_loop import train_step
from src.common.utils import save_checkpoint, create_directories, init_weights
from tqdm import tqdm
import os

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
z_dim = 100
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
os.makedirs("outputs/logs", exist_ok=True)

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
gen = DCGANGenerator(z_dim, channels, feature_maps).to(device)
disc = DCGANDiscriminator(channels, feature_maps).to(device)

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
    
    for batch_idx, (real, _) in enumerate(pbar):
        real = real.to(device)
        loss_D, loss_G = train_step(
            real, gen, disc, opt_gen, opt_disc, criterion, device, z_dim
        )
        
        loss_d_total += loss_D
        loss_g_total += loss_G
        batch_count += 1
        
        pbar.set_postfix({
            'loss_d': loss_D,
            'loss_g': loss_G
        })
    
    avg_loss_d = loss_d_total / batch_count
    avg_loss_g = loss_g_total / batch_count
    
    print(f"Epoch [{epoch}/{epochs}] - Loss D: {avg_loss_d:.4f}, Loss G: {avg_loss_g:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"outputs/checkpoints/dcgan_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'generator': gen.state_dict(),
            'discriminator': disc.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_disc.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Generate sample images
    if (epoch + 1) % checkpoint_interval == 0:
        gen.eval()
        with torch.no_grad():
            z = torch.randn(16, z_dim).to(device)
            fake_images = gen(z)
            # Save visualization
            from torchvision.utils import save_image
            save_image(fake_images * 0.5 + 0.5, f"outputs/images/dcgan_epoch_{epoch}.png", nrow=4)
        gen.train()

print("Training completed!")
