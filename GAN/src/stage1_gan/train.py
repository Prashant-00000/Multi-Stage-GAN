import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.stage1_gan.generator import Generator
from src.stage1_gan.discriminator import Discriminator
from src.training_engine.gan_loop import train_step

# 1. Setup Hyperparameters from Blueprints
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
z_dim = 100
batch_size = 64
epochs = 50

# 2. Data Loading (MNIST)
# Standardize images to [-1, 1] to match Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
loader = DataLoader(datasets.MNIST(root="data/", transform=transform, download=True), 
                    batch_size=batch_size, shuffle=True)

# 3. Initialize Models & Optimizers
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 4. The Training Loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        loss_D, loss_G = train_step(
            real, gen, disc, opt_gen, opt_disc, criterion, device, z_dim
        )
        
        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} "
                  f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")