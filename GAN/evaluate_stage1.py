import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from evaluation.fid import FIDCalculator
from src.stage1_gan.generator import Generator
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup
z_dim = 100
num_real_samples = 2048  # Use 2048 real samples
num_fake_samples = 2048  # Generate 2048 fake images
batch_size = 64

# 1. Load MNIST dataset for real images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_loader = DataLoader(
    datasets.MNIST(root="data/", train=True, transform=transform, download=True),
    batch_size=batch_size,
    shuffle=False
)

# Collect only first N samples
real_images = []
for batch, _ in full_loader:
    real_images.append(batch)
    if sum(img.shape[0] for img in real_images) >= num_real_samples:
        break
real_images = torch.cat(real_images, dim=0)[:num_real_samples]

class RealImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]

real_dataset = RealImageDataset(real_images)
real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

# 2. Initialize FID Calculator
fid_calculator = FIDCalculator(device)

# 3. Calculate and save real image statistics
real_stats_path = "outputs/real_mnist_stats.npz"
os.makedirs("outputs", exist_ok=True)

print("Calculating real MNIST statistics...")
fid_calculator.save_statistics(real_loader, real_stats_path)

# 4. Generate fake images from generator
print(f"\nGenerating {num_fake_samples} fake MNIST images...")
gen = Generator(z_dim=z_dim).to(device)
gen.eval()

fake_images = []
with torch.no_grad():
    for i in range(0, num_fake_samples, batch_size):
        batch_size_current = min(batch_size, num_fake_samples - i)
        z = torch.randn(batch_size_current, z_dim).to(device)
        fake_batch = gen(z).reshape(batch_size_current, 1, 28, 28)
        fake_images.append(fake_batch.cpu())

fake_images = torch.cat(fake_images, dim=0)
print(f"Generated {fake_images.shape[0]} fake images")

# 5. Create dataloader for fake images
class FakeImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]

fake_dataset = FakeImageDataset(fake_images)
fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

# 6. Extract features from fake images
print("Extracting features from fake images...")
fake_features = fid_calculator.get_features(fake_loader)

# 7. Calculate FID score
print("Calculating FID score...")
fid_score = fid_calculator.calculate_fid_from_stats(fake_features, real_stats_path)

print(f"\n{'='*50}")
print(f"FID Score (Fréchet Inception Distance): {fid_score:.2f}")
print(f"{'='*50}")
print(f"Note: Lower FID is better. Typical MNIST FID ranges from 3-15.")
