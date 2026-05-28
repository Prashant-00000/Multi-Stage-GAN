import torch
import torch.nn as nn
from tqdm import tqdm
from src.common.utils import AverageMeter, save_checkpoint
from src.common.metrics import MetricsTracker

class GANTrainer:
    """Trainer for GAN models"""
    def __init__(self, generator, discriminator, opt_g, opt_d, criterion, device, config=None):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        self.metrics = MetricsTracker()
    
    def train_epoch(self, dataloader, epoch, z_dim=100):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        loss_d_avg = AverageMeter()
        loss_g_avg = AverageMeter()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (real, _) in enumerate(pbar):
            batch_size = real.size(0)
            real = real.view(batch_size, -1).to(self.device)
            
            # Train Discriminator
            z = torch.randn(batch_size, z_dim).to(self.device)
            fake = self.generator(z).detach()
            
            disc_real = self.discriminator(real)
            disc_fake = self.discriminator(fake)
            
            loss_d = self.criterion(disc_real, torch.full_like(disc_real, 0.9)) + \
                     self.criterion(disc_fake, torch.full_like(disc_fake, 0.1))
            
            self.opt_d.zero_grad()
            loss_d.backward()
            self.opt_d.step()
            
            # Train Generator
            z = torch.randn(batch_size, z_dim).to(self.device)
            fake = self.generator(z)
            output = self.discriminator(fake)
            loss_g = self.criterion(output, torch.ones_like(output))
            
            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()
            
            # Update metrics
            loss_d_avg.update(loss_d.item())
            loss_g_avg.update(loss_g.item())
            self.metrics.update('loss_d', loss_d.item())
            self.metrics.update('loss_g', loss_g.item())
            
            pbar.set_postfix({'loss_d': loss_d_avg.avg, 'loss_g': loss_g_avg.avg})
        
        return loss_d_avg.avg, loss_g_avg.avg
    
    def validate(self, dataloader, z_dim=100, num_samples=10):
        """Validate generator on real data"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            # Generate samples
            z = torch.randn(num_samples, z_dim).to(self.device)
            fake_images = self.generator(z)
            
            # Check discriminator output on fake images
            disc_output = self.discriminator(fake_images)
        
        return fake_images, disc_output.mean().item()
