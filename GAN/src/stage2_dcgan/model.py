import torch
import torch.nn as nn
from src.stage2_dcgan.generator import DCGANGenerator
from src.stage2_dcgan.discriminator import DCGANDiscriminator

class DCGAN(nn.Module):
    """DCGAN Model - Wrapper for generator and discriminator"""
    def __init__(self, z_dim=100, channels=1, feature_maps=64):
        super().__init__()
        self.generator = DCGANGenerator(z_dim, channels, feature_maps)
        self.discriminator = DCGANDiscriminator(channels, feature_maps)
        self.z_dim = z_dim
    
    def forward(self, z):
        return self.generator(z)
    
    def discriminate(self, x):
        return self.discriminator(x)

__all__ = ['DCGANGenerator', 'DCGANDiscriminator', 'DCGAN']
