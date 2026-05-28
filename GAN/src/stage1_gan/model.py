import torch
import torch.nn as nn
from src.stage1_gan.generator import Generator
from src.stage1_gan.discriminator import Discriminator

class GAN(nn.Module):
    """Wrapper for Stage 1 GAN components"""
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, z):
        return self.generator(z)
    
    def discriminate(self, x):
        return self.discriminator(x)

__all__ = ['Generator', 'Discriminator', 'GAN']
