import torch
import torch.nn as nn
from src.stage2b_cgan.generator import ConditionalGenerator
from src.stage2b_cgan.discriminator import ConditionalDiscriminator

class CGAN(nn.Module):
    """Conditional GAN Model"""
    def __init__(self, z_dim=100, num_classes=10, embedding_dim=100, channels=1, feature_maps=64):
        super().__init__()
        self.generator = ConditionalGenerator(z_dim, num_classes, embedding_dim, channels, feature_maps)
        self.discriminator = ConditionalDiscriminator(num_classes, embedding_dim, channels, feature_maps)
        self.z_dim = z_dim
    
    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def discriminate(self, x, labels):
        return self.discriminator(x, labels)

__all__ = ['ConditionalGenerator', 'ConditionalDiscriminator', 'CGAN']
