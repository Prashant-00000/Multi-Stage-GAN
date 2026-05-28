import torch
import torch.nn as nn
from src.stage3_cyclegan.generator_resnet import ResNetGenerator
from src.stage3_cyclegan.discriminator_patchgan import PatchGANDiscriminator

class CycleGAN(nn.Module):
    """CycleGAN Model for unpaired image-to-image translation"""
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9, feature_maps=64):
        super().__init__()
        
        # Generators: A->B and B->A
        self.gen_AB = ResNetGenerator(in_channels, out_channels, num_residual_blocks, feature_maps)
        self.gen_BA = ResNetGenerator(in_channels, out_channels, num_residual_blocks, feature_maps)
        
        # Discriminators: D_A and D_B
        self.disc_A = PatchGANDiscriminator(in_channels, feature_maps)
        self.disc_B = PatchGANDiscriminator(in_channels, feature_maps)
    
    def forward(self, real_A, real_B):
        """
        Forward pass for CycleGAN
        
        Args:
            real_A: real images from domain A
            real_B: real images from domain B
        
        Returns:
            dict with all generated images
        """
        # A -> B -> A
        fake_B = self.gen_AB(real_A)
        cycled_A = self.gen_BA(fake_B)
        
        # B -> A -> B
        fake_A = self.gen_BA(real_B)
        cycled_B = self.gen_AB(fake_A)
        
        # Identity mapping (for same domain)
        identity_A = self.gen_BA(real_A)
        identity_B = self.gen_AB(real_B)
        
        return {
            'fake_A': fake_A,
            'fake_B': fake_B,
            'cycled_A': cycled_A,
            'cycled_B': cycled_B,
            'identity_A': identity_A,
            'identity_B': identity_B
        }

__all__ = ['CycleGAN', 'ResNetGenerator', 'PatchGANDiscriminator']
