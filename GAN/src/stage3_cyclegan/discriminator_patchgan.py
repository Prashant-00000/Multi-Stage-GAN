import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator - classifies image patches"""
    def __init__(self, in_channels=3, feature_maps=64):
        super().__init__()
        
        # Discriminator architecture
        self.model = nn.Sequential(
            # Input: (batch_size, in_channels, 256, 256) or any size
            nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, feature_maps, 128, 128)
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, feature_maps*2, 64, 64)
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, feature_maps*4, 32, 32)
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, feature_maps*8, 16, 16)
            
            # PatchGAN: Final layer outputs logits for each patch
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 1, bias=True),
            # Output: (batch_size, 1, 16, 16) - one classification per patch
        )
    
    def forward(self, x):
        return self.model(x)
