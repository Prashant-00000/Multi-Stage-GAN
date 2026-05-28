import torch
import torch.nn as nn

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator - Convolutional architecture"""
    def __init__(self, channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: (batch_size, channels, 28, 28)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps, 14, 14)
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps*2, 7, 7)
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps*4, 4, 4)
            
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (batch_size, 1, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x).view(x.size(0), -1)
