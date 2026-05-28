import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """DCGAN Generator - Convolutional architecture"""
    def __init__(self, z_dim=100, channels=1, feature_maps=64):
        super().__init__()
        self.z_dim = z_dim
        
        # Input: z_dim noise vector
        # Output: channels x 28 x 28
        
        self.model = nn.Sequential(
            # Input: (batch_size, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (batch_size, feature_maps*4, 4, 4)
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (batch_size, feature_maps*2, 8, 8)
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (batch_size, feature_maps, 16, 16)
            
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # State: (batch_size, channels, 32, 32) -> adjust for 28x28
        )
    
    def forward(self, z):
        # Reshape noise for conv layer
        z = z.view(z.size(0), self.z_dim, 1, 1)
        x = self.model(z)
        # Crop to 28x28 (output is 32x32)
        return x[:, :, :28, :28]


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
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps*4, 3, 3) or similar
            
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (batch_size, 1, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x).view(x.size(0), -1)
