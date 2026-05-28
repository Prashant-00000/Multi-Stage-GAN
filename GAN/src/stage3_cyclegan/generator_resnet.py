import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with reflection padding"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN"""
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9, feature_maps=64):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, feature_maps, 7, 1, 0, bias=True),
            nn.InstanceNorm2d(feature_maps),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, 3, 2, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, 2, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 4),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(feature_maps * 4, feature_maps * 4)
            for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 3, 2, 1, 1, bias=True),
            nn.InstanceNorm2d(feature_maps * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 3, 2, 1, 1, bias=True),
            nn.InstanceNorm2d(feature_maps),
            nn.ReLU(True)
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(feature_maps, out_channels, 7, 1, 0, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x
