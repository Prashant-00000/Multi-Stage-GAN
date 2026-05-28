import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    """Conditional GAN Discriminator - class-conditioned discrimination"""
    def __init__(self, num_classes=10, embedding_dim=100, channels=1, feature_maps=64):
        super().__init__()
        
        # Class embedding for discriminator
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Project class embedding to image dimensions
        self.class_proj = nn.Sequential(
            nn.Linear(embedding_dim, channels * 28 * 28),
            nn.ReLU(True)
        )
        
        # Main convolutional path
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, channels*2, 28, 28) [concatenated with projected class]
            nn.Conv2d(channels + channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps, 14, 14)
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps*2, 7, 7)
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (batch_size, feature_maps*4, 3, 3)
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(feature_maps * 4 * 3 * 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        """
        Args:
            x: images of shape (batch_size, channels, 28, 28)
            labels: class labels of shape (batch_size,)
        
        Returns:
            discrimination scores of shape (batch_size, 1)
        """
        # Embed and project class labels
        class_embed = self.class_embedding(labels)
        class_proj = self.class_proj(class_embed).view(x.size(0), -1, 28, 28)
        
        # Concatenate image and projected class
        x_concat = torch.cat([x, class_proj], dim=1)
        
        # Pass through convolutional layers
        features = self.conv_layers(x_concat)
        
        # Flatten and classify
        features_flat = features.view(features.size(0), -1)
        output = self.fc(features_flat)
        
        return output
