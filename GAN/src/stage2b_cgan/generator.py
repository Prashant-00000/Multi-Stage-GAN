import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    """Conditional GAN Generator - class-conditioned generation"""
    def __init__(self, z_dim=100, num_classes=10, embedding_dim=100, channels=1, feature_maps=64):
        super().__init__()
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Concatenate z and class embedding, then project to initial size
        self.fc = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, feature_maps * 4 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolutional layers
        self.model = nn.Sequential(
            # Input: (batch_size, feature_maps*4, 4, 4)
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
            # State: (batch_size, channels, 32, 32) -> 28x28 after cropping
        )
    
    def forward(self, z, labels):
        """
        Args:
            z: noise vector of shape (batch_size, z_dim)
            labels: class labels of shape (batch_size,)
        
        Returns:
            fake images of shape (batch_size, channels, 28, 28)
        """
        # Embed class labels
        class_embed = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        z_concat = torch.cat([z, class_embed], dim=1)
        
        # Project to initial feature map size
        x = self.fc(z_concat)
        x = x.view(x.size(0), -1, 4, 4)
        
        # Generate image
        img = self.model(x)
        
        # Crop to 28x28 if needed
        if img.size(2) > 28:
            img = img[:, :, :28, :28]
        
        return img
