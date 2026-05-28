import torch
import torch.nn as nn

class ClassEmbedding(nn.Module):
    """Embedding layer for class labels"""
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, labels):
        """
        Args:
            labels: tensor of shape (batch_size,) with class indices
        
        Returns:
            embeddings: tensor of shape (batch_size, embedding_dim)
        """
        return self.embedding(labels)

class ConditionalBatchNorm2d(nn.Module):
    """Batch normalization conditioned on class labels"""
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_fc = nn.Linear(num_classes, num_features)
        self.beta_fc = nn.Linear(num_classes, num_features)
    
    def forward(self, x, labels):
        """
        Args:
            x: feature maps of shape (batch_size, channels, height, width)
            labels: class embeddings of shape (batch_size, num_classes)
        
        Returns:
            output: conditioned feature maps
        """
        # Apply batch norm
        out = self.bn(x)
        
        # Get conditional scale and shift
        gamma = self.gamma_fc(labels).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(labels).unsqueeze(2).unsqueeze(3)
        
        # Apply conditional scale and shift
        out = gamma * out + beta
        return out
