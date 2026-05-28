"""
Comprehensive Evaluation Module
Evaluates GAN models using multiple metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import os

class InceptionModel:
    """Inception Model for feature extraction"""
    def __init__(self, device):
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(device)
        self.model.fc = nn.Identity()
        self.model.eval()
    
    def extract_features(self, images):
        """Extract features from images using Inception V3"""
        with torch.no_grad():
            features = self.model(images)
        return features
    
    def classify(self, images):
        """Get classification logits"""
        inception_model_full = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(self.device)
        inception_model_full.eval()
        with torch.no_grad():
            logits = inception_model_full(images)
        return logits
