import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import os

class FIDCalculator:
    def __init__(self, device):
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).to(device)
        self.model.fc = nn.Identity()
        self.model.eval()

    def get_features(self, dataloader):
        features = []
        with torch.no_grad():
            for batch in dataloader:
                # Handle both tuple (images, labels) and single tensor inputs
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                if batch.shape[1] == 1:
                    batch = batch.repeat(1, 3, 1, 1)
                batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    def compute_statistics(self, features):
        """Calculates mean and covariance for a feature set."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def save_statistics(self, dataloader, export_path):
        """Processes real images once and saves the stats to a file."""
        print(f"Extracting features for {export_path}...")
        feats = self.get_features(dataloader)
        mu, sigma = self.compute_statistics(feats)
        np.savez(export_path, mu=mu, sigma=sigma)
        print("Statistics saved successfully.")

    def calculate_fid_from_stats(self, fake_features, real_stats_path):
        """Calculates FID using fake features and saved real stats."""
        # Load pre-calculated real stats
        with np.load(real_stats_path) as data:
            mu1, sigma1 = data['mu'], data['sigma']
        
        # Calculate stats for fake images
        mu2, sigma2 = self.compute_statistics(fake_features)

        # FID Math
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid