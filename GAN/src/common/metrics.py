import torch
import numpy as np

class MetricsTracker:
    """Track metrics during training"""
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name, last_n=100):
        """Get average of last N values"""
        if metric_name not in self.metrics:
            return 0.0
        values = self.metrics[metric_name][-last_n:]
        return np.mean(values) if values else 0.0
    
    def reset(self):
        self.metrics = {}

def compute_inception_score(images, batch_size=32, num_splits=10):
    """
    Compute Inception Score for generated images
    
    Note: Inception Score requires pre-trained classifier. For proper 
    implementation, use FID score instead (see evaluation/fid.py).
    
    Args:
        images: tensor of shape (N, C, H, W) with values in [-1, 1]
        batch_size: batch size for inception network
        num_splits: number of splits for computing mean and std
    
    Returns:
        is_mean, is_std: mean and std of Inception Score (0.0 if unavailable)
    """
    # Normalize images to [0, 1]
    images = (images + 1) / 2.0
    
    # Clip to valid range
    images = torch.clamp(images, 0, 1)
    
    # Note: Inception Score computation requires pre-trained classifier
    # For now, return placeholder values. Use FID score for evaluation instead.
    return 0.0, 0.0

class L1Distance:
    """Compute L1 distance between image pairs"""
    def __init__(self):
        self.criterion = torch.nn.L1Loss()
    
    def compute(self, img1, img2):
        """Compute L1 distance"""
        return self.criterion(img1, img2).item()

class SSIMDistance:
    """Compute Structural Similarity Index"""
    def __init__(self, window_size=11, sigma=1.5):
        self.window_size = window_size
        self.sigma = sigma
    
    def compute(self, img1, img2):
        """Compute SSIM between two images"""
        # Gaussian window
        kernel_x = torch.exp(-torch.arange(self.window_size).float()**2 / (2 * self.sigma**2))
        kernel_x = kernel_x / kernel_x.sum()
        
        # Create 2D Gaussian kernel
        kernel = kernel_x.unsqueeze(1) @ kernel_x.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # Normalize to [-1, 1]
        img1_norm = img1 / 2.0 + 0.5
        img2_norm = img2 / 2.0 + 0.5
        
        # Compute SSIM (simplified version)
        c1, c2 = 0.01**2, 0.03**2
        
        mean1 = torch.nn.functional.conv2d(img1_norm, kernel, padding=self.window_size//2)
        mean2 = torch.nn.functional.conv2d(img2_norm, kernel, padding=self.window_size//2)
        
        var1 = torch.nn.functional.conv2d(img1_norm**2, kernel, padding=self.window_size//2) - mean1**2
        var2 = torch.nn.functional.conv2d(img2_norm**2, kernel, padding=self.window_size//2) - mean2**2
        cov12 = torch.nn.functional.conv2d(img1_norm*img2_norm, kernel, padding=self.window_size//2) - mean1*mean2
        
        ssim = ((2*mean1*mean2 + c1) * (2*cov12 + c2)) / ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        return ssim.mean().item()
