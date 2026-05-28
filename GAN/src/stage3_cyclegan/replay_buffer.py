import torch
import random
import numpy as np

class ReplayBuffer:
    """Replay buffer for storing generated images"""
    def __init__(self, capacity=50):
        """
        Args:
            capacity: maximum number of images to store
        """
        self.capacity = capacity
        self.buffer = []
    
    def push(self, images):
        """
        Add images to the buffer
        
        Args:
            images: tensor of shape (batch_size, channels, height, width)
        """
        batch_size = images.size(0)
        
        for i in range(batch_size):
            img = images[i].detach().cpu()
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(img)
            else:
                # Replace random image in buffer
                if random.rand() > 0.5:
                    idx = random.randint(0, self.capacity - 1)
                    self.buffer[idx] = img
    
    def sample(self, batch_size):
        """
        Sample batch of images from buffer
        
        Args:
            batch_size: number of images to sample
        
        Returns:
            sampled images tensor
        """
        if len(self.buffer) == 0:
            return None
        
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        images = [self.buffer[i] for i in indices]
        
        # Pad if necessary
        if len(images) < batch_size:
            indices = random.sample(range(len(self.buffer)), batch_size - len(images))
            images.extend([self.buffer[i] for i in indices])
        
        return torch.stack(images)

# Fix the random.rand() -> torch.rand() issue
class ReplayBufferFixed:
    """Replay buffer for storing generated images - Fixed version"""
    def __init__(self, capacity=50):
        """
        Args:
            capacity: maximum number of images to store
        """
        self.capacity = capacity
        self.buffer = []
    
    def push(self, images):
        """
        Add images to the buffer
        
        Args:
            images: tensor of shape (batch_size, channels, height, width)
        """
        batch_size = images.size(0)
        
        for i in range(batch_size):
            img = images[i].detach().cpu()
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(img)
            else:
                # Replace random image in buffer with 50% probability
                if torch.rand(1).item() > 0.5:
                    idx = random.randint(0, self.capacity - 1)
                    self.buffer[idx] = img
    
    def sample(self, batch_size, device):
        """
        Sample batch of images from buffer
        
        Args:
            batch_size: number of images to sample
            device: device to move images to
        
        Returns:
            sampled images tensor on specified device
        """
        if len(self.buffer) == 0:
            return None
        
        # Sample indices
        available_count = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), available_count)
        images = [self.buffer[i] for i in indices]
        
        # Pad if necessary
        if len(images) < batch_size:
            pad_indices = random.sample(range(len(self.buffer)), batch_size - len(images))
            images.extend([self.buffer[i] for i in pad_indices])
        
        # Stack and move to device
        return torch.stack(images).to(device)
