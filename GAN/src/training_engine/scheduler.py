import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR

class Scheduler:
    """Learning rate scheduler wrapper"""
    def __init__(self, optimizer, scheduler_type='step', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'step':
            self.scheduler = StepLR(optimizer, step_size=kwargs.get('step_size', 10), 
                                   gamma=kwargs.get('gamma', 0.5))
        elif scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.99))
        else:
            self.scheduler = None
    
    def step(self):
        if self.scheduler:
            self.scheduler.step()
    
    def get_last_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()
        return [self.optimizer.defaults['lr']]
