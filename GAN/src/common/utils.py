import torch
import os
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_directories(config):
    """Create necessary output directories"""
    paths = config.get('paths', {})
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)

def save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, checkpoint_dir):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict() if generator else None,
        'discriminator': discriminator.state_dict(),
        'opt_g': opt_g.state_dict() if opt_g else None,
        'opt_d': opt_d.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, generator, discriminator, opt_g, opt_d, device):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if generator and checkpoint['generator']:
        generator.load_state_dict(checkpoint['generator'])
    
    discriminator.load_state_dict(checkpoint['discriminator'])
    
    if opt_g and checkpoint['opt_g']:
        opt_g.load_state_dict(checkpoint['opt_g'])
    
    opt_d.load_state_dict(checkpoint['opt_d'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch

def get_device():
    """Get available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def print_model_summary(model, input_size):
    """Print model summary"""
    try:
        from torchsummary import summary
        summary(model, input_size=input_size)
    except ImportError:
        print("torchsummary not installed. Skipping model summary.")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

def denormalize(img_tensor):
    """Denormalize image from [-1, 1] to [0, 1] or [0, 255]"""
    if len(img_tensor.shape) == 4:
        # Batch of images
        return (img_tensor + 1) / 2.0
    else:
        # Single image
        return (img_tensor + 1) / 2.0

def normalize(img_tensor):
    """Normalize image from [0, 1] to [-1, 1]"""
    return img_tensor * 2.0 - 1.0

def set_requires_grad(model, requires_grad):
    """Set requires_grad for all parameters in model"""
    for param in model.parameters():
        param.requires_grad = requires_grad

def init_weights(model, init_type='normal', gain=0.02):
    """Initialize model weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'
