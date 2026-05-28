import torch
import torch.nn as nn

class WassersteinLoss(nn.Module):
    """Wasserstein loss for improved training stability"""
    def forward(self, real_pred, fake_pred):
        return -torch.mean(real_pred) + torch.mean(fake_pred)

class HingeLoss(nn.Module):
    """Hinge loss for GAN training"""
    def forward(self, real_pred, fake_pred):
        loss_real = torch.mean(torch.relu(1.0 - real_pred))
        loss_fake = torch.mean(torch.relu(1.0 + fake_pred))
        return loss_real + loss_fake

class CycleLoss(nn.Module):
    """Cycle consistency loss for unpaired image translation"""
    def __init__(self, lambda_cycle=10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real_A, cycled_A, real_B, cycled_B):
        """
        Calculate cycle consistency loss
        A->B->A should reconstruct A, B->A->B should reconstruct B
        """
        loss_A = self.l1_loss(real_A, cycled_A)
        loss_B = self.l1_loss(real_B, cycled_B)
        return self.lambda_cycle * (loss_A + loss_B)

class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN - preserves color when translating similar domains"""
    def __init__(self, lambda_identity=0.5):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real_A, identity_A, real_B, identity_B):
        """If translating to a similar domain, images should remain unchanged"""
        loss_A = self.l1_loss(real_A, identity_A)
        loss_B = self.l1_loss(real_B, identity_B)
        return self.lambda_identity * (loss_A + loss_B)

def adversarial_loss(discriminator_output, is_real, label_smoothing=True):
    """
    Binary cross-entropy loss for GAN
    
    Args:
        discriminator_output: D(x) values
        is_real: whether these are real samples
        label_smoothing: apply label smoothing
    
    Returns:
        BCE loss with optional label smoothing
    """
    criterion = nn.BCELoss()
    
    if label_smoothing:
        if is_real:
            target = torch.full_like(discriminator_output, 0.9)
        else:
            target = torch.full_like(discriminator_output, 0.1)
    else:
        target = torch.ones_like(discriminator_output) if is_real else torch.zeros_like(discriminator_output)
    
    return criterion(discriminator_output, target)
