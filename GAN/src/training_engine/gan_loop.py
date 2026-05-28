import torch

def train_step(real, G, D, opt_G, opt_D, criterion, device, z_dim):
    batch_size = real.size(0)
    real = real.view(batch_size, -1).to(device)

    # --- 1. Train Discriminator (D) ---
    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)
    
    # CRITICAL: Detach stops gradients from flowing into G during D's turn
    disc_real = D(real)
    disc_fake = D(fake.detach()) 

    # Loss with Label Smoothing: 0.9 for real, 0.1 for fake
    loss_D = criterion(disc_real, torch.full_like(disc_real, 0.9)) + \
             criterion(disc_fake, torch.full_like(disc_fake, 0.1))
    
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # --- 2. Train Generator (G) ---
    # Fresh z, never reuse the one from D's turn
    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)
    output = D(fake)
    
    # G wants D to output 1.0 (fooled)
    loss_G = criterion(output, torch.ones_like(output))
    
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()


def train_step_conv(real, G, D, opt_G, opt_D, criterion, device, z_dim):
    """Train step for Convolutional GANs (DCGAN, CGAN, etc) - keeps 4D shape"""
    batch_size = real.size(0)
    real = real.to(device)  # Don't flatten - keep 4D shape

    # --- 1. Train Discriminator (D) ---
    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)
    
    # CRITICAL: Detach stops gradients from flowing into G during D's turn
    disc_real = D(real)
    disc_fake = D(fake.detach()) 

    # Loss with Label Smoothing: 0.9 for real, 0.1 for fake
    loss_D = criterion(disc_real, torch.full_like(disc_real, 0.9)) + \
             criterion(disc_fake, torch.full_like(disc_fake, 0.1))
    
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # --- 2. Train Generator (G) ---
    # Fresh z, never reuse the one from D's turn
    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)
    output = D(fake)
    
    # G wants D to output 1.0 (fooled)
    loss_G = criterion(output, torch.ones_like(output))
    
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()