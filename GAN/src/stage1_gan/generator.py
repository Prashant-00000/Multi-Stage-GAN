import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super(Generator, self).__init__()
        # Architecture follows Blueprint: Linear -> LeakyReLU -> Linear -> Tanh
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh() # Output in [-1, 1] for normalization
        )

    def forward(self, z):
        return self.gen(z)
