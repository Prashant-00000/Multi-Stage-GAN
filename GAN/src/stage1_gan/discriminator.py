import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self,img_dim=784):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Linear(img_dim,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)    