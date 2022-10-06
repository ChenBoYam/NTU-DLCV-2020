import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import torch
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib import cm

output_dir = os.path.join(sys.argv[1])

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#workspace_dir = os.path.join(sys.argv[1])
#output_dir = os.path.join(sys.argv[2])
import pandas as pd
same_seeds(0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=100):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=12544, z_dim=100):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(z_dim, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x, ):
        z, mu, log_var = self.encode(x)
        _z = self.decode(z)
        return _z, mu, log_var ,z

def loss_function(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD, MSE ,KLD

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64 )
    return x

model = VAE()
if torch.cuda.is_available():
    model.cuda()
    
model.load_state_dict(torch.load('./model/vae2_45.pkl?dl=1'))
model.eval()
sample1 = torch.randn(32, 100).view(-1, 100, 1, 1)
if torch.cuda.is_available():
    sample1 = sample1.cuda()
recon_batch = model.decode(sample1)     
recon_batch =recon_batch*0.5+0.5
save = to_img(recon_batch.cpu().data)
save_image(save, output_dir)
