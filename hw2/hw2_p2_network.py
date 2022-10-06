
import torch
import torchvision.models
from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import os
class fcn32s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn32s, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, num_classes, 64 , 32 , 0, bias=False),
        )
    def  forward (self, x) :        
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        return x

class fcn16s(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(fcn16s, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.to_pool4 = nn.Sequential(*list(self.vgg.features.children())[:24])
        self.to_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, 512, 4 , 2 , 0, bias=False)
            )
        self.upsample16 = nn.ConvTranspose2d(512, num_classes, 16 , 16 , 0, bias=False)
    def  forward (self, x) :        
        pool4_output = self.to_pool4(x) 
        x = self.to_pool5(pool4_output)
        x = self.vgg.classifier(x)    
        x = self.upsample16(x+pool4_output)
        return x

