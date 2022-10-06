import torch
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim
import itertools
from visutils import tb_writer, select_n_random
import tblog 
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import time
import random
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from matplotlib import cm

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class dataReader(Dataset):
    def __init__(self, root, transforms, train = True):
        self.root = root
        self.train = train
        if self.train:
            self.label_file = os.path.join(self.root, "train.csv")
        else:
            self.label_file = os.path.join(self.root, "test.csv")
        self.transforms = transforms
        with open(self.label_file) as f:
            next(f)
            lines = [line.strip('\n').split(',') for line in f]
            self.X_name, self.y = np.array(lines, dtype = str)[:,0], np.array(lines)[:,1].astype(int)
        self.length = len(self.X_name)
        self.label = torch.from_numpy(self.y).long()

    def __getitem__(self, idx):
        if self.train:
            img = Image.open(os.path.join(os.path.join(self.root, "train"), self.X_name[idx]))
        else:
            img = Image.open(os.path.join(os.path.join(self.root, "test"), self.X_name[idx]))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return self.length

    def get_label(self):
        return self.label

same_seeds(0)

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((28, 28)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((28, 28)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
root = "./hw3_data/digits/"
source = "svhn"
target = "usps"
source_path = os.path.join(root,source)
target_path = os.path.join(root,target)

train_source_dataset = dataReader(source_path, transforms=source_transform)
train_target_dataset = dataReader(target_path, transforms=target_transform)
test_source_dataset = dataReader(source_path, transforms=test_transform, train = False)
test_target_dataset = dataReader(target_path, transforms=test_transform, train = False)

train_source_dataloader = DataLoader(train_source_dataset, batch_size = 32, shuffle=True)
train_target_dataloader = DataLoader(train_target_dataset, batch_size = 32, shuffle=True)
test_source_dataloader = DataLoader(test_source_dataset, batch_size = 128, shuffle=False)
test_target_dataloader = DataLoader(test_target_dataset, batch_size = 128, shuffle=False)


img_size = 28

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        def block(in_features, out_features, parameterized=True):
            if parameterized:
                layers = [
                    nn.Conv2d(in_features, out_features, 5, 1, 1), 
                    nn.PReLU(), 
                    nn.MaxPool2d(2, stride=2)
                ]
            else:
                layers = [nn.Conv2d(in_features, out_features, 5, 1, 1), 
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(2, stride=2)]
            return layers
        
        reduced_img_size = ((img_size-2)/2-2)/2
        
        self.private = nn.Sequential(
            *block(1, 32)
        )
        self.shared = nn.Sequential(
            *block(32, 48),
            Flatten(),
            nn.Linear(48*(int(reduced_img_size)**2), 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
        
    def forward(self, img):
        return self.shared(self.private(img))
class Discriminator(nn.Module):
    def __init__(self, in_features=1, out_features=64):
        super(Discriminator, self).__init__()
        def block(in_features, out_features):
            layers = [
                nn.Conv2d(in_features, out_features, 3, 2, 1),
                nn.Dropout(p=0.1),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
            
            return layers
        
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.blocks = nn.Sequential(
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024)
        )
        
        self.l2 = nn.Sequential(
            nn.Linear(1024*(int(32/2**4))**2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        x = self.l1(img)
        x = self.blocks(x)
        out = self.l2(x.view(img.shape[0], -1))
        return out
class Generator(nn.Module):
    def __init__(self, l1_features=64):
        super(Generator, self).__init__()
        self.fc = nn.Linear(10, img_size**2)
        self.l1 = nn.Sequential(
            nn.Conv2d(1*2, l1_features, 3, 1, 1), 
            nn.ReLU(inplace=True)
        )        
        resblocks = []
        for _ in range(6):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        
        self.l2 = nn.Sequential(nn.Conv2d(l1_features, 1, 3, 1, 1), nn.Tanh())
        
    def forward(self, img, z):
        x = torch.cat((img, self.fc(z).view(img.shape[0])), dim=1)
        x = self.l1(x)
        x = self.resblocks(x)
        x = self.l2(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features)
        )
        
    def forward(self, x):
        return x + self.block(x)





     
n_epochs = 200  
lr = 0.001
print(f'Training for {n_epochs} epochs...\n')

FloatTensor = torch.cuda.FloatTensor 
LongTensor = torch.cuda.LongTensor 

discriminator = Discriminator().cuda()
generator = Generator().cuda()
classifier = Classifier().cuda()

from torchsummary import summary
summary(generator,[(1,28,28),(32,10)])

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0, 0.2)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0, 0.2)
        nn.init.constant_(m.bias.data, 0.0)
discriminator.apply(weights_init_normal)
generator.apply(weights_init_normal)

adversarial_loss = nn.BCELoss()
task_loss = nn.CrossEntropyLoss()

optim_G = optim.Adam(itertools.chain(generator.parameters(), classifier.parameters()), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
optim_C = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))

scheduler_G = optim.lr_scheduler.StepLR(optim_G, step_size=200, gamma=0.95)
scheduler_D = optim.lr_scheduler.StepLR(optim_D, step_size=200, gamma=0.95)
scheduler_C = optim.lr_scheduler.StepLR(optim_C, step_size=200, gamma=0.95)



writer = tb_writer('logs')

for e in range(n_epochs):

    running_gen_loss = 0.0
    running_disc_loss = 0.0
    running_task_loss = 0.0

    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(train_source_dataloader, train_target_dataloader)):
        
        N = imgs_A.shape[0]

        imgs_A = imgs_A.type(FloatTensor)
        labels_A = labels_A.type(LongTensor)
        imgs_B = imgs_B.type(FloatTensor)
        
        truth_label = torch.zeros([32, 1]).cuda()
        false_label = torch.zeros([32, 1]).cuda()
        truth_label[:] = 1
        # --------------
        # Train Generator
        # --------------
        
        noise_z = FloatTensor(np.random.uniform(-1, 1, (32, 10)))
        
        optim_G.zero_grad()
        
        fake_img = generator(imgs_A, noise_z)
        pred = classifier(fake_img)
        lt = (task_loss(pred, labels_A) + task_loss(classifier(imgs_A), labels_A))/2
        gen_loss = 0.011*adversarial_loss(discriminator(fake_img), truth_label) + 0.01*lt
        gen_loss.backward()
        optim_G.step()
        scheduler_G.step()

        running_gen_loss += gen_loss.item()
        running_task_loss += 0.01*lt.item()

        # --------------
        # Train Discriminator
        # --------------
        
        optim_D.zero_grad()
        
        ld = adversarial_loss(discriminator(fake_img.detach()), false_label) + adversarial_loss(discriminator(imgs_B), truth_label)
        disc_loss = 0.13*ld
        disc_loss.backward()
        optim_D.step()
        scheduler_D.step()
        
        running_disc_loss += disc_loss.item()

        losses = [disc_loss.item(), gen_loss.item(), 0.01*lt.item()]

        # --------------
        # Logging
        # --------------
        if i %50 == 50-1:
            global_stepsize = e*min(len(train_source_dataloader),len(train_target_dataloader))+i
            sample_real, sample_fake = select_n_random(imgs_A.detach(), fake_img.detach())

            tblog.log_losses_tb(writer, losses, global_stepsize)
            tblog.log_comparisons_grid(writer, sample_real, sample_fake)
            tblog.log_predictions_grid(writer, classifier, imgs_B, global_stepsize)

            writer.close()
        
    
    print(f"[ Epoch #{e+1}/{n_epochs} ]\t[ Batch #{i}/{min(len(train_source_dataloader),len(train_target_dataloader))} ]\t[ Gen Loss: {running_gen_loss/N} ]\t[ Disc Loss: {running_disc_loss/N} ]\t[ Classifier Loss: {running_task_loss/N} ]")


